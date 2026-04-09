"""
Diffusion Policy – Gated Composition.

Combines two ideas from sibling algorithms:

  Gating (diffusion_policy_gating.py)
      Hard binary contact gate: when no contact is detected the force encoder's
      output is replaced by a learnable neutral embedding h*, preventing the
      force expert from being confused by noisy, uninformative torque readings.
      The gate is supervised by ground-truth contact labels during training and
      computed from a torque threshold during inference.

  Composition v2 (diffusion_policy_composition_v2.py)
      Two specialist UNets (scene / force) whose noise predictions are blended
      using a CFG-style formula:

          eps_final = eps_scene + w_force * (eps_force − eps_scene)

      A learned router predicts the (unbounded) guidance weight.

Combination – Contact-Gated Expert Composition
──────────────────────────────────────────────
The guidance weight is gated by the contact signal:

    learn_guidance_scale=False  →  w_force = phi * max_guidance_scale
    learn_guidance_scale=True   →  w_force = phi * softplus(guidance_logit)

where phi ∈ {0, 1} is the contact gate.

  phi = 0  (no contact) → w_force = 0 → eps_final = eps_scene  (pure vision)
  phi = 1  (contact)    → w_force = fixed or learned positive scale
                          → CFG-style blend with force expert

This gives the model:
  • A strong physics-informed inductive bias (gating) that is directly
    supervised, so the router does not need to discover the contact boundary
    from scratch.
  • Specialised denoisers that are never asked to share a single conditioning
    context (unlike the original gating approach which concatenates features).
  • Smooth, continuous blending during contact rather than a hard switch.

New config keys (all optional, with defaults):
  algo_config.gating.contact_force_threshold   (float, default 1.0)  [N or Nm]
  algo_config.max_guidance_scale               (float, default 2.0)
  algo_config.learn_guidance_scale             (bool,  default False)
      When True, a guidance_predictor MLP takes (scene_feats + force_feats)
      and outputs a positive scale via softplus, replacing max_guidance_scale.
"""

from typing import Callable
from collections import OrderedDict, deque
from packaging.version import parse as parse_version

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.training_utils import EMAModel

import robomimic.models.obs_nets as ObsNets
import robomimic.models.base_nets as BaseNets
import robomimic.models.diffusion_policy_nets as DPNets
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils

from robomimic.algo import register_algo_factory_func, PolicyAlgo


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

@register_algo_factory_func("diffusion_policy_gated_composition")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the algorithm class to instantiate.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): additional kwargs
    """
    if algo_config.unet.enabled:
        return DiffusionPolicyUNetGatedComposition, {}
    elif algo_config.transformer.enabled:
        return DiffusionPolicyTransformerGatedComposition, {}
    else:
        raise RuntimeError()


# ---------------------------------------------------------------------------
# Algorithm
# ---------------------------------------------------------------------------

class DiffusionPolicyUNetGatedComposition(PolicyAlgo):
    """
    Diffusion Policy with Contact-Gated Expert Composition.

    Networks
    --------
    scene_encoder      ObsGroupEncoder  (rgb + point_cloud + kinematics)
    force_encoder      ObsGroupEncoder  (torque/force + kinematics)
    neutral_embedding  Parameter        learnable fallback for force features
    scene_noise_net    ConditionalUnet1D  conditioned on scene features
    force_noise_net    ConditionalUnet1D  conditioned on gated force features
    weight_predictor   MLP → single guidance-scale logit

    Forward pass (training)
    -----------------------
    1. Encode scene and force observations.
    2. Apply contact gate: replace force features with h* when phi = 0.
    3. Predict guidance weight: w_force = phi * sigmoid(logit) * max_scale.
    4. Add noise to clean actions (forward diffusion).
    5. Predict per-expert noise: eps_scene, eps_force.
    6. Compose: eps = eps_scene + w_force * (eps_force − eps_scene).
    7. MSE loss against the sampled noise.

    Inference
    ---------
    phi is computed from the torque threshold on the current observation.
    Encoding, weight prediction, and the denoising loop follow the same logic.
    """

    # ------------------------------------------------------------------
    # Network construction
    # ------------------------------------------------------------------

    def _create_networks(self):
        encoder_kwargs = ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder)

        all_obs_keys = list(self.obs_shapes.keys())
        rgb_keys         = [k for k in all_obs_keys if ObsUtils.key_is_obs_modality(k, "rgb")]
        low_dim_keys     = [k for k in all_obs_keys if ObsUtils.key_is_obs_modality(k, "low_dim")]
        point_cloud_keys = [k for k in all_obs_keys if ObsUtils.key_is_obs_modality(k, "point_cloud")]

        # Split low-dim keys into torque/force vs kinematics
        force_keys = [k for k in low_dim_keys if "torque" in k or "force" in k]
        kin_keys   = [k for k in low_dim_keys if k not in force_keys]

        # Scene group: visual + kinematics
        scene_group_keys = rgb_keys + point_cloud_keys + kin_keys
        # Force group: force/torque + kinematics (shared kin gives each expert
        # the same proprioceptive context so their noise predictions are
        # comparable and can be meaningfully blended).
        force_group_keys = force_keys + kin_keys

        assert len(scene_group_keys) > 0, (
            "scene group is empty – no rgb, point_cloud, or kinematics keys found")
        assert len(force_group_keys) > 0, (
            "force group is empty – no torque/force keys found")

        # Store force keys for contact detection during inference
        self._force_obs_keys = force_keys

        To = self.algo_config.horizon.observation_horizon

        def _make_encoder_and_noise_net(keys):
            group_shapes = OrderedDict([(k, self.obs_shapes[k]) for k in keys])
            enc = ObsNets.ObservationGroupEncoder(
                observation_group_shapes=OrderedDict([("obs", group_shapes)]),
                encoder_kwargs=encoder_kwargs,
            )
            enc = replace_bn_with_gn(enc)
            feat_dim = enc.output_shape()[0]
            noise_net = DPNets.ConditionalUnet1D(
                input_dim=self.ac_dim,
                global_cond_dim=feat_dim * To,
            )
            return enc, noise_net, feat_dim

        scene_enc, scene_noise_net, scene_feat_dim = _make_encoder_and_noise_net(scene_group_keys)
        force_enc, force_noise_net, force_feat_dim = _make_encoder_and_noise_net(force_group_keys)

        # Learnable neutral embedding h* (same dim as force encoder output).
        # Replaces force features when no contact is detected.
        neutral_embedding = BaseNets.Parameter(torch.zeros(force_feat_dim))

        total_feat_dim = (scene_feat_dim + force_feat_dim) * To

        policy_nets = nn.ModuleDict({
            "scene_encoder":     scene_enc,
            "force_encoder":     force_enc,
            "neutral_embedding": neutral_embedding,
            "scene_noise_net":   scene_noise_net,
            "force_noise_net":   force_noise_net,
        })

        # Optional learned guidance scale: (scene + force) features → positive scale.
        # softplus(logit) ensures the output is strictly positive.
        if getattr(self.algo_config, "learn_guidance_scale", False):
            policy_nets["guidance_predictor"] = nn.Sequential(
                nn.Linear(total_feat_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )

        nets = nn.ModuleDict({"policy": policy_nets})
        nets = nets.float().to(self.device)

        # Noise scheduler
        noise_scheduler = None
        if self.algo_config.ddpm.enabled:
            noise_scheduler = DDPMScheduler(
                num_train_timesteps=self.algo_config.ddpm.num_train_timesteps,
                beta_schedule=self.algo_config.ddpm.beta_schedule,
                clip_sample=self.algo_config.ddpm.clip_sample,
                prediction_type=self.algo_config.ddpm.prediction_type,
            )
        elif self.algo_config.ddim.enabled:
            noise_scheduler = DDIMScheduler(
                num_train_timesteps=self.algo_config.ddim.num_train_timesteps,
                beta_schedule=self.algo_config.ddim.beta_schedule,
                clip_sample=self.algo_config.ddim.clip_sample,
                set_alpha_to_one=self.algo_config.ddim.set_alpha_to_one,
                steps_offset=self.algo_config.ddim.steps_offset,
                prediction_type=self.algo_config.ddim.prediction_type,
            )
        else:
            raise RuntimeError()

        ema = None
        if self.algo_config.ema.enabled:
            ema = EMAModel(model=nets, power=self.algo_config.ema.power)

        self.nets = nets
        self.noise_scheduler = noise_scheduler
        self.ema = ema
        self.action_check_done = False
        self.obs_queue = None
        self.action_queue = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def max_guidance_scale(self):
        return float(getattr(self.algo_config, "max_guidance_scale", 2.0))

    @property
    def learn_guidance_scale(self):
        return bool(getattr(self.algo_config, "learn_guidance_scale", False))

    @property
    def contact_force_threshold(self):
        gating_cfg = getattr(self.algo_config, "gating", None)
        return float(getattr(gating_cfg, "contact_force_threshold", 1.0))

    def _encode_obs(self, inputs, nets):
        """
        Run both encoders on the observation.

        Args:
            inputs (dict): {"obs": obs_dict, "goal": goal_dict}
            nets (nn.ModuleDict): self.nets or ema.averaged_model

        Returns:
            scene_feats (Tensor): [B, To, D_s]
            force_feats (Tensor): [B, To, D_f]
        """
        scene_feats = TensorUtils.time_distributed(
            inputs, nets["policy"]["scene_encoder"], inputs_as_kwargs=True)
        force_feats = TensorUtils.time_distributed(
            inputs, nets["policy"]["force_encoder"], inputs_as_kwargs=True)
        assert scene_feats.ndim == 3 and force_feats.ndim == 3  # [B, T, D]
        return scene_feats, force_feats

    def _apply_contact_gate(self, force_feats, phi, nets):
        """
        Replace force features with the neutral embedding where phi == 0.

        Args:
            force_feats (Tensor): [B, To, D_f]
            phi         (Tensor): [B, 1, 1]  contact gate (0 or 1 during training,
                                              0/1 float derived from threshold at inference)
            nets (nn.ModuleDict): network dict (for neutral_embedding access)

        Returns:
            gated_force (Tensor): [B, To, D_f]
        """
        h_star = nets["policy"]["neutral_embedding"]()  # [D_f]
        h_star = h_star.view(1, 1, -1)                  # [1, 1, D_f]
        return phi * force_feats + (1.0 - phi) * h_star

    def _compute_contact_phi(self, obs_dict, To):
        """
        Compute binary contact gate phi from raw observations at inference time.

        Reads the last 7 elements of robot0_joint_ext_torque (current step's
        joint torques) and compares their max absolute value to the threshold.

        Args:
            obs_dict (dict): observation dict with key "robot0_joint_ext_torque"
                             shaped [B, T, D] or [B, D]
            To (int): observation horizon

        Returns:
            phi (Tensor): [B, 1, 1] float gate
        """
        # Find the first available torque key (fall back gracefully)
        torque_key = None
        for k in self._force_obs_keys:
            if "torque" in k and k in obs_dict:
                torque_key = k
                break
        if torque_key is None and len(self._force_obs_keys) > 0:
            torque_key = self._force_obs_keys[0]

        if torque_key is None or torque_key not in obs_dict:
            # No torque key available → assume no contact
            B = next(iter(obs_dict.values())).shape[0]
            return torch.zeros(B, 1, 1, device=self.device)

        torque_obs = obs_dict[torque_key]  # [B, T, D] or [B, D]
        if torque_obs.ndim == 3:
            current_torque = torque_obs[:, -1, -7:]  # [B, 7]
        else:
            current_torque = torque_obs[:, -7:]      # [B, 7]

        # Unnormalize torque if obs_normalization_stats are available so the
        # threshold is applied in the original physical units.
        norm_stats = getattr(self, "obs_normalization_stats", None)
        if norm_stats is not None and torque_key in norm_stats:
            stats = norm_stats[torque_key]
            offset = TensorUtils.to_float(TensorUtils.to_device(
                TensorUtils.to_tensor(stats["offset"]), self.device))  # [1, D]
            scale  = TensorUtils.to_float(TensorUtils.to_device(
                TensorUtils.to_tensor(stats["scale"]),  self.device))  # [1, D]
            # offset/scale cover the full obs dim; slice the last 7 to match
            offset = offset[..., -7:]
            scale  = scale[...,  -7:]
            current_torque = current_torque * scale + offset

        phi = (current_torque.abs().max(dim=-1).values > self.contact_force_threshold
               ).float().unsqueeze(1).unsqueeze(2)   # [B, 1, 1]
        return phi

    def _guidance_weight(self, w, phi, apply_gate=True):
        """
        Apply the contact gate to a precomputed guidance scale.

            w_force = phi * w   (apply_gate=True, default)
            w_force = w         (apply_gate=False)

        Args:
            w          (Tensor | float): [B, 1, 1] or scalar guidance scale.
            phi        (Tensor): [B, 1, 1] contact gate (0 or 1).
            apply_gate (bool): if True (default), zero out w_force when phi=0.

        Returns:
            w_force (Tensor): [B, 1, 1]
        """
        return phi * w if apply_gate else w

    def _predict_guidance_scale(self, combined, nets):
        """
        Compute the guidance scale w for _guidance_weight.

        If learn_guidance_scale is True, runs the guidance_predictor MLP and
        applies softplus so the output is strictly positive.  Otherwise
        returns the fixed max_guidance_scale scalar from config.

        Args:
            combined (Tensor): [B, total_feat_dim] concatenated scene+force cond.
            nets (nn.ModuleDict): self.nets or ema.averaged_model

        Returns:
            w (Tensor | float): [B, 1, 1] or scalar
        """
        if self.learn_guidance_scale:
            raw = nets["policy"]["guidance_predictor"](combined)  # [B, 1]
            return F.softplus(raw).view(-1, 1, 1)
        return self.max_guidance_scale

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def process_batch_for_training(self, batch):
        """
        Filter and prepare a batch for training.

        Args:
            batch (dict): raw batch from the data loader

        Returns:
            input_batch (dict): processed batch
        """
        To = self.algo_config.horizon.observation_horizon
        Tp = self.algo_config.horizon.prediction_horizon

        input_batch = dict()
        input_batch["obs"]      = {k: batch["obs"][k][:, :To] for k in batch["obs"]}
        input_batch["goal_obs"] = batch.get("goal_obs", None)
        input_batch["actions"]  = batch["actions"][:, :Tp, :]  # [B, Tp, Da]

        if not self.action_check_done:
            actions = input_batch["actions"]
            if not torch.all((-1 <= actions) & (actions <= 1)).item():
                raise ValueError(
                    "'actions' must be in [-1, 1] for Diffusion Policy! "
                    "Check if hdf5_normalize_action is enabled."
                )
            self.action_check_done = True

        # Ground-truth contact label for gating supervision.
        # robot0_joint_ext_torque: [B, To, D] where the last 7 cols are
        # current joint torques.  Contact if max abs torque > threshold.
        contact_force_thres = self.contact_force_threshold
        torque_key = None
        for k in self._force_obs_keys:
            if "torque" in k and k in batch["obs"]:
                torque_key = k
                break
        if torque_key is None and len(self._force_obs_keys) > 0:
            torque_key = self._force_obs_keys[0]

        if torque_key is not None and torque_key in batch["obs"]:
            current_torque = batch["obs"][torque_key][:, To - 1, -7:]  # [B, 7]
            contact_label = (
                current_torque.abs().max(dim=-1).values > contact_force_thres
            ).float().unsqueeze(-1)  # [B, 1]
        else:
            B = batch["actions"].shape[0]
            contact_label = torch.zeros(B, 1)

        input_batch["contact_label"] = contact_label

        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)

    def train_on_batch(self, batch, epoch, validate=False):
        """
        Train / validate on one batch.

        Args:
            batch    (dict): processed batch from @process_batch_for_training
            epoch    (int):  current epoch
            validate (bool): skip parameter updates if True

        Returns:
            info (dict): losses and diagnostics
        """
        B = batch["actions"].shape[0]

        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = super(DiffusionPolicyUNetGatedComposition, self).train_on_batch(
                batch, epoch, validate=validate)
            actions = batch["actions"]

            inputs = {
                "obs":  batch["obs"],
                "goal": batch["goal_obs"],
            }
            for k in self.obs_shapes:
                assert inputs["obs"][k].ndim - 2 == len(self.obs_shapes[k])

            # ── 1. Encode ─────────────────────────────────────────────────
            scene_feats, force_feats = self._encode_obs(inputs, self.nets)

            # ── 2. Gated force features (GT phi) ──────────────────────────
            # phi from ground-truth contact labels: [B, 1] → [B, 1, 1]
            phi = batch["contact_label"].view(B, 1, 1)
            gated_force = self._apply_contact_gate(force_feats, phi, self.nets)

            # Flatten to global conditioning vectors
            scene_cond = scene_feats.flatten(start_dim=1)   # [B, To*D_s]
            force_cond = gated_force.flatten(start_dim=1)   # [B, To*D_f]
            combined   = torch.cat([scene_cond, force_cond], dim=-1)

            # ── 3. Guidance weight ─────────────────────────────────────────
            w_force = self._guidance_weight(
                self._predict_guidance_scale(combined, self.nets), phi)  # [B, 1, 1]

            # ── 4. Forward diffusion ───────────────────────────────────────
            noise = torch.randn(actions.shape, device=self.device)
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,
                (B,), device=self.device,
            ).long()
            noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)

            # ── 5. Per-expert noise predictions ───────────────────────────
            eps_scene = self.nets["policy"]["scene_noise_net"](
                noisy_actions, timesteps, global_cond=scene_cond)
            eps_force = self.nets["policy"]["force_noise_net"](
                noisy_actions, timesteps, global_cond=force_cond)

            # ── 6. CFG-style composition ───────────────────────────────────
            # When w_force = 0 (no contact): eps = eps_scene
            # When w_force > 0 (contact):    eps blends toward force expert
            noise_pred = eps_scene + w_force * (eps_force - eps_scene)

            # ── 7. Loss ────────────────────────────────────────────────────
            total_loss = F.mse_loss(noise_pred, noise)

            losses = {"total_loss": total_loss}
            info["losses"] = TensorUtils.detach(losses)

            if not validate:
                policy_grad_norms = TorchUtils.backprop_for_loss(
                    net=self.nets,
                    optim=self.optimizers["policy"],
                    loss=total_loss,
                )
                if self.ema is not None:
                    self.ema.step(self.nets)
                info.update({"policy_grad_norms": policy_grad_norms})

        return info

    def log_info(self, info):
        """
        Process info dict for tensorboard logging.

        Args:
            info (dict): dict from @train_on_batch

        Returns:
            loss_log (dict): name → scalar
        """
        log = super(DiffusionPolicyUNetGatedComposition, self).log_info(info)
        log["Loss"] = info["losses"]["total_loss"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def reset(self):
        """Reset state for a new rollout."""
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        self.obs_queue    = deque(maxlen=To)
        self.action_queue = deque(maxlen=Ta)

    def get_action(self, obs_dict, goal_dict=None):
        """
        Return the next action for the current observation.

        Args:
            obs_dict  (dict): current observation, each value [1, D]
            goal_dict (dict): optional goal

        Returns:
            action (Tensor): [1, Da]
        """
        if len(self.action_queue) == 0:
            action_sequence = self._get_action_trajectory(obs_dict=obs_dict)
            self.action_queue.extend(action_sequence[0])
        return self.action_queue.popleft().unsqueeze(0)

    def _get_action_trajectory(self, obs_dict, goal_dict=None):
        """
        Run the full DDPM/DDIM denoising loop and return the action chunk.

        Args:
            obs_dict  (dict): current observation
            goal_dict (dict): optional goal

        Returns:
            naction (Tensor): [B, Ta, Da] action chunk
        """
        assert not self.nets.training

        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        Tp = self.algo_config.horizon.prediction_horizon
        action_dim = self.ac_dim

        if self.algo_config.ddpm.enabled:
            num_inference_timesteps = self.algo_config.ddpm.num_inference_timesteps
        elif self.algo_config.ddim.enabled:
            num_inference_timesteps = self.algo_config.ddim.num_inference_timesteps
        else:
            raise ValueError

        nets = self.ema.averaged_model if self.ema is not None else self.nets

        inputs = {"obs": obs_dict, "goal": goal_dict}
        for k in self.obs_shapes:
            # Add time dimension if frame stacking is not active
            if inputs["obs"][k].ndim - 1 == len(self.obs_shapes[k]):
                inputs["obs"][k] = inputs["obs"][k].unsqueeze(1)
            assert inputs["obs"][k].ndim - 2 == len(self.obs_shapes[k])

        # ── Encode observations ───────────────────────────────────────────
        scene_feats, force_feats = self._encode_obs(inputs, nets)
        B = scene_feats.shape[0]

        # ── Contact gate from observed torque ─────────────────────────────
        phi = self._compute_contact_phi(inputs["obs"], To)  # [B, 1, 1]

        # ── Gated force features ──────────────────────────────────────────
        gated_force = self._apply_contact_gate(force_feats, phi, nets)

        scene_cond = scene_feats.flatten(start_dim=1)
        force_cond = gated_force.flatten(start_dim=1)
        combined   = torch.cat([scene_cond, force_cond], dim=-1)

        # ── Guidance weight ───────────────────────────────────────────────
        guidance_scale = self._predict_guidance_scale(combined, nets)
        w_force = self._guidance_weight(guidance_scale, phi)  # [B, 1, 1]

        if self.learn_guidance_scale:
            print(f"[inference] learned guidance scale: {guidance_scale.squeeze().tolist()}")
        print(f"[inference] phi (contact gate): {phi.squeeze().tolist()}")
        print(f"[inference] w_force: {w_force.squeeze().tolist()}")

        # ── Log inference stats to file ───────────────────────────────────
        import os
        _log_path = "/home/jiuzl/robomimic_suite/temp/diffusion_policy_gated_composition_inference.txt"
        os.makedirs(os.path.dirname(_log_path), exist_ok=True)
        with open(_log_path, "a") as _f:
            if self.learn_guidance_scale:
                _f.write(f"guidance_scale: {guidance_scale.squeeze().tolist()}\n")
            _f.write(f"phi: {phi.squeeze().tolist()}\n")
            _f.write(f"w_force: {w_force.squeeze().tolist()}\n")
            _f.write(f"gated_force: {gated_force.squeeze().tolist()}\n")
            _f.write("---\n")

        # ── Denoising loop ────────────────────────────────────────────────
        naction = torch.randn((B, Tp, action_dim), device=self.device)
        self.noise_scheduler.set_timesteps(num_inference_timesteps)

        for k in self.noise_scheduler.timesteps:
            eps_scene = nets["policy"]["scene_noise_net"](
                sample=naction, timestep=k, global_cond=scene_cond)
            eps_force = nets["policy"]["force_noise_net"](
                sample=naction, timestep=k, global_cond=force_cond)

            # CFG-style composition with contact gate
            noise_pred = eps_scene + w_force * (eps_force - eps_scene)

            naction = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction,
            ).prev_sample

        # Return the executed action slice
        start = To - 1
        end   = start + Ta
        return naction[:, start:end]

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def serialize(self):
        """Return dict of current model parameters for checkpointing."""
        return {
            "nets": self.nets.state_dict(),
            "optimizers": {k: self.optimizers[k].state_dict() for k in self.optimizers},
            "lr_schedulers": {
                k: self.lr_schedulers[k].state_dict() if self.lr_schedulers[k] is not None else None
                for k in self.lr_schedulers
            },
            "ema": self.ema.averaged_model.state_dict() if self.ema is not None else None,
        }

    def deserialize(self, model_dict, load_optimizers=False):
        """
        Load model from a checkpoint.

        Args:
            model_dict      (dict): dict saved by self.serialize()
            load_optimizers (bool): also restore optimizer / lr_scheduler states
        """
        self.nets.load_state_dict(model_dict["nets"])

        if "optimizers" not in model_dict:
            model_dict["optimizers"] = {}
        if "lr_schedulers" not in model_dict:
            model_dict["lr_schedulers"] = {}

        if model_dict.get("ema", None) is not None:
            self.ema.averaged_model.load_state_dict(model_dict["ema"])

        if load_optimizers:
            for k in model_dict["optimizers"]:
                self.optimizers[k].load_state_dict(model_dict["optimizers"][k])
            for k in model_dict["lr_schedulers"]:
                if model_dict["lr_schedulers"][k] is not None:
                    self.lr_schedulers[k].load_state_dict(model_dict["lr_schedulers"][k])


# ---------------------------------------------------------------------------
# Transformer variant
# ---------------------------------------------------------------------------

class DiffusionPolicyTransformerGatedComposition(DiffusionPolicyUNetGatedComposition):
    """
    Transformer variant of DiffusionPolicyUNetGatedComposition.

    Replaces the two ConditionalUnet1D noise nets with TransformerForDiffusion
    networks that consume observation tokens directly as (B, To, D) sequences
    rather than a flattened global conditioning vector.

    Everything else (gating, CFG composition, guidance weight, EMA, scheduler)
    is inherited unchanged.
    """

    def _create_networks(self):
        encoder_kwargs = ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder)

        all_obs_keys = list(self.obs_shapes.keys())
        rgb_keys         = [k for k in all_obs_keys if ObsUtils.key_is_obs_modality(k, "rgb")]
        low_dim_keys     = [k for k in all_obs_keys if ObsUtils.key_is_obs_modality(k, "low_dim")]
        point_cloud_keys = [k for k in all_obs_keys if ObsUtils.key_is_obs_modality(k, "point_cloud")]

        force_keys = [k for k in low_dim_keys if "torque" in k or "force" in k]
        kin_keys   = [k for k in low_dim_keys if k not in force_keys]

        scene_group_keys = rgb_keys + point_cloud_keys + kin_keys
        force_group_keys = force_keys + kin_keys

        assert len(scene_group_keys) > 0, "scene group is empty"
        assert len(force_group_keys) > 0, "force group is empty"

        self._force_obs_keys = force_keys

        To = self.algo_config.horizon.observation_horizon
        Tp = self.algo_config.horizon.prediction_horizon
        transformer_cfg = self.algo_config.transformer

        def _make_encoder_and_noise_net(keys):
            group_shapes = OrderedDict([(k, self.obs_shapes[k]) for k in keys])
            enc = ObsNets.ObservationGroupEncoder(
                observation_group_shapes=OrderedDict([("obs", group_shapes)]),
                encoder_kwargs=encoder_kwargs,
            )
            enc = replace_bn_with_gn(enc)
            feat_dim = enc.output_shape()[0]
            noise_net = DPNets.TransformerForDiffusion(
                input_dim=self.ac_dim,
                output_dim=self.ac_dim,
                horizon=Tp,
                obs_dim=feat_dim,
                obs_horizon=To,
                n_layer=transformer_cfg.n_layer,
                n_head=transformer_cfg.n_head,
                n_emb=transformer_cfg.n_emb,
                p_drop_emb=transformer_cfg.p_drop_emb,
                p_drop_attn=transformer_cfg.p_drop_attn,
            )
            return enc, noise_net, feat_dim

        scene_enc, scene_noise_net, scene_feat_dim = _make_encoder_and_noise_net(scene_group_keys)
        force_enc, force_noise_net, force_feat_dim = _make_encoder_and_noise_net(force_group_keys)

        neutral_embedding = BaseNets.Parameter(torch.zeros(force_feat_dim))

        total_feat_dim = (scene_feat_dim + force_feat_dim) * To

        policy_nets = nn.ModuleDict({
            "scene_encoder":     scene_enc,
            "force_encoder":     force_enc,
            "neutral_embedding": neutral_embedding,
            "scene_noise_net":   scene_noise_net,
            "force_noise_net":   force_noise_net,
        })

        if getattr(self.algo_config, "learn_guidance_scale", False):
            policy_nets["guidance_predictor"] = nn.Sequential(
                nn.Linear(total_feat_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )

        nets = nn.ModuleDict({"policy": policy_nets})
        nets = nets.float().to(self.device)

        noise_scheduler = None
        if self.algo_config.ddpm.enabled:
            noise_scheduler = DDPMScheduler(
                num_train_timesteps=self.algo_config.ddpm.num_train_timesteps,
                beta_schedule=self.algo_config.ddpm.beta_schedule,
                clip_sample=self.algo_config.ddpm.clip_sample,
                prediction_type=self.algo_config.ddpm.prediction_type,
            )
        elif self.algo_config.ddim.enabled:
            noise_scheduler = DDIMScheduler(
                num_train_timesteps=self.algo_config.ddim.num_train_timesteps,
                beta_schedule=self.algo_config.ddim.beta_schedule,
                clip_sample=self.algo_config.ddim.clip_sample,
                set_alpha_to_one=self.algo_config.ddim.set_alpha_to_one,
                steps_offset=self.algo_config.ddim.steps_offset,
                prediction_type=self.algo_config.ddim.prediction_type,
            )
        else:
            raise RuntimeError()

        ema = None
        if self.algo_config.ema.enabled:
            ema = EMAModel(model=nets, power=self.algo_config.ema.power)

        self.nets = nets
        self.noise_scheduler = noise_scheduler
        self.ema = ema
        self.action_check_done = False
        self.obs_queue = None
        self.action_queue = None

    def train_on_batch(self, batch, epoch, validate=False):
        """
        Same composition logic as the UNet variant but noise nets receive
        obs_cond as (B, To, D) token sequences instead of flattened vectors.
        """
        B = batch["actions"].shape[0]

        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = super(DiffusionPolicyUNetGatedComposition, self).train_on_batch(
                batch, epoch, validate=validate)
            actions = batch["actions"]

            inputs = {"obs": batch["obs"], "goal": batch["goal_obs"]}
            for k in self.obs_shapes:
                assert inputs["obs"][k].ndim - 2 == len(self.obs_shapes[k])

            # ── 1. Encode ─────────────────────────────────────────────────
            scene_feats, force_feats = self._encode_obs(inputs, self.nets)
            # scene_feats: (B, To, D_s),  force_feats: (B, To, D_f)

            # ── 2. Gated force features (GT phi) ──────────────────────────
            phi = batch["contact_label"].view(B, 1, 1)
            gated_force = self._apply_contact_gate(force_feats, phi, self.nets)

            # Flatten only for the guidance predictor MLP
            scene_cond_flat = scene_feats.flatten(start_dim=1)
            force_cond_flat = gated_force.flatten(start_dim=1)
            combined = torch.cat([scene_cond_flat, force_cond_flat], dim=-1)

            # ── 3. Guidance weight ─────────────────────────────────────────
            w_force = self._guidance_weight(
                self._predict_guidance_scale(combined, self.nets), phi)

            # ── 4. Forward diffusion ───────────────────────────────────────
            noise = torch.randn(actions.shape, device=self.device)
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,
                (B,), device=self.device,
            ).long()
            noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)

            # ── 5. Per-expert noise predictions (obs tokens, not flat cond) ─
            eps_scene = self.nets["policy"]["scene_noise_net"](
                noisy_actions, timesteps, obs_cond=scene_feats)
            eps_force = self.nets["policy"]["force_noise_net"](
                noisy_actions, timesteps, obs_cond=gated_force)

            # ── 6. CFG-style composition ───────────────────────────────────
            noise_pred = eps_scene + w_force * (eps_force - eps_scene)

            # ── 7. Loss ────────────────────────────────────────────────────
            total_loss = F.mse_loss(noise_pred, noise)
            losses = {"total_loss": total_loss}
            info["losses"] = TensorUtils.detach(losses)

            if not validate:
                policy_grad_norms = TorchUtils.backprop_for_loss(
                    net=self.nets,
                    optim=self.optimizers["policy"],
                    loss=total_loss,
                )
                if self.ema is not None:
                    self.ema.step(self.nets)
                info.update({"policy_grad_norms": policy_grad_norms})

        return info

    def _get_action_trajectory(self, obs_dict, goal_dict=None):
        """
        Denoising loop using transformer noise nets.
        """
        assert not self.nets.training

        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        Tp = self.algo_config.horizon.prediction_horizon
        action_dim = self.ac_dim

        if self.algo_config.ddpm.enabled:
            num_inference_timesteps = self.algo_config.ddpm.num_inference_timesteps
        elif self.algo_config.ddim.enabled:
            num_inference_timesteps = self.algo_config.ddim.num_inference_timesteps
        else:
            raise ValueError

        nets = self.ema.averaged_model if self.ema is not None else self.nets

        inputs = {"obs": obs_dict, "goal": goal_dict}
        for k in self.obs_shapes:
            if inputs["obs"][k].ndim - 1 == len(self.obs_shapes[k]):
                inputs["obs"][k] = inputs["obs"][k].unsqueeze(1)
            assert inputs["obs"][k].ndim - 2 == len(self.obs_shapes[k])

        # ── Encode ────────────────────────────────────────────────────────
        scene_feats, force_feats = self._encode_obs(inputs, nets)
        # (B, To, D_s) and (B, To, D_f)
        B = scene_feats.shape[0]

        # ── Contact gate ──────────────────────────────────────────────────
        phi = self._compute_contact_phi(inputs["obs"], To)
        gated_force = self._apply_contact_gate(force_feats, phi, nets)

        # Flatten only for guidance predictor
        combined = torch.cat(
            [scene_feats.flatten(start_dim=1), gated_force.flatten(start_dim=1)], dim=-1)
        guidance_scale = self._predict_guidance_scale(combined, nets)
        w_force = self._guidance_weight(guidance_scale, phi)

        # ── Denoising loop ────────────────────────────────────────────────
        naction = torch.randn((B, Tp, action_dim), device=self.device)
        self.noise_scheduler.set_timesteps(num_inference_timesteps)

        for k in self.noise_scheduler.timesteps:
            eps_scene = nets["policy"]["scene_noise_net"](
                sample=naction, timestep=k, obs_cond=scene_feats)
            eps_force = nets["policy"]["force_noise_net"](
                sample=naction, timestep=k, obs_cond=gated_force)

            noise_pred = eps_scene + w_force * (eps_force - eps_scene)
            naction = self.noise_scheduler.step(
                model_output=noise_pred, timestep=k, sample=naction).prev_sample

        start = To - 1
        end   = start + Ta
        return naction[:, start:end]


# ---------------------------------------------------------------------------
# Utility functions (shared with other diffusion policy variants)
# ---------------------------------------------------------------------------

def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with func(module).
    """
    if predicate(root_module):
        return func(root_module)

    if parse_version(torch.__version__) < parse_version("1.9.0"):
        raise ImportError("This function requires pytorch >= 1.9.0")

    bn_list = [k.split(".") for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule(".".join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    bn_list = [k.split(".") for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module


def replace_bn_with_gn(root_module: nn.Module, features_per_group: int = 16) -> nn.Module:
    """Replace all BatchNorm2d layers with GroupNorm."""
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features // features_per_group,
            num_channels=x.num_features),
    )
    return root_module
