"""
Diffusion Policy – Gated Weighted Composition.

Combines the contact-gating mechanism from diffusion_policy_gated_composition
with the softmax weight predictor from diffusion_policy_composition.

Architecture
────────────
  scene_encoder      ObsGroupEncoder  (rgb + point_cloud + kinematics)
  force_encoder      ObsGroupEncoder  (torque/force + kinematics)
  neutral_embedding  Parameter        learnable fallback for force features when no contact
  scene_noise_net    ConditionalUnet1D  conditioned on scene features
  force_noise_net    ConditionalUnet1D  conditioned on (gated) force features
  weight_predictor   MLP → 2 logits → softmax weights [w_scene, w_force]

Forward pass (training)
───────────────────────
1. Encode scene and force observations.
2. Apply contact gate: replace force features with neutral embedding h*
   where phi = 0 (no contact), using ground-truth contact labels.
3. Predict softmax blending weights [w_scene, w_force] from combined
   (scene, gated_force) features.
4. Add noise to clean actions (forward diffusion).
5. Predict per-expert noise: eps_scene, eps_force.
6. Compose: eps = w_scene * eps_scene + w_force * eps_force.
7. MSE loss against the sampled noise.

Inference
─────────
phi is derived from the torque threshold on the current observation.
Encoding, weight prediction, and the denoising loop follow the same logic.
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

@register_algo_factory_func("diffusion_policy_gated_weighted_composition")
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
        return DiffusionPolicyUNetGatedWeightedComposition, {}
    elif algo_config.transformer.enabled:
        raise NotImplementedError()
    else:
        raise RuntimeError()


# ---------------------------------------------------------------------------
# Algorithm
# ---------------------------------------------------------------------------

class DiffusionPolicyUNetGatedWeightedComposition(PolicyAlgo):
    """
    Diffusion Policy with Contact-Gated Softmax-Weighted Expert Composition.

    Two specialist UNets (scene / force) whose noise predictions are blended
    by a softmax weight predictor, with the force expert's contribution
    explicitly zeroed out when no contact is detected.
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

        force_keys = [k for k in low_dim_keys if "torque" in k or "force" in k]
        kin_keys   = [k for k in low_dim_keys if k not in force_keys]

        scene_group_keys = rgb_keys + point_cloud_keys + kin_keys
        force_group_keys = force_keys + kin_keys

        assert len(scene_group_keys) > 0, (
            "scene group is empty – no rgb, point_cloud, or kinematics keys found")
        assert len(force_group_keys) > 0, (
            "force group is empty – no torque/force keys found")

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

        # Learnable neutral embedding h* – replaces force features when phi = 0.
        neutral_embedding = BaseNets.Parameter(torch.zeros(force_feat_dim))

        total_feat_dim = (scene_feat_dim + force_feat_dim) * To

        # Weight predictor: combined features → 2 logits → softmax [w_scene, w_force]
        weight_predictor = nn.Sequential(
            nn.Linear(total_feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

        nets = nn.ModuleDict({
            "policy": nn.ModuleDict({
                "scene_encoder":     scene_enc,
                "force_encoder":     force_enc,
                "neutral_embedding": neutral_embedding,
                "scene_noise_net":   scene_noise_net,
                "force_noise_net":   force_noise_net,
                "weight_predictor":  weight_predictor,
            })
        })
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
    def contact_force_threshold(self):
        gating_cfg = getattr(self.algo_config, "gating", None)
        return float(getattr(gating_cfg, "contact_force_threshold", 1.0))

    def _encode_obs(self, inputs, nets):
        """
        Run both encoders on the observation.

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
            phi         (Tensor): [B, 1, 1]
            nets (nn.ModuleDict): for neutral_embedding access

        Returns:
            gated_force (Tensor): [B, To, D_f]
        """
        h_star = nets["policy"]["neutral_embedding"]()  # [D_f]
        h_star = h_star.view(1, 1, -1)                  # [1, 1, D_f]
        return phi * force_feats + (1.0 - phi) * h_star

    def _compute_contact_phi(self, obs_dict, To):
        """
        Compute binary contact gate phi from raw observations at inference time.

        Returns:
            phi (Tensor): [B, 1, 1] float gate
        """
        torque_key = None
        for k in self._force_obs_keys:
            if "torque" in k and k in obs_dict:
                torque_key = k
                break
        if torque_key is None and len(self._force_obs_keys) > 0:
            torque_key = self._force_obs_keys[0]

        if torque_key is None or torque_key not in obs_dict:
            B = next(iter(obs_dict.values())).shape[0]
            return torch.zeros(B, 1, 1, device=self.device)

        torque_obs = obs_dict[torque_key]  # [B, T, D] or [B, D]
        if torque_obs.ndim == 3:
            current_torque = torque_obs[:, -1, -7:]
        else:
            current_torque = torque_obs[:, -7:]

        # Unnormalize torque if obs_normalization_stats are available so the
        # threshold is applied in the original physical units.
        norm_stats = getattr(self, "obs_normalization_stats", None)
        if norm_stats is not None and torque_key in norm_stats:
            stats = norm_stats[torque_key]
            offset = TensorUtils.to_float(TensorUtils.to_device(
                TensorUtils.to_tensor(stats["offset"]), self.device))  # [1, D]
            scale  = TensorUtils.to_float(TensorUtils.to_device(
                TensorUtils.to_tensor(stats["scale"]),  self.device))  # [1, D]
            offset = offset[..., -7:]
            scale  = scale[...,  -7:]
            current_torque = current_torque * scale + offset

        phi = (current_torque.abs().max(dim=-1).values > self.contact_force_threshold
               ).float().unsqueeze(1).unsqueeze(2)   # [B, 1, 1]
        return phi

    def _predict_weights(self, combined, nets):
        """
        Predict softmax blending weights from combined (scene, gated_force) features.

        Args:
            combined (Tensor): [B, total_feat_dim]
            nets (nn.ModuleDict): network dict

        Returns:
            w_scene (Tensor): [B, 1, 1]
            w_force (Tensor): [B, 1, 1]
        """
        logits = nets["policy"]["weight_predictor"](combined)  # [B, 2]
        soft = F.softmax(logits, dim=1)                        # [B, 2]
        w_s = soft[:, 0].view(-1, 1, 1)
        w_f = soft[:, 1].view(-1, 1, 1)
        return w_s, w_f

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def process_batch_for_training(self, batch):
        To = self.algo_config.horizon.observation_horizon
        Tp = self.algo_config.horizon.prediction_horizon

        input_batch = dict()
        input_batch["obs"]      = {k: batch["obs"][k][:, :To] for k in batch["obs"]}
        input_batch["goal_obs"] = batch.get("goal_obs", None)
        input_batch["actions"]  = batch["actions"][:, :Tp, :]

        if not self.action_check_done:
            actions = input_batch["actions"]
            if not torch.all((-1 <= actions) & (actions <= 1)).item():
                raise ValueError(
                    "'actions' must be in [-1, 1] for Diffusion Policy! "
                    "Check if hdf5_normalize_action is enabled."
                )
            self.action_check_done = True

        # Ground-truth contact label for gating supervision
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
        B = batch["actions"].shape[0]

        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = super(DiffusionPolicyUNetGatedWeightedComposition, self).train_on_batch(
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
            phi = batch["contact_label"].view(B, 1, 1)  # [B, 1, 1]
            gated_force = self._apply_contact_gate(force_feats, phi, self.nets)

            # Flatten to global conditioning vectors
            scene_cond = scene_feats.flatten(start_dim=1)   # [B, To*D_s]
            force_cond = gated_force.flatten(start_dim=1)   # [B, To*D_f]
            combined   = torch.cat([scene_cond, force_cond], dim=-1)

            # ── 3. Blending weights ────────────────────────────────────────
            w_scene, w_force = self._predict_weights(combined, self.nets)

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

            # ── 6. Weighted composition ────────────────────────────────────
            noise_pred = w_scene * eps_scene + w_force * eps_force

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
        log = super(DiffusionPolicyUNetGatedWeightedComposition, self).log_info(info)
        log["Loss"] = info["losses"]["total_loss"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def reset(self):
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        self.obs_queue    = deque(maxlen=To)
        self.action_queue = deque(maxlen=Ta)

    def get_action(self, obs_dict, goal_dict=None):
        if len(self.action_queue) == 0:
            action_sequence = self._get_action_trajectory(obs_dict=obs_dict)
            self.action_queue.extend(action_sequence[0])
        return self.action_queue.popleft().unsqueeze(0)

    def _get_action_trajectory(self, obs_dict, goal_dict=None):
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

        # ── Blending weights (computed once before denoising loop) ────────
        w_scene, w_force = self._predict_weights(combined, nets)

        print(f"[inference] phi (contact gate): {phi.squeeze().tolist()}")
        print(f"[inference] w_scene: {w_scene.squeeze().tolist()}, "
              f"w_force: {w_force.squeeze().tolist()}")

        # ── Log inference stats to file ───────────────────────────────────
        import os
        _log_path = "/home/jiuzl/robomimic_suite/temp/diffusion_policy_gated_weighted_composition_inference.txt"
        os.makedirs(os.path.dirname(_log_path), exist_ok=True)
        with open(_log_path, "a") as _f:
            _f.write(f"phi: {phi.squeeze().tolist()}\n")
            _f.write(f"w_scene: {w_scene.squeeze().tolist()}\n")
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

            noise_pred = w_scene * eps_scene + w_force * eps_force

            naction = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction,
            ).prev_sample

        start = To - 1
        end   = start + Ta
        return naction[:, start:end]

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def serialize(self):
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
# Utility functions
# ---------------------------------------------------------------------------

def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
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
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features // features_per_group,
            num_channels=x.num_features),
    )
    return root_module
