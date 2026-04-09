"""
Enhanced Diffusion Policy Composition v2.

Two algorithmic improvements over the baseline DiffusionPolicyUNetComposition:

Improvement 2 – CFG-Style Modality Guidance
    Instead of a weighted average of noise predictions, we treat the vision expert
    as a baseline and the force expert as a guidance direction:

        eps_final = eps_vision + w_force * (eps_force - eps_vision)

    This is mathematically identical to Classifier-Free Guidance (CFG).  When
    w_force ∈ [0, 1] the result is an interpolation (equivalent to the original
    weighted sum).  When w_force > 1 the force direction is *extrapolated*,
    amplifying haptic feedback even beyond what the router weight alone would
    express.  The router outputs a single unconstrained logit that is mapped to
    [0, max_guidance_scale] via sigmoid, allowing it to exceed 1.

Improvement 4 – Variance-Based Confidence Routing
    During inference the algorithm tracks per-expert x0 estimates at every
    denoising step.  High variance in the x0 history indicates the expert is
    uncertain (e.g. vision is confused by two visually identical objects).
    When the ratio of image-expert variance to torque-expert variance exceeds a
    configurable threshold the guidance weight is hard-switched to 1.0 (pure
    force), overriding the learned router weight.

    An optional training auxiliary loss teaches the router to anticipate this
    variance-based override: for K noise realizations at a fixed timestep the x0
    variance of each expert is estimated (with stop-gradient) and used to
    supervise the guidance weight toward a variance-proportional target.
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
import robomimic.models.diffusion_policy_nets as DPNets
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils

from robomimic.algo import register_algo_factory_func, PolicyAlgo
from robomimic.algo.diffusion_policy_composition import replace_bn_with_gn


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

@register_algo_factory_func("diffusion_policy_composition_v2")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the BC algo class to instantiate.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): additional kwargs
    """
    if algo_config.unet.enabled:
        return DiffusionPolicyUNetCompositionV2, {}
    elif algo_config.transformer.enabled:
        raise NotImplementedError()
    else:
        raise RuntimeError()


# ---------------------------------------------------------------------------
# Algorithm
# ---------------------------------------------------------------------------

class DiffusionPolicyUNetCompositionV2(PolicyAlgo):
    """
    Diffusion Policy Composition with CFG-style guidance and variance-based
    confidence routing.

    Architecture differences from DiffusionPolicyUNetComposition
    ─────────────────────────────────────────────────────────────
    • The weight_predictor now outputs a *single* logit that is converted to a
      guidance scale w_force ∈ [0, max_guidance_scale] via sigmoid scaling.
    • Composition formula: eps_final = eps_vis + w_force*(eps_tor − eps_vis)
      (CFG style, not a normalized weighted sum).
    • At inference, per-expert x0 estimates are accumulated over a sliding window
      of denoising steps; if the image-to-torque variance ratio exceeds
      `inference_threshold`, w_force is hard-set to 1.0 for that denoising step.

    New config keys (all optional, fall back to defaults)
    ─────────────────────────────────────────────────────
    algo_config.max_guidance_scale            (float, default 2.0)
    algo_config.variance_routing.enabled      (bool,  default False)
    algo_config.variance_routing.K_samples    (int,   default 4)
    algo_config.variance_routing.lambda_var   (float, default 0.1)
    algo_config.variance_routing.inference_threshold (float, default 5.0)
    algo_config.variance_routing.inference_window    (int,   default 5)
    """

    # ------------------------------------------------------------------
    # Network construction
    # ------------------------------------------------------------------

    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.

        Architecture:
          - image_encoder    : ObservationGroupEncoder (rgb + kinematics)
          - torque_encoder   : ObservationGroupEncoder (torque/force + kinematics)
          - image_noise_net  : ConditionalUnet1D conditioned on image_encoder
          - torque_noise_net : ConditionalUnet1D conditioned on torque_encoder
          - weight_predictor : MLP → single guidance-scale logit
        """
        encoder_kwargs = ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder)

        all_obs_keys = list(self.obs_shapes.keys())
        rgb_keys = [k for k in all_obs_keys if ObsUtils.key_is_obs_modality(k, "rgb")]
        low_dim_keys = [k for k in all_obs_keys if ObsUtils.key_is_obs_modality(k, "low_dim")]
        point_cloud_keys = [k for k in all_obs_keys if ObsUtils.key_is_obs_modality(k, "point_cloud")]

        torque_keys = [k for k in low_dim_keys if "torque" in k or "force" in k]
        kin_keys = [k for k in low_dim_keys if k not in torque_keys]

        image_group_keys = rgb_keys + point_cloud_keys + kin_keys
        torque_group_keys = torque_keys + kin_keys

        assert len(image_group_keys) > 0, (
            "image group is empty (no rgb, point_cloud, or kinematics keys found)")
        assert len(torque_group_keys) > 0, (
            "torque group is empty (no torque/force keys found)")

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
                global_cond_dim=feat_dim * self.algo_config.horizon.observation_horizon,
            )
            return enc, noise_net, feat_dim

        image_enc, image_noise_net, image_feat_dim = _make_encoder_and_noise_net(image_group_keys)
        torque_enc, torque_noise_net, torque_feat_dim = _make_encoder_and_noise_net(torque_group_keys)

        To = self.algo_config.horizon.observation_horizon
        total_feat_dim = (image_feat_dim + torque_feat_dim) * To

        # Guidance-scale predictor: outputs a single unconstrained logit.
        # sigmoid(logit) * max_guidance_scale maps it to [0, max_guidance_scale].
        weight_predictor = nn.Sequential(
            nn.Linear(total_feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        nets = nn.ModuleDict({
            "policy": nn.ModuleDict({
                "image_encoder": image_enc,
                "torque_encoder": torque_enc,
                "image_noise_net": image_noise_net,
                "torque_noise_net": torque_noise_net,
                "weight_predictor": weight_predictor,
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
    # Helper properties / small utilities
    # ------------------------------------------------------------------

    @property
    def max_guidance_scale(self):
        """Maximum value the guidance weight can reach (config or default 2.0)."""
        return float(getattr(self.algo_config, "max_guidance_scale", 2.0))

    def _guidance_weight(self, logit):
        """
        Map a raw logit to a guidance scale in [0, max_guidance_scale].

        Args:
            logit (Tensor): [..., 1] unconstrained logit from weight_predictor

        Returns:
            w (Tensor): same shape, values in (0, max_guidance_scale)
        """
        return torch.sigmoid(logit) * self.max_guidance_scale

    def _compute_x0_from_eps(self, x_t, eps, t):
        """
        Recover the clean-action estimate (x0) from the epsilon prediction.

            x0 = (x_t - sqrt(1 - alpha_bar_t) * eps) / sqrt(alpha_bar_t)

        Works for both scalar timesteps (inference loop) and batched timesteps
        (training, shape [B]).

        Args:
            x_t  (Tensor): [B, Tp, Da] noisy sample
            eps  (Tensor): [B, Tp, Da] epsilon prediction
            t    (Tensor): scalar or [B] long timestep indices

        Returns:
            x0 (Tensor): [B, Tp, Da] clean-action estimate, clamped to [-1, 1]
        """
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(self.device)
        t = t.long()
        if t.dim() == 0:
            alpha_t = alphas_cumprod[t].view(1, 1, 1)
        else:
            alpha_t = alphas_cumprod[t].view(-1, 1, 1)  # [B, 1, 1]
        x0 = (x_t - (1.0 - alpha_t).sqrt() * eps) / (alpha_t.sqrt() + 1e-8)
        return x0.clamp(-1.0, 1.0)

    def _encode_obs(self, inputs, nets):
        """
        Encode observations through both group encoders.

        Args:
            inputs (dict): {"obs": obs_dict, "goal": goal_dict}
            nets (nn.ModuleDict): network dict (self.nets or ema.averaged_model)

        Returns:
            image_cond   (Tensor): [B, To * D_img]
            torque_cond  (Tensor): [B, To * D_tor]
            obs_cond_all (Tensor): [B, To * (D_img + D_tor)]
        """
        image_feats = TensorUtils.time_distributed(
            inputs, nets["policy"]["image_encoder"], inputs_as_kwargs=True)
        torque_feats = TensorUtils.time_distributed(
            inputs, nets["policy"]["torque_encoder"], inputs_as_kwargs=True)
        assert image_feats.ndim == 3 and torque_feats.ndim == 3  # [B, T, D]
        image_cond = image_feats.flatten(start_dim=1)
        torque_cond = torque_feats.flatten(start_dim=1)
        obs_cond_all = torch.cat([image_cond, torque_cond], dim=-1)
        return image_cond, torque_cond, obs_cond_all

    # ------------------------------------------------------------------
    # Variance estimation (used for aux loss and inference routing)
    # ------------------------------------------------------------------

    def _estimate_x0_variances(self, actions, timesteps, image_cond, torque_cond, K=4):
        """
        Estimate per-sample x0 variance for each expert by sampling K different
        noise vectors at the given (fixed) timesteps.

        Concretely, for each of K independent noise draws we compute the noisy
        sample, run each expert, recover x0, then measure variance across the K
        draws.  High variance means the expert's clean-action estimate is
        unstable with respect to noise – a proxy for ambiguous observations.

        This method does NOT set up gradients (intended to be called inside
        torch.no_grad() for the aux-loss target computation).

        Args:
            actions    (Tensor): [B, Tp, Da] clean actions
            timesteps  (Tensor): [B] timestep indices
            image_cond (Tensor): [B, D_img]
            torque_cond(Tensor): [B, D_tor]
            K          (int):   number of noise samples

        Returns:
            image_var  (Tensor): [B] per-sample variance for image expert
            torque_var (Tensor): [B] per-sample variance for torque expert
        """
        image_x0_list = []
        torque_x0_list = []
        for _ in range(K):
            noise_k = torch.randn_like(actions)
            noisy_k = self.noise_scheduler.add_noise(actions, noise_k, timesteps)

            eps_img = self.nets["policy"]["image_noise_net"](
                noisy_k, timesteps, global_cond=image_cond)
            eps_tor = self.nets["policy"]["torque_noise_net"](
                noisy_k, timesteps, global_cond=torque_cond)

            image_x0_list.append(self._compute_x0_from_eps(noisy_k, eps_img, timesteps))
            torque_x0_list.append(self._compute_x0_from_eps(noisy_k, eps_tor, timesteps))

        # [K, B, Tp, Da] → variance over K → mean over (Tp, Da) → [B]
        image_var = torch.stack(image_x0_list).var(dim=0).mean(dim=[-1, -2])
        torque_var = torch.stack(torque_x0_list).var(dim=0).mean(dim=[-1, -2])
        return image_var, torque_var

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader.

        Args:
            batch (dict): raw batch from data loader

        Returns:
            input_batch (dict): filtered batch ready for training
        """
        To = self.algo_config.horizon.observation_horizon
        Tp = self.algo_config.horizon.prediction_horizon

        input_batch = dict()
        input_batch["obs"] = {k: batch["obs"][k][:, :To] for k in batch["obs"]}
        input_batch["goal_obs"] = batch.get("goal_obs", None)
        input_batch["actions"] = batch["actions"][:, :Tp, :]

        if not self.action_check_done:
            actions = input_batch["actions"]
            if not torch.all((-1 <= actions) & (actions <= 1)).item():
                raise ValueError(
                    "'actions' must be in range [-1,1] for Diffusion Policy! "
                    "Check if hdf5_normalize_action is enabled."
                )
            self.action_check_done = True

        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)

    def train_on_batch(self, batch, epoch, validate=False):
        """
        Training on a single batch of data.

        Forward pass:
          1. Encode both observation groups.
          2. Predict guidance weight w_force from the combined features.
          3. Sample random noise and timesteps; add noise to clean actions.
          4. Run both noise nets to get per-expert epsilon predictions.
          5. Apply CFG-style composition:
                 eps_final = eps_vis + w_force * (eps_tor − eps_vis)
          6. Compute MSE denoising loss against the sampled noise.
          7. (optional) Compute x0-variance auxiliary loss that steers the router
             toward assigning larger guidance scales when the image expert is
             more uncertain than the torque expert.

        Args:
            batch    (dict): processed batch from @process_batch_for_training
            epoch    (int):  epoch number
            validate (bool): if True, skip gradient updates

        Returns:
            info (dict): losses and diagnostic information
        """
        B = batch["actions"].shape[0]

        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = super(DiffusionPolicyUNetCompositionV2, self).train_on_batch(
                batch, epoch, validate=validate)
            actions = batch["actions"]

            inputs = {
                "obs": batch["obs"],
                "goal": batch["goal_obs"],
            }
            for k in self.obs_shapes:
                assert inputs["obs"][k].ndim - 2 == len(self.obs_shapes[k])

            # ── Encode observations ───────────────────────────────────────
            image_cond, torque_cond, obs_cond_all = self._encode_obs(inputs, self.nets)

            # ── Guidance weight from router ───────────────────────────────
            guidance_logit = self.nets["policy"]["weight_predictor"](obs_cond_all)  # [B, 1]
            w_force = self._guidance_weight(guidance_logit).view(-1, 1, 1)          # [B, 1, 1]

            # ── Forward diffusion ─────────────────────────────────────────
            noise = torch.randn(actions.shape, device=self.device)
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,
                (B,), device=self.device,
            ).long()
            noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)

            # ── Per-expert noise predictions ──────────────────────────────
            image_pred = self.nets["policy"]["image_noise_net"](
                noisy_actions, timesteps, global_cond=image_cond)
            torque_pred = self.nets["policy"]["torque_noise_net"](
                noisy_actions, timesteps, global_cond=torque_cond)

            # ── CFG-style composition (Improvement 2) ─────────────────────
            # eps_final = eps_vision + w_force * (eps_force - eps_vision)
            # Equivalent to interpolation when w_force ∈ [0,1]; extrapolates
            # toward force when w_force > 1.
            noise_pred = image_pred + w_force * (torque_pred - image_pred)

            # ── Main denoising loss ───────────────────────────────────────
            main_loss = F.mse_loss(noise_pred, noise)

            # ── Variance auxiliary loss (Improvement 4, training side) ────
            # Disabled during validation and when not configured.
            var_aux_loss = torch.tensor(0.0, device=self.device)
            vcfg = getattr(self.algo_config, "variance_routing", None)
            use_var_loss = (
                not validate
                and vcfg is not None
                and getattr(vcfg, "enabled", False)
            )
            if use_var_loss:
                K = getattr(vcfg, "K_samples", 4)
                lambda_var = float(getattr(vcfg, "lambda_var", 0.1))

                # Variance targets are computed without gradients so that the
                # noise nets are not trained to artificially inflate/deflate
                # their own variance – only the router receives the gradient.
                with torch.no_grad():
                    image_var, torque_var = self._estimate_x0_variances(
                        actions, timesteps, image_cond, torque_cond, K=K)

                # Target guidance weight: proportional to image expert's share
                # of total variance, scaled to [0, max_guidance_scale].
                # When image_var >> torque_var the target approaches max_guidance_scale,
                # encouraging the router to rely heavily on force.
                var_total = image_var + torque_var + 1e-8
                target_w_force = (image_var / var_total) * self.max_guidance_scale  # [B]

                var_aux_loss = lambda_var * F.mse_loss(
                    w_force.view(-1), target_w_force.detach())

            total_loss = main_loss + var_aux_loss

            losses = {
                "l2_loss": main_loss,
                "var_aux_loss": var_aux_loss,
                "total_loss": total_loss,
            }
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
        Process info dict from @train_on_batch for tensorboard logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = super(DiffusionPolicyUNetCompositionV2, self).log_info(info)
        log["Loss"] = info["losses"]["total_loss"].item()
        log["L2_Loss"] = info["losses"]["l2_loss"].item()
        if info["losses"]["var_aux_loss"].item() != 0.0:
            log["Var_Aux_Loss"] = info["losses"]["var_aux_loss"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def reset(self):
        """Reset algo state to prepare for environment rollouts."""
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        self.obs_queue = deque(maxlen=To)
        self.action_queue = deque(maxlen=Ta)

    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.

        Args:
            obs_dict  (dict): current observation [1, Do]
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

        Variance-based hard-switching (Improvement 4, inference side):
          At each denoising step we compute the x0 estimate for each expert and
          append it to a sliding window of length `inference_window`.  Once the
          window is full we compute the per-expert variance and check the ratio:

              var_ratio = image_var / (torque_var + ε)

          If var_ratio > `inference_threshold` the guidance weight is overridden
          to 1.0 (pure force) for that step, hard-switching away from the
          confused vision expert.

        Args:
            obs_dict  (dict): current observation
            goal_dict (dict): optional goal

        Returns:
            naction (Tensor): [B, Ta, Da] action chunk to execute
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

        # ── Encode and compute base guidance weight ───────────────────────
        image_cond, torque_cond, obs_cond_all = self._encode_obs(inputs, nets)
        B = image_cond.shape[0]

        guidance_logit = nets["policy"]["weight_predictor"](obs_cond_all)  # [B, 1]
        w_force_base = self._guidance_weight(guidance_logit).view(-1, 1, 1)  # [B, 1, 1]

        # ── Variance-routing inference config ─────────────────────────────
        vcfg = getattr(self.algo_config, "variance_routing", None)
        use_var_routing = (
            vcfg is not None and getattr(vcfg, "enabled", False)
        )
        var_window = int(getattr(vcfg, "inference_window", 5)) if use_var_routing else 5
        var_threshold = float(getattr(vcfg, "inference_threshold", 5.0)) if use_var_routing else 5.0

        # ── Denoising loop ────────────────────────────────────────────────
        naction = torch.randn((B, Tp, action_dim), device=self.device)
        self.noise_scheduler.set_timesteps(num_inference_timesteps)

        x0_image_history = []   # sliding window of x0 estimates for image expert
        x0_torque_history = []  # sliding window of x0 estimates for torque expert
        w_force_current = w_force_base

        for k in self.noise_scheduler.timesteps:
            image_pred = nets["policy"]["image_noise_net"](
                sample=naction, timestep=k, global_cond=image_cond)
            torque_pred = nets["policy"]["torque_noise_net"](
                sample=naction, timestep=k, global_cond=torque_cond)

            # ── Variance-based guidance override (Improvement 4) ──────────
            if use_var_routing:
                with torch.no_grad():
                    x0_image = self._compute_x0_from_eps(naction, image_pred, k)
                    x0_torque = self._compute_x0_from_eps(naction, torque_pred, k)

                x0_image_history.append(x0_image)
                x0_torque_history.append(x0_torque)

                if len(x0_image_history) >= var_window:
                    # [W, B, Tp, Da] → variance over W → mean over (Tp, Da) → [B]
                    recent_img = torch.stack(x0_image_history[-var_window:])
                    recent_tor = torch.stack(x0_torque_history[-var_window:])

                    img_var = recent_img.var(dim=0).mean(dim=[-1, -2])   # [B]
                    tor_var = recent_tor.var(dim=0).mean(dim=[-1, -2])   # [B]
                    var_ratio = img_var / (tor_var + 1e-8)               # [B]

                    # Hard-switch: if image expert variance dominates, go pure force
                    hard_switch = (var_ratio > var_threshold).view(-1, 1, 1)  # [B, 1, 1]
                    w_force_current = torch.where(
                        hard_switch,
                        torch.ones_like(w_force_base),  # w_force = 1 → pure force
                        w_force_base,
                    )

            # ── CFG-style composition (Improvement 2) ────────────────────
            # eps_final = eps_vision + w_force * (eps_force - eps_vision)
            noise_pred = image_pred + w_force_current * (torque_pred - image_pred)

            naction = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction,
            ).prev_sample

        # Return the executed action slice
        start = To - 1
        end = start + Ta
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
