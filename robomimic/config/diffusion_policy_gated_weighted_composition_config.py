"""
Config for Diffusion Policy Gated Weighted Composition algorithm.

Combines gating (contact-gated force expert with neutral embedding) with
softmax-weighted expert composition:
  - Hard binary contact gate supervises the force encoder
  - Two specialist UNets (scene / force) blended by a learned weight predictor
  - w_force_n = phi * w_force / (w_scene + phi * w_force)
    → phi=0: pure vision; phi=1: softmax-weighted blend
"""

from robomimic.config.base_config import BaseConfig


class DiffusionPolicyGatedWeightedCompositionConfig(BaseConfig):
    ALGO_NAME = "diffusion_policy_gated_weighted_composition"

    def train_config(self):
        super(DiffusionPolicyGatedWeightedCompositionConfig, self).train_config()

        # next_obs required for contact gate supervision
        self.train.hdf5_load_next_obs = True

        # must match algo.horizon.prediction_horizon and observation_horizon
        self.train.seq_length = 16
        self.train.frame_stack = 2

    def algo_config(self):
        # ── Optimiser ─────────────────────────────────────────────────────
        self.algo.optim_params.policy.optimizer_type = "adamw"
        self.algo.optim_params.policy.learning_rate.initial = 1e-4
        self.algo.optim_params.policy.learning_rate.decay_factor = 0.1
        self.algo.optim_params.policy.learning_rate.step_every_batch = True
        self.algo.optim_params.policy.learning_rate.scheduler_type = "cosine"
        self.algo.optim_params.policy.learning_rate.num_cycles = 0.5
        self.algo.optim_params.policy.learning_rate.warmup_steps = 500
        self.algo.optim_params.policy.learning_rate.epoch_schedule = []
        self.algo.optim_params.policy.learning_rate.do_not_lock_keys()
        self.algo.optim_params.policy.regularization.L2 = 1e-6

        # ── Horizon ───────────────────────────────────────────────────────
        self.algo.horizon.observation_horizon = 2
        self.algo.horizon.action_horizon = 8
        self.algo.horizon.prediction_horizon = 16

        # ── Gating ────────────────────────────────────────────────────────
        self.algo.gating.contact_force_threshold = 1.0

        # ── UNet ──────────────────────────────────────────────────────────
        self.algo.unet.enabled = True
        self.algo.unet.diffusion_step_embed_dim = 256
        self.algo.unet.down_dims = [256, 512, 1024]
        self.algo.unet.kernel_size = 5
        self.algo.unet.n_groups = 8

        # ── Transformer (disabled; only UNet supported) ───────────────────
        self.algo.transformer.enabled = False

        # ── EMA ───────────────────────────────────────────────────────────
        self.algo.ema.enabled = True
        self.algo.ema.power = 0.75

        # ── Noise scheduler ───────────────────────────────────────────────
        ## DDPM (default)
        self.algo.ddpm.enabled = True
        self.algo.ddpm.num_train_timesteps = 100
        self.algo.ddpm.num_inference_timesteps = 100
        self.algo.ddpm.beta_schedule = 'squaredcos_cap_v2'
        self.algo.ddpm.clip_sample = True
        self.algo.ddpm.prediction_type = 'epsilon'

        ## DDIM (disabled by default)
        self.algo.ddim.enabled = False
        self.algo.ddim.num_train_timesteps = 100
        self.algo.ddim.num_inference_timesteps = 10
        self.algo.ddim.beta_schedule = 'squaredcos_cap_v2'
        self.algo.ddim.clip_sample = True
        self.algo.ddim.set_alpha_to_one = True
        self.algo.ddim.steps_offset = 0
        self.algo.ddim.prediction_type = 'epsilon'
