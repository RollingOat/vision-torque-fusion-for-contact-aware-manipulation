"""
Config for Diffusion Policy Composition v2 algorithm.

Two improvements over the baseline composition:
  1. CFG-style modality guidance (Improvement 2):
       eps_final = eps_vision + w_force * (eps_force - eps_vision)
     The router outputs a single guidance scale in [0, max_guidance_scale]
     instead of two softmax weights; w_force > 1 amplifies haptic feedback.
  2. Variance-based confidence routing (Improvement 4):
     At inference, x0 prediction variance across denoising steps triggers a
     hard-switch to the force expert when the vision expert is uncertain.
     An optional training auxiliary loss teaches the router to be variance-aware.
"""

from robomimic.config.base_config import BaseConfig


class DiffusionPolicyCompositionV2Config(BaseConfig):
    ALGO_NAME = "diffusion_policy_composition_v2"

    def train_config(self):
        """
        Training parameters for Diffusion Policy Composition v2.
        - next_obs not required (no contact labels needed)
        - seq_length / frame_stack must match prediction / observation horizons
        """
        super(DiffusionPolicyCompositionV2Config, self).train_config()

        # next_obs not needed
        self.train.hdf5_load_next_obs = False

        # must match algo.horizon.prediction_horizon and observation_horizon
        self.train.seq_length = 16
        self.train.frame_stack = 2

    def algo_config(self):
        """
        Populate `config.algo` with all parameters the v2 algorithm expects.
        """

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

        # ── CFG guidance scale ────────────────────────────────────────────
        # Router output is mapped to [0, max_guidance_scale] via sigmoid.
        # Values > 1 extrapolate the force signal beyond simple blending.
        self.algo.max_guidance_scale = 2.0

        # ── Variance-based confidence routing ─────────────────────────────
        # Set enabled=True to activate both the training aux loss and the
        # inference hard-switch.
        self.algo.variance_routing.enabled = False
        # Number of noise samples drawn per step to estimate x0 variance
        # during the auxiliary training loss computation.
        self.algo.variance_routing.K_samples = 4
        # Weight of the variance auxiliary loss relative to the main L2 loss.
        self.algo.variance_routing.lambda_var = 0.1
        # Inference: variance ratio threshold above which the guidance weight
        # is hard-set to 1.0 (pure force).  Ratio = image_var / torque_var.
        self.algo.variance_routing.inference_threshold = 5.0
        # Sliding window size (denoising steps) for x0 variance estimation.
        self.algo.variance_routing.inference_window = 5

        # ── UNet ──────────────────────────────────────────────────────────
        self.algo.unet.enabled = True
        self.algo.unet.diffusion_step_embed_dim = 256
        self.algo.unet.down_dims = [256, 512, 1024]
        self.algo.unet.kernel_size = 5
        self.algo.unet.n_groups = 8

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

        ## DDIM (disabled by default; swap with DDPM for faster inference)
        self.algo.ddim.enabled = False
        self.algo.ddim.num_train_timesteps = 100
        self.algo.ddim.num_inference_timesteps = 10
        self.algo.ddim.beta_schedule = 'squaredcos_cap_v2'
        self.algo.ddim.clip_sample = True
        self.algo.ddim.set_alpha_to_one = True
        self.algo.ddim.steps_offset = 0
        self.algo.ddim.prediction_type = 'epsilon'
