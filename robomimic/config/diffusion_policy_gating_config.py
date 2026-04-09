"""
Config for Diffusion Policy with Gating algorithm.

This config mirrors the standard diffusion policy configs but adds
gating-specific parameters such as contact prediction thresholds and
loss weighting.
"""

from robomimic.config.base_config import BaseConfig


class DiffusionPolicyGatingConfig(BaseConfig):
    ALGO_NAME = "diffusion_policy_gating"

    def train_config(self):
        """
        Setting up training parameters for Diffusion Policy with Gating.

        - ENABLE next_obs loading from hdf5 to get force/torque observations
        - set compatible data loading parameters
        """
        super(DiffusionPolicyGatingConfig, self).train_config()

        # ENABLE next_obs loading from hdf5 for contact/force supervision
        self.train.hdf5_load_next_obs = True

        # set compatible data loading parameters
        self.train.seq_length = 16  # should match self.algo.horizon.prediction_horizon
        self.train.frame_stack = 2  # should match self.algo.horizon.observation_horizon

    def algo_config(self):
        """
        Populate `config.algo` with parameters the gating algorithm expects.
        """

        # optimization parameters
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

        # horizon parameters
        self.algo.horizon.observation_horizon = 2
        self.algo.horizon.action_horizon = 8
        self.algo.horizon.prediction_horizon = 16

        # gating-specific parameters
        # contact force threshold (N) used to derive contact labels from next_obs
        self.algo.gating.contact_force_threshold = 1.0
        # weight for the contact prediction loss when combined with action loss
        self.algo.gating.contact_loss_weight = 0.1
        # whether to create a learnable neutral embedding (h*) for gating fusion
        self.algo.gating.neutral_embedding_enabled = True
        # contact predictor MLP hidden dims
        self.algo.gating.contact_predictor_hidden = [256, 128]

        # UNet parameters (policy network)
        self.algo.unet.enabled = True
        self.algo.unet.diffusion_step_embed_dim = 256
        self.algo.unet.down_dims = [256, 512, 1024]
        self.algo.unet.kernel_size = 5
        self.algo.unet.n_groups = 8

        # EMA parameters
        self.algo.ema.enabled = True
        self.algo.ema.power = 0.75

        # Noise Scheduler
        ## DDPM
        self.algo.ddpm.enabled = True
        self.algo.ddpm.num_train_timesteps = 100
        self.algo.ddpm.num_inference_timesteps = 100
        self.algo.ddpm.beta_schedule = 'squaredcos_cap_v2'
        self.algo.ddpm.clip_sample = True
        self.algo.ddpm.prediction_type = 'epsilon'

        ## DDIM
        self.algo.ddim.enabled = False
        self.algo.ddim.num_train_timesteps = 100
        self.algo.ddim.num_inference_timesteps = 10
        self.algo.ddim.beta_schedule = 'squaredcos_cap_v2'
        self.algo.ddim.clip_sample = True
        self.algo.ddim.set_alpha_to_one = True
        self.algo.ddim.steps_offset = 0
        self.algo.ddim.prediction_type = 'epsilon'
