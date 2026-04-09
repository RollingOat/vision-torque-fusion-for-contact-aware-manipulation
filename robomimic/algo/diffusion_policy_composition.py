"""
Implementation of Diffusion Policy https://diffusion-policy.cs.columbia.edu/ by Cheng Chi

Composition variant: two noise prediction networks (image+kinematics and torque+kinematics)
whose outputs are blended by a learned weight predictor.
"""
from typing import Callable
from collections import OrderedDict, deque
from packaging.version import parse as parse_version
import torch
import torch.nn as nn
import torch.nn.functional as F
# requires diffusers==0.11.1
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.training_utils import EMAModel

import robomimic.models.obs_nets as ObsNets
import robomimic.models.diffusion_policy_nets as DPNets
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils

from robomimic.algo import register_algo_factory_func, PolicyAlgo


@register_algo_factory_func("diffusion_policy_composition")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the BC algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """
    if algo_config.unet.enabled:
        return DiffusionPolicyUNetComposition, {}
    elif algo_config.transformer.enabled:
        raise NotImplementedError()
    else:
        raise RuntimeError()


class DiffusionPolicyUNetComposition(PolicyAlgo):
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.

        Architecture:
          - image_encoder    : ObservationGroupEncoder for (rgb keys + kinematic keys)
          - torque_encoder   : ObservationGroupEncoder for (torque/force keys + kinematic keys)
          - image_noise_net  : ConditionalUnet1D conditioned on image_encoder output
          - torque_noise_net : ConditionalUnet1D conditioned on torque_encoder output
          - weight_predictor : MLP that takes concatenated features from both encoders
                               and outputs 2 softmax weights for blending the noise preds
        """
        encoder_kwargs = ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder)

        # Categorize observation keys by modality
        all_obs_keys = list(self.obs_shapes.keys())
        rgb_keys = [k for k in all_obs_keys if ObsUtils.key_is_obs_modality(k, "rgb")]
        low_dim_keys = [k for k in all_obs_keys if ObsUtils.key_is_obs_modality(k, "low_dim")]
        point_cloud_keys = [k for k in all_obs_keys if ObsUtils.key_is_obs_modality(k, "point_cloud")]

        # Split low_dim: torque/force keys vs. kinematics keys (joint pos, vel, etc.)
        torque_keys = [k for k in low_dim_keys if "torque" in k or "force" in k]
        kin_keys = [k for k in low_dim_keys if k not in torque_keys]

        # Group 1 (image model): rgb images + point clouds + kinematics
        # Group 2 (torque model): torques/forces + kinematics
        # joint_pos (kin_keys) is shared — each encoder gets its own copy
        image_group_keys = rgb_keys + point_cloud_keys + kin_keys
        torque_group_keys = torque_keys + kin_keys

        assert len(image_group_keys) > 0, "image group is empty (no rgb, point_cloud, or kinematics keys found)"
        assert len(torque_group_keys) > 0, "torque group is empty (no torque/force keys found)"

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

        # Weight predictor: concatenated features from both encoders → 2 logits
        To = self.algo_config.horizon.observation_horizon
        total_feat_dim = (image_feat_dim + torque_feat_dim) * To
        weight_predictor = nn.Sequential(
            nn.Linear(total_feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
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

        # EMA
        ema = None
        if self.algo_config.ema.enabled:
            ema = EMAModel(model=nets, power=self.algo_config.ema.power)

        self.nets = nets
        self.noise_scheduler = noise_scheduler
        self.ema = ema
        self.action_check_done = False
        self.obs_queue = None
        self.action_queue = None

    def _encode_obs(self, inputs, nets):
        """
        Encode observations through both group encoders.

        Args:
            inputs (dict): {"obs": obs_dict, "goal": goal_dict}
            nets (nn.ModuleDict): network dict (either self.nets or ema.averaged_model)

        Returns:
            image_cond  (Tensor): [B, To * D_img]  -- global cond for image noise net
            torque_cond (Tensor): [B, To * D_tor]  -- global cond for torque noise net
            obs_cond_all (Tensor): [B, To * (D_img + D_tor)]  -- input to weight predictor
        """
        image_feats = TensorUtils.time_distributed(
            inputs, nets["policy"]["image_encoder"], inputs_as_kwargs=True)
        torque_feats = TensorUtils.time_distributed(
            inputs, nets["policy"]["torque_encoder"], inputs_as_kwargs=True)
        assert image_feats.ndim == 3 and torque_feats.ndim == 3  # [B, T, D]
        image_cond = image_feats.flatten(start_dim=1)    # [B, To*D_img]
        torque_cond = torque_feats.flatten(start_dim=1)  # [B, To*D_tor]
        obs_cond_all = torch.cat([image_cond, torque_cond], dim=-1)
        return image_cond, torque_cond, obs_cond_all

    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader

        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training
        """
        To = self.algo_config.horizon.observation_horizon
        Tp = self.algo_config.horizon.prediction_horizon

        input_batch = dict()
        input_batch["obs"] = {k: batch["obs"][k][:, :To] for k in batch["obs"]}  # [B, To, ...]
        input_batch["goal_obs"] = batch.get("goal_obs", None)
        input_batch["actions"] = batch["actions"][:, :Tp, :]  # [B, Tp, Da]

        # Check that actions are normalized to [-1, 1]
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

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training
            epoch (int): epoch number
            validate (bool): if True, don't perform any learning updates.

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
        """
        Tp = self.algo_config.horizon.prediction_horizon
        B = batch["actions"].shape[0]

        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = super(DiffusionPolicyUNetComposition, self).train_on_batch(batch, epoch, validate=validate)
            actions = batch["actions"]

            inputs = {
                "obs": batch["obs"],
                "goal": batch["goal_obs"],
            }
            for k in self.obs_shapes:
                assert inputs["obs"][k].ndim - 2 == len(self.obs_shapes[k])

            # Encode both observation groups
            image_cond, torque_cond, obs_cond_all = self._encode_obs(inputs, self.nets)

            # Forward diffusion: add noise to clean actions
            noise = torch.randn(actions.shape, device=self.device)
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,
                (B,), device=self.device,
            ).long()
            noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)

            # Predict blending weights from combined features
            weight_logits = self.nets["policy"]["weight_predictor"](obs_cond_all)
            norm_weights = F.softmax(weight_logits, dim=1)  # [B, 2]
            w_img = norm_weights[:, 0].view(-1, 1, 1)
            w_tor = norm_weights[:, 1].view(-1, 1, 1)

            # Per-model noise predictions
            image_pred = self.nets["policy"]["image_noise_net"](
                noisy_actions, timesteps, global_cond=image_cond)
            torque_pred = self.nets["policy"]["torque_noise_net"](
                noisy_actions, timesteps, global_cond=torque_cond)

            # Weighted composition of noise predictions
            noise_pred = w_img * image_pred + w_tor * torque_pred

            # L2 loss against added noise (epsilon prediction)
            loss = F.mse_loss(noise_pred, noise)

            losses = {"l2_loss": loss}
            info["losses"] = TensorUtils.detach(losses)

            if not validate:
                policy_grad_norms = TorchUtils.backprop_for_loss(
                    net=self.nets,
                    optim=self.optimizers["policy"],
                    loss=loss,
                )
                if self.ema is not None:
                    self.ema.step(self.nets)
                info.update({"policy_grad_norms": policy_grad_norms})

        return info

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = super(DiffusionPolicyUNetComposition, self).log_info(info)
        log["Loss"] = info["losses"]["l2_loss"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log

    def reset(self):
        """
        Reset algo state to prepare for environment rollouts.
        """
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        self.obs_queue = deque(maxlen=To)
        self.action_queue = deque(maxlen=Ta)

    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation [1, Do]
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor [1, Da]
        """
        if len(self.action_queue) == 0:
            action_sequence = self._get_action_trajectory(obs_dict=obs_dict)
            self.action_queue.extend(action_sequence[0])

        action = self.action_queue.popleft()
        return action.unsqueeze(0)

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
            # Add time dimension if frame stacking is not active (sequence length == 1)
            if inputs["obs"][k].ndim - 1 == len(self.obs_shapes[k]):
                inputs["obs"][k] = inputs["obs"][k].unsqueeze(1)
            assert inputs["obs"][k].ndim - 2 == len(self.obs_shapes[k])

        # Encode observations and compute weights ONCE before the denoising loop
        image_cond, torque_cond, obs_cond_all = self._encode_obs(inputs, nets)
        B = image_cond.shape[0]

        weight_logits = nets["policy"]["weight_predictor"](obs_cond_all)
        norm_weights = F.softmax(weight_logits, dim=1)  # [B, 2]
        w_img = norm_weights[:, 0].view(-1, 1, 1)
        w_tor = norm_weights[:, 1].view(-1, 1, 1)

        # Initialize action from Gaussian noise
        naction = torch.randn((B, Tp, action_dim), device=self.device)
        self.noise_scheduler.set_timesteps(num_inference_timesteps)

        for k in self.noise_scheduler.timesteps:
            image_pred = nets["policy"]["image_noise_net"](
                sample=naction, timestep=k, global_cond=image_cond)
            torque_pred = nets["policy"]["torque_noise_net"](
                sample=naction, timestep=k, global_cond=torque_cond)
            noise_pred = w_img * image_pred + w_tor * torque_pred

            naction = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction,
            ).prev_sample

        # Return only the executed action slice
        start = To - 1
        end = start + Ta
        return naction[:, start:end]

    def serialize(self):
        """
        Get dictionary of current model parameters.
        """
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
            model_dict (dict): a dictionary saved by self.serialize() that contains
                the same keys as @self.network_classes
            load_optimizers (bool): whether to load optimizers and lr_schedulers from the
                model_dict; used when resuming training from a checkpoint
        """
        self.nets.load_state_dict(model_dict["nets"])

        # for backwards compatibility
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


def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
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
    # verify that all modules are replaced
    bn_list = [k.split(".") for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module


def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int = 16) -> nn.Module:
    """
    Replace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features // features_per_group,
            num_channels=x.num_features)
    )
    return root_module