"""
Implementation of Diffusion Policy https://diffusion-policy.cs.columbia.edu/ by Cheng Chi
"""
from typing import Callable, Union
import math
from collections import OrderedDict, deque
from packaging.version import parse as parse_version
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
# requires diffusers==0.11.1
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

import random
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils


@register_algo_factory_func("diffusion_policy_gating")
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
        return DiffusionPolicyUNetGating, {}
    elif algo_config.transformer.enabled:
        raise NotImplementedError()
    else:
        raise RuntimeError()


class DiffusionPolicyUNetGating(PolicyAlgo):
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        # set up different observation groups for @MIMO_MLP
        
        # Split obs keys into scene and force
        scene_obs_shapes = OrderedDict()
        force_obs_shapes = OrderedDict()
        
        self.force_keys = ["robot0_contact_force", "robot0_contact_torque"]
        
        for k, v in self.obs_shapes.items():
            if k in self.force_keys:
                force_obs_shapes[k] = v
            else:
                scene_obs_shapes[k] = v
                
        # Get general encoder kwargs from config
        encoder_kwargs = ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder)
        
        # 1. Scene Encoder for Policy
        self.scene_encoder = ObsNets.ObservationGroupEncoder(
            observation_group_shapes=OrderedDict([("obs", scene_obs_shapes)]),
            encoder_kwargs=encoder_kwargs,
        )
        self.scene_encoder = replace_bn_with_gn(self.scene_encoder)
        scene_dim = self.scene_encoder.output_shape()[0]

        # 2. Force Encoder for Policy
        self.force_encoder = ObsNets.ObservationGroupEncoder(
            observation_group_shapes=OrderedDict([("obs", force_obs_shapes)]),
            encoder_kwargs=encoder_kwargs,
        )
        self.force_encoder = replace_bn_with_gn(self.force_encoder)
        force_dim = self.force_encoder.output_shape()[0]

        # 3. Scene Encoder for Contact
        self.scene_encoder_contact = ObsNets.ObservationGroupEncoder(
            observation_group_shapes=OrderedDict([("obs", scene_obs_shapes)]),
            encoder_kwargs=encoder_kwargs,
        )
        self.scene_encoder_contact = replace_bn_with_gn(self.scene_encoder_contact)

        # 4. Force Encoder for Contact
        self.force_encoder_contact = ObsNets.ObservationGroupEncoder(
            observation_group_shapes=OrderedDict([("obs", force_obs_shapes)]),
            encoder_kwargs=encoder_kwargs,
        )
        self.force_encoder_contact = replace_bn_with_gn(self.force_encoder_contact)
        
        # 3. Learnable neutral embedding (h*)
        # Same dimension as force feature
        if force_dim > 0:
            self.neutral_embedding = BaseNets.Parameter(torch.zeros(force_dim))
        else:
            self.neutral_embedding = None

        # Total obs dim is scene + force (after gating, it's concatenation)
        obs_dim = scene_dim + force_dim

        # create network object
        noise_pred_net = DPNets.ConditionalUnet1D(
            input_dim=self.ac_dim,
            global_cond_dim=obs_dim*self.algo_config.horizon.observation_horizon
        )

        # create a contact predictor MLP taking the encoded obs as input
        # It takes raw combined? 
        # Usually contact prediction depends on visual + force history
        contact_predictor = nn.Sequential(
            nn.Linear(obs_dim*self.algo_config.horizon.observation_horizon, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),  # output dim 1 for binary contact
            nn.Sigmoid()  # assuming binary contact prediction
        )

        # the final arch has 2 parts
        # Store individual encoders in policy dict
        policy_nets = nn.ModuleDict({
            "scene_encoder": self.scene_encoder,
            "force_encoder": self.force_encoder,
            "scene_encoder_contact": self.scene_encoder_contact,
            "force_encoder_contact": self.force_encoder_contact,
            "noise_pred_net": noise_pred_net,
            "contact_predictor": contact_predictor
        })
        
        if self.neutral_embedding is not None:
             policy_nets["neutral_embedding"] = self.neutral_embedding

        nets = nn.ModuleDict({
            "policy": policy_nets
        })

        nets = nets.float().to(self.device)
        
        # setup noise scheduler
        noise_scheduler = None
        if self.algo_config.ddpm.enabled:
            noise_scheduler = DDPMScheduler(
                num_train_timesteps=self.algo_config.ddpm.num_train_timesteps,
                beta_schedule=self.algo_config.ddpm.beta_schedule,
                clip_sample=self.algo_config.ddpm.clip_sample,
                prediction_type=self.algo_config.ddpm.prediction_type
            )
        elif self.algo_config.ddim.enabled:
            noise_scheduler = DDIMScheduler(
                num_train_timesteps=self.algo_config.ddim.num_train_timesteps,
                beta_schedule=self.algo_config.ddim.beta_schedule,
                clip_sample=self.algo_config.ddim.clip_sample,
                set_alpha_to_one=self.algo_config.ddim.set_alpha_to_one,
                steps_offset=self.algo_config.ddim.steps_offset,
                prediction_type=self.algo_config.ddim.prediction_type
            )
        else:
            raise RuntimeError()
        
        # setup EMA
        ema = None
        if self.algo_config.ema.enabled:
            ema = EMAModel(model=nets, power=self.algo_config.ema.power)
                
        # set attrs
        self.nets = nets
        self.noise_scheduler = noise_scheduler
        self.ema = ema
        self.action_check_done = False
        self.obs_queue = None
        self.action_queue = None
    
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
        Ta = self.algo_config.horizon.action_horizon
        Tp = self.algo_config.horizon.prediction_horizon

        input_batch = dict()
        input_batch["obs"] = {k: batch["obs"][k][:, :To, :] for k in batch["obs"]} # [B, To, *dim_k]
        input_batch["goal_obs"] = batch.get("goal_obs", None) # goals may not be present
        input_batch["actions"] = batch["actions"][:, :Tp, :] # [B, Tp, Da]
        
        # check if actions are normalized to [-1,1]
        if not self.action_check_done:
            actions = input_batch["actions"]
            in_range = (-1 <= actions) & (actions <= 1)
            all_in_range = torch.all(in_range).item()
            if not all_in_range:
                raise ValueError("'actions' must be in range [-1,1] for Diffusion Policy! Check if hdf5_normalize_action is enabled.")
            self.action_check_done = True
            
        # Extract contact force from next_obs for supervision
        # if any of the force > thres in the observation horizon, we use use force features
        contact_force_thres = self.algo_config.gating.contact_force_threshold # N, tunable threshold
        next_force = batch["obs"][self.force_keys[0]][:, :Ta, -3:] # [B, To, Df=3]
        next_contact_label = next_force.norm(dim=-1) > contact_force_thres  # [B, To]
        # if any contact in To horizon
        next_contact_label = next_contact_label.float().max(dim=1)[0].unsqueeze(-1)  # [B, 1]
        input_batch["next_contact_label"] = next_contact_label

        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)
        
    def train_on_batch(self, batch, epoch, validate=False):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            epoch (int): epoch number - required by some Algos that need
                to perform staged training and early stopping

            validate (bool): if True, don't perform any learning updates.

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        Tp = self.algo_config.horizon.prediction_horizon
        action_dim = self.ac_dim
        B = batch["actions"].shape[0]
        
        
        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = super(DiffusionPolicyUNetGating, self).train_on_batch(batch, epoch, validate=validate)
            actions = batch["actions"]
            
            # encode obs
            inputs = {
                "obs": batch["obs"],
                "goal": batch["goal_obs"]
            }
            # Note: inputs["obs"] contains ALL keys. Encoders filter automatically via GroupEncoder logic.
            
            # 1. Encode Scene
            scene_features = TensorUtils.time_distributed(inputs, self.nets["policy"]["scene_encoder"], inputs_as_kwargs=True)
            
            # 2. Encode Force [B, To, 60]
            force_features = TensorUtils.time_distributed(inputs, self.nets["policy"]["force_encoder"], inputs_as_kwargs=True)

            # 3. Encode Scene for Contact
            scene_features_contact = TensorUtils.time_distributed(inputs, self.nets["policy"]["scene_encoder_contact"], inputs_as_kwargs=True)
            
            # 4. Encode Force for Contact
            force_features_contact = TensorUtils.time_distributed(inputs, self.nets["policy"]["force_encoder_contact"], inputs_as_kwargs=True)
            
            # [B, T, D_s] and [B, T, dim_img_feature + 60]
            # Concatenate for contact predictor input (using raw force for prediction)
            raw_features = torch.cat([scene_features_contact, force_features_contact], dim=-1) # [B, T, D]

            # Gating: we use raw features as input to a MLP to predict whether there will be contact
            contact_preds = self.nets["policy"]["contact_predictor"](
                raw_features.flatten(start_dim=1)
            )  # [B, 1]
            info["contact_preds"] = TensorUtils.detach(contact_preds)
            
            # Gating Fusion logic:
            # h_t = [h^s_t ; phi(t) * h^f_t + (1 - phi(t)) * h^* ]
            # phi(t) here assumed to be contact_preds broadcasted over time T
            
            phi = contact_preds.unsqueeze(1) # [B, 1, 1]
            
            if "neutral_embedding" in self.nets["policy"]:
                h_star = self.nets["policy"]["neutral_embedding"]() # [D_f]
                # reshape h_star to [1, 1, D_f] to broadcast
                h_star = h_star.view(1, 1, -1)
                
                # Apply gating math
                # phi is [B, 1, 1], force_features is [B, T, D_f]
                gated_force = phi * force_features + (1.0 - phi) * h_star
            else:
                gated_force = force_features

            # Fused features
            obs_features = torch.cat([scene_features, gated_force], dim=-1)
            assert obs_features.ndim == 3  # [B, T, D]

            obs_cond = obs_features.flatten(start_dim=1)
            
            # sample noise to add to actions
            noise = torch.randn(actions.shape, device=self.device)
            
            # sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, 
                (B,), device=self.device
            ).long()
            
            # add noise to the clean actions according to the noise magnitude at each diffusion iteration
            # (this is the forward diffusion process)
            noisy_actions = self.noise_scheduler.add_noise(
                actions, noise, timesteps)
            
            # predict the noise residual
            noise_pred = self.nets["policy"]["noise_pred_net"](
                noisy_actions, timesteps, global_cond=obs_cond)
            
            # L2 loss
            action_loss = F.mse_loss(noise_pred, noise)
            
            # Contact Loss
            # Target derived from next_contact_force if available
            
            contact_labels = batch["next_contact_label"]  # [B, 1]

            contact_loss = F.binary_cross_entropy(contact_preds, contact_labels)
            alpha = self.algo_config.gating.contact_loss_weight # Default weighting, tunable param
            total_loss = action_loss + alpha * contact_loss
            
            # logging
            losses = {
                "action_loss": action_loss,
                "contact_loss": contact_loss,
                "total_loss": total_loss
            }
            info["losses"] = TensorUtils.detach(losses)

            if not validate:
                # gradient step
                policy_grad_norms = TorchUtils.backprop_for_loss(
                    net=self.nets,
                    optim=self.optimizers["policy"],
                    loss=total_loss,
                )
                
                # update Exponential Moving Average of the model weights
                if self.ema is not None:
                    self.ema.step(self.nets)
                
                step_info = {
                    "policy_grad_norms": policy_grad_norms
                }
                info.update(step_info)

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
        log = super(DiffusionPolicyUNetGating, self).log_info(info)
        log["Action_Loss"] = info["losses"]["action_loss"].item()
        log["Contact_Loss"] = info["losses"]["contact_loss"].item()
        log["Total_Loss"] = info["losses"]["total_loss"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log
    
    def reset(self):
        """
        Reset algo state to prepare for environment rollouts.
        """
        # setup inference queues
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        obs_queue = deque(maxlen=To)
        action_queue = deque(maxlen=Ta)
        self.obs_queue = obs_queue
        self.action_queue = action_queue
    
    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation [1, Do]
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor [1, Da]
        """
        # obs_dict: key: [1,D]
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        
        if len(self.action_queue) == 0:
            # no actions left, run inference
            # [1,T,Da]
            action_sequence = self._get_action_trajectory(obs_dict=obs_dict)
            
            # put actions into the queue
            self.action_queue.extend(action_sequence[0])
        
        # has action, execute from left to right
        # [Da]
        action = self.action_queue.popleft()
        
        # [1,Da]
        action = action.unsqueeze(0)
        return action
        
    def _get_action_trajectory(self, obs_dict, goal_dict=None):
        assert not self.nets.training
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        Tp = self.algo_config.horizon.prediction_horizon
        action_dim = self.ac_dim
        if self.algo_config.ddpm.enabled is True:
            num_inference_timesteps = self.algo_config.ddpm.num_inference_timesteps
        elif self.algo_config.ddim.enabled is True:
            num_inference_timesteps = self.algo_config.ddim.num_inference_timesteps
        else:
            raise ValueError
        
        # select network
        nets = self.nets
        if self.ema is not None:
            nets = self.ema.averaged_model
        
        # encode obs
        inputs = {
            "obs": obs_dict,
            "goal": goal_dict
        }
        for k in self.obs_shapes:
            # first two dimensions should be [B, T] for inputs
            if inputs["obs"][k].ndim - 1 == len(self.obs_shapes[k]):
                # adding time dimension if not present -- this is required as
                # frame stacking is not invoked when sequence length is 1
                inputs["obs"][k] = inputs["obs"][k].unsqueeze(1)
            assert inputs["obs"][k].ndim - 2 == len(self.obs_shapes[k])
            
        # Inference pass with gating
        scene_features = TensorUtils.time_distributed(inputs, nets["policy"]["scene_encoder"], inputs_as_kwargs=True)
        force_features = TensorUtils.time_distributed(inputs, nets["policy"]["force_encoder"], inputs_as_kwargs=True)
        
        # Encoders for contact prediction
        scene_features_contact = TensorUtils.time_distributed(inputs, nets["policy"]["scene_encoder_contact"], inputs_as_kwargs=True)
        force_features_contact = TensorUtils.time_distributed(inputs, nets["policy"]["force_encoder_contact"], inputs_as_kwargs=True)
        
        raw_features = torch.cat([scene_features_contact, force_features_contact], dim=-1)

        contact_preds = nets["policy"]["contact_predictor"](
             raw_features.flatten(start_dim=1)
        )
        phi = contact_preds.unsqueeze(1) # [B, 1, 1]

        if "neutral_embedding" in nets["policy"]:
             h_star = nets["policy"]["neutral_embedding"]()
             h_star = h_star.view(1, 1, -1)
             gated_force = phi * force_features + (1.0 - phi) * h_star
        else:
             gated_force = force_features
             
        obs_features = torch.cat([scene_features, gated_force], dim=-1)
        
        assert obs_features.ndim == 3  # [B, T, D]
        B = obs_features.shape[0]

        # reshape observation to (B,obs_horizon*obs_dim)
        obs_cond = obs_features.flatten(start_dim=1)

        # initialize action from Guassian noise
        noisy_action = torch.randn(
            (B, Tp, action_dim), device=self.device)
        naction = noisy_action
        
        # init scheduler
        self.noise_scheduler.set_timesteps(num_inference_timesteps)

        for k in self.noise_scheduler.timesteps:
            # predict noise
            noise_pred = nets["policy"]["noise_pred_net"](
                sample=naction, 
                timestep=k,
                global_cond=obs_cond
            )

            # inverse diffusion step (remove noise)
            naction = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction
            ).prev_sample

        # process action using Ta
        start = To - 1
        end = start + Ta
        action = naction[:,start:end]
        return action

    def serialize(self):
        """
        Get dictionary of current model parameters.
        """
        return {
            "nets": self.nets.state_dict(),
            "optimizers": { k : self.optimizers[k].state_dict() for k in self.optimizers },
            "lr_schedulers": { k : self.lr_schedulers[k].state_dict() if self.lr_schedulers[k] is not None else None for k in self.lr_schedulers },
            "ema": self.ema.averaged_model.state_dict() if self.ema is not None else None,
        }

    def deserialize(self, model_dict, load_optimizers=False):
        """
        Load model from a checkpoint.

        Args:
            model_dict (dict): a dictionary saved by self.serialize() that contains
                the same keys as @self.network_classes
            load_optimizers (bool): whether to load optimizers and lr_schedulers from the model_dict;
                used when resuming training from a checkpoint
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
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group, 
            num_channels=x.num_features)
    )
    return root_module
