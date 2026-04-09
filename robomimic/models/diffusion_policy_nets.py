""" This file contains nets used for Diffusion Policy. """
import math
from typing import Union

import torch
import torch.nn as nn

from robomimic.models.transformers import GPT_Backbone


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """
        Conv1d --> GroupNorm --> Mish
    """

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, 
            in_channels, 
            out_channels, 
            cond_dim,
            kernel_size=3,
            n_groups=8):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (-1, 1))
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        """
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        """
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)

        embed = embed.reshape(
            embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:,0,...]
        bias = embed[:,1,...]
        out = scale * out + bias

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class TransformerForDiffusion(nn.Module):
    """
    Transformer-based noise prediction network for Diffusion Policy.

    Builds a flat sequence [timestep_token | obs_tokens (To) | action_tokens (Tp)]
    and applies causal self-attention (via GPT_Backbone) so each token only attends
    to itself and prior tokens:
      - Observation tokens attend to the timestep + earlier obs tokens.
      - Action tokens attend to the timestep, all obs tokens, and prior action tokens.
    Only the last Tp token outputs are projected to produce noise predictions;
    timestep and observation token outputs are discarded.
    """

    def __init__(self,
        input_dim,
        output_dim,
        horizon,
        obs_dim,
        obs_horizon,
        n_layer=4,
        n_head=4,
        n_emb=256,
        p_drop_emb=0.1,
        p_drop_attn=0.1,
    ):
        """
        Args:
            input_dim (int): action dimension
            output_dim (int): noise prediction dimension (= input_dim)
            horizon (int): prediction horizon Tp
            obs_dim (int): observation feature dimension (after encoder)
            obs_horizon (int): observation horizon To
            n_layer (int): number of causal transformer blocks
            n_head (int): number of attention heads (must divide n_emb)
            n_emb (int): embedding / model dimension
            p_drop_emb (float): dropout probability on input embeddings
            p_drop_attn (float): dropout probability inside transformer blocks
        """
        super().__init__()

        # sinusoidal embedding for the diffusion timestep k, then MLP projection
        self.timestep_emb = nn.Sequential(
            SinusoidalPosEmb(n_emb),
            nn.Linear(n_emb, n_emb * 4),
            nn.Mish(),
            nn.Linear(n_emb * 4, n_emb),
        )

        # project obs features and noisy actions into the model dimension
        self.obs_proj = nn.Linear(obs_dim, n_emb)
        self.action_proj = nn.Linear(input_dim, n_emb)

        # learned positional embedding for the full sequence
        seq_len = 1 + obs_horizon + horizon  # timestep + obs + actions
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, n_emb))
        nn.init.normal_(self.pos_emb, std=0.02)

        self.drop = nn.Dropout(p_drop_emb)

        # causal transformer: stacked SelfAttentionBlocks with lower-triangular mask
        # reuses GPT_Backbone from robomimic/models/transformers.py
        self.transformer = GPT_Backbone(
            embed_dim=n_emb,
            context_length=seq_len,
            attn_dropout=p_drop_attn,
            block_output_dropout=p_drop_attn,
            num_layers=n_layer,
            num_heads=n_head,
            activation="gelu",
        )

        # project action token outputs to noise predictions
        self.output_proj = nn.Linear(n_emb, output_dim)

        self.horizon = horizon
        self.obs_horizon = obs_horizon

        # re-apply init to cover projection layers added outside the backbone
        self.apply(self._init_weights)

        print("TransformerForDiffusion number of parameters: {:e}".format(
            sum(p.numel() for p in self.parameters()))
        )

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, sample, timestep, obs_cond):
        """
        Args:
            sample (torch.Tensor): noisy actions of shape (B, Tp, action_dim)
            timestep (torch.Tensor or int): diffusion step k, shape (B,) or scalar
            obs_cond (torch.Tensor): observation token features of shape (B, To, obs_dim)

        Returns:
            noise_pred (torch.Tensor): predicted noise of shape (B, Tp, action_dim)
        """
        B, Tp, _ = sample.shape

        # normalise timestep to a 1-D batch tensor
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timestep) and len(timestep.shape) == 0:
            timestep = timestep[None].to(sample.device)
        timestep = timestep.expand(B)

        # encode diffusion step k -> (B, 1, n_emb)
        t_emb = self.timestep_emb(timestep).unsqueeze(1)

        # project obs and noisy actions into the embedding space
        obs_emb = self.obs_proj(obs_cond)      # (B, To, n_emb)
        action_emb = self.action_proj(sample)  # (B, Tp, n_emb)

        # build sequence: [timestep | obs | actions] -> (B, 1+To+Tp, n_emb)
        x = torch.cat([t_emb, obs_emb, action_emb], dim=1)

        # add learned positional embedding and apply embedding dropout
        x = x + self.pos_emb
        x = self.drop(x)

        # causal self-attention: each position attends only to prior positions
        x = self.transformer(x)  # (B, 1+To+Tp, n_emb)

        # extract action token outputs (last Tp positions); discard timestep + obs outputs
        action_out = x[:, -Tp:, :]              # (B, Tp, n_emb)
        noise_pred = self.output_proj(action_out)  # (B, Tp, action_dim)

        return noise_pred


class ConditionalUnet1D(nn.Module):
    def __init__(self, 
        input_dim,
        global_cond_dim,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=5,
        n_groups=8
        ):
        """
        input_dim: Dim of actions.
        global_cond_dim: Dim of global conditioning applied with FiLM 
          in addition to diffusion step embedding. This is usually obs_horizon * obs_dim
        diffusion_step_embed_dim: Size of positional encoding for diffusion iteration k
        down_dims: Channel size for each UNet level. 
          The length of this array determines numebr of levels.
        kernel_size: Conv kernel size
        n_groups: Number of groups for GroupNorm
        """

        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))
        
        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        print("number of parameters: {:e}".format(
            sum(p.numel() for p in self.parameters()))
        )

    def forward(self, 
            sample: torch.Tensor, 
            timestep: Union[torch.Tensor, float, int], 
            global_cond=None):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        # (B,T,C)
        sample = sample.moveaxis(-1,-2)
        # (B,C,T)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            global_feature = torch.cat([
                global_feature, global_cond
            ], axis=-1)
        
        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        # (B,C,T)
        x = x.moveaxis(-1,-2)
        # (B,T,C)
        return x
