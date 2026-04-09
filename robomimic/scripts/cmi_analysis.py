"""
Conditional Mutual Information (CMI) analysis for diffusion policies.

Approximates I(action ; modality_k | remaining_modalities) by comparing
the denoising loss (noise-prediction MSE) under full conditioning vs
conditioning with a specific modality zeroed out.

    I_hat(a ; m_k | rest) ∝ E[ L_eps(cond=rest) − L_eps(cond=rest+m_k) ]
                           = E[ L_eps(without m_k)  − L_eps(full)       ]

A positive value means modality k *reduces* action uncertainty → it carries
information about the action beyond what the other modalities provide.

Usage
-----
    python cmi_analysis.py \\
        --agent /path/to/last.pth \\
        --dataset /path/to/data.hdf5 \\
        --plot_path cmi_results.png \\
        --num_batches 50

The script works entirely offline on a dataset (no environment rollout
needed).  It loads the frozen policy, iterates over dataset batches,
and for every sample:

1. Adds noise to the ground-truth action to form x_t  (same noise & t for
   all conditioning variants).
2. Predicts noise with **full** conditioning  → MSE_full.
3. For each obs key k, predicts noise with key k **zeroed** (set to the
   value that normalises to zero) → MSE_without_k.
4. Records  CMI_k ∝ MSE_without_k − MSE_full  per sample.

The final output is the average CMI per modality, optionally broken down
per diffusion timestep.
"""
from __future__ import annotations

import argparse
import copy
import os
from collections import OrderedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.utils.dataset import SequenceDataset


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_policy_and_config(ckpt_path, device):
    """Load a frozen diffusion policy + its config from a checkpoint."""
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=ckpt_path, device=device, verbose=True
    )
    config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
    return policy, ckpt_dict, config


def _build_dataset(config, ckpt_dict, dataset_path=None):
    """Build a SequenceDataset compatible with the checkpoint's config."""
    shape_meta = ckpt_dict["shape_metadata"]
    obs_keys = list(shape_meta["all_obs_keys"])

    if dataset_path is None:
        # Fall back to the first dataset path in the training config.
        dataset_path = os.path.expanduser(config.train.data[0]["path"])
    else:
        dataset_path = os.path.expanduser(dataset_path)

    ds = SequenceDataset(
        hdf5_path=dataset_path,
        obs_keys=obs_keys,
        action_keys=config.train.action_keys,
        dataset_keys=config.train.dataset_keys,
        action_config=config.train.action_config,
        load_next_obs=False,
        frame_stack=config.train.frame_stack,
        seq_length=config.train.seq_length,
        pad_frame_stack=config.train.pad_frame_stack,
        pad_seq_length=config.train.pad_seq_length,
        get_pad_mask=False,
        goal_mode=config.train.goal_mode,
        hdf5_cache_mode=config.train.hdf5_cache_mode,
        hdf5_use_swmr=config.train.hdf5_use_swmr,
        hdf5_normalize_obs=False,  # We normalise manually below.
        filter_by_attribute=config.train.hdf5_filter_key,
    )
    return ds


def _get_nets(inner_policy):
    """Return the network dict (use EMA weights if available)."""
    nets = inner_policy.nets
    if inner_policy.ema is not None:
        nets = inner_policy.ema.averaged_model
    return nets


def _zero_obs_key(obs_dict, key, obs_norm_stats):
    """Set *one* observation key to its normalisation offset so that,
    after normalisation, the model sees zeros for that modality.

    For image modalities (rgb / depth) the raw value that normalises to zero
    is ``offset * 255`` (since images are divided by 255 in preprocessing).
    For low-dim keys the raw value is just the offset.
    """
    obs_out = {k: v.clone() for k, v in obs_dict.items()}

    if obs_norm_stats is not None and key in obs_norm_stats:
        offset = obs_norm_stats[key].get("offset", None)
        if offset is None:
            offset = obs_norm_stats[key].get("mean", None)
        if offset is not None:
            offset = torch.as_tensor(offset, dtype=obs_out[key].dtype, device=obs_out[key].device)
    else:
        # No stats → just zero out.
        obs_out[key] = torch.zeros_like(obs_out[key])
        return obs_out

    if ObsUtils.key_is_obs_modality(key, "rgb") or ObsUtils.key_is_obs_modality(key, "depth"):
        # Images are stored uint8 [0-255] in the dataset, divided by 255 during
        # process_obs, so the raw value that normalises to zero is offset*255.
        fill = offset * 255.0
    else:
        fill = offset

    # Broadcast to match obs shape (B, T, ...)
    fill = fill.expand_as(obs_out[key])
    obs_out[key] = fill.clone()
    return obs_out


def _preprocess_and_normalise(batch_obs, obs_norm_stats, device):
    """Apply the same preprocessing pipeline the policy uses at training:
    ``process_obs_dict`` (channel swap, /255 for images) then ``normalize_dict``.
    """
    obs = {k: v.clone() for k, v in batch_obs.items()}
    obs = TensorUtils.to_float(TensorUtils.to_device(obs, device))
    obs = ObsUtils.process_obs_dict(obs)
    if obs_norm_stats is not None:
        obs = ObsUtils.normalize_dict(obs, normalization_stats=obs_norm_stats)
    return obs


# ─────────────────────────────────────────────────────────────────────────────
# Core: compute denoising MSE given an obs conditioning variant
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_denoising_mse(
    nets,
    noise_scheduler,
    obs_cond: torch.Tensor,
    actions: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    """Forward-diffuse ``actions`` to ``x_t`` and compute noise-prediction MSE.

    All randomness (noise, timesteps) is **pre-sampled** so that comparisons
    between conditioning variants are fair.

    Returns per-sample MSE: shape ``(B,)``.
    """
    noisy_actions = noise_scheduler.add_noise(actions, noise, timesteps)
    noise_pred = nets["policy"]["noise_pred_net"](
        sample=noisy_actions,
        timestep=timesteps,
        global_cond=obs_cond,
    )
    # Per-sample MSE (average over action dims and prediction horizon).
    mse = F.mse_loss(noise_pred, noise, reduction="none")  # (B, Tp, Da)
    return mse.mean(dim=(1, 2))  # (B,)


def encode_obs(nets, obs_dict, obs_shapes, device):
    """Encode observations → obs_cond vector [B, To*D].

    Mimics ``DiffusionPolicyUNet._get_action_trajectory``'s encoding step.
    """
    inputs = {"obs": obs_dict, "goal": None}
    for k in obs_shapes:
        if inputs["obs"][k].ndim - 2 != len(obs_shapes[k]):
            # Should not happen after process_batch_for_training, but be safe.
            if inputs["obs"][k].ndim - 1 == len(obs_shapes[k]):
                inputs["obs"][k] = inputs["obs"][k].unsqueeze(1)

    obs_features = TensorUtils.time_distributed(
        inputs, nets["policy"]["obs_encoder"], inputs_as_kwargs=True
    )  # (B, To, D)
    return obs_features.flatten(start_dim=1)  # (B, To*D)


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis loop
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_cmi_analysis(
    inner_policy,
    dataset,
    obs_norm_stats_np,
    batch_size: int = 64,
    num_batches: int | None = None,
    num_t_samples: int = 8,
):
    """Iterate over ``dataset`` and compute per-modality CMI estimates.

    Args:
        inner_policy: the ``DiffusionPolicyUNet`` Algo object (frozen).
        dataset: a ``SequenceDataset``.
        obs_norm_stats_np: observation normalisation stats (numpy dicts from
            the checkpoint).
        batch_size: DataLoader batch size.
        num_batches: if given, stop after this many batches (useful for speed).
        num_t_samples: how many diffusion timesteps to sample per data sample.

    Returns:
        results: dict  –  ``{obs_key: {"cmi_samples": list[float]}}``
    """
    device = inner_policy.device
    nets = _get_nets(inner_policy)
    nets.eval()
    noise_scheduler = inner_policy.noise_scheduler
    num_train_timesteps = noise_scheduler.config.num_train_timesteps
    obs_shapes = inner_policy.obs_shapes

    To = inner_policy.algo_config.horizon.observation_horizon
    Tp = inner_policy.algo_config.horizon.prediction_horizon

    # Convert obs_norm_stats to tensors on device (done once).
    obs_norm_stats_t = None
    if obs_norm_stats_np is not None:
        obs_norm_stats_t = TensorUtils.to_float(
            TensorUtils.to_device(TensorUtils.to_tensor(obs_norm_stats_np), device)
        )

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
    )

    used_keys = list(obs_shapes.keys())
    # Accumulators: per-key list of per-sample CMI values.
    cmi_all = {k: [] for k in used_keys}
    # Also accumulate per-timestep for richer plots.
    cmi_by_t = {k: {} for k in used_keys}

    n_batches_done = 0
    for batch in tqdm(loader, desc="CMI analysis", total=num_batches):
        if num_batches is not None and n_batches_done >= num_batches:
            break

        # ── 1. Prepare batch on device ────────────────────────────────
        batch = TensorUtils.to_float(TensorUtils.to_device(batch, device))

        raw_obs = batch["obs"]  # {key: (B, T_seq, ...)}
        actions = batch["actions"][:, :Tp, :]  # (B, Tp, Da)
        B = actions.shape[0]

        # Slice observation horizon.
        obs_full_raw = {k: raw_obs[k][:, :To] for k in used_keys}

        for _ in range(num_t_samples):
            # ── 2. Sample shared noise & timestep ─────────────────────
            noise = torch.randn_like(actions)
            timesteps = torch.randint(0, num_train_timesteps, (B,), device=device).long()

            # ── 3. Full-conditioning MSE ──────────────────────────────
            obs_full_proc = _preprocess_and_normalise(obs_full_raw, obs_norm_stats_t, device)
            cond_full = encode_obs(nets, obs_full_proc, obs_shapes, device)
            mse_full = compute_denoising_mse(
                nets, noise_scheduler, cond_full, actions, noise, timesteps
            )  # (B,)

            # ── 4. Per-key ablation MSE ───────────────────────────────
            for key in used_keys:
                obs_ablated_raw = _zero_obs_key(obs_full_raw, key, obs_norm_stats_t)
                obs_ablated_proc = _preprocess_and_normalise(obs_ablated_raw, obs_norm_stats_t, device)
                cond_ablated = encode_obs(nets, obs_ablated_proc, obs_shapes, device)
                mse_ablated = compute_denoising_mse(
                    nets, noise_scheduler, cond_ablated, actions, noise, timesteps
                )  # (B,)

                # CMI ∝ MSE_without_k − MSE_full  (positive ⇒ k helps)
                cmi_samples = (mse_ablated - mse_full).cpu().numpy().tolist()
                cmi_all[key].extend(cmi_samples)

                # Per-timestep breakdown
                for b_idx in range(B):
                    t_val = int(timesteps[b_idx].item())
                    cmi_by_t[key].setdefault(t_val, []).append(cmi_samples[b_idx])

        n_batches_done += 1

    return cmi_all, cmi_by_t


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_cmi_bar(cmi_all, save_path):
    """Bar chart of average CMI per modality."""
    keys = sorted(cmi_all.keys())
    means = [np.mean(cmi_all[k]) if cmi_all[k] else 0.0 for k in keys]
    stds = [np.std(cmi_all[k]) / max(1, np.sqrt(len(cmi_all[k]))) if cmi_all[k] else 0.0 for k in keys]

    fig, ax = plt.subplots(figsize=(max(6, len(keys) * 1.2), 5))
    x = np.arange(len(keys))
    ax.bar(x, means, yerr=stds, capsize=4, color="steelblue", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("CMI estimate  (MSE_without_k − MSE_full)")
    ax.set_title("Conditional Mutual Information: I(action ; modality | rest)")
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved CMI bar chart → {save_path}")


def plot_cmi_vs_timestep(cmi_by_t, save_path):
    """Line plot of mean CMI vs diffusion timestep for each modality."""
    keys = sorted(cmi_by_t.keys())
    fig, ax = plt.subplots(figsize=(10, 6))
    for k in keys:
        t_dict = cmi_by_t[k]
        if not t_dict:
            continue
        ts = sorted(t_dict.keys())
        means = [np.mean(t_dict[t]) for t in ts]
        ax.plot(ts, means, label=k, linewidth=1.5)

    ax.set_xlabel("Diffusion timestep t")
    ax.set_ylabel("CMI estimate")
    ax.set_title("CMI vs Diffusion Timestep")
    ax.legend(loc="upper right", fontsize="small")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    base, ext = os.path.splitext(save_path)
    ts_path = f"{base}_vs_timestep{ext}"
    plt.savefig(ts_path, dpi=150)
    plt.close()
    print(f"Saved CMI-vs-timestep plot → {ts_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Conditional Mutual Information analysis for diffusion policies."
    )
    parser.add_argument(
        "--agent", type=str, required=True, help="Path to checkpoint (.pth)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to HDF5 dataset. If omitted, uses the path stored in the checkpoint config.",
    )
    parser.add_argument(
        "--plot_path",
        type=str,
        default="cmi_results.png",
        help="Output path for the CMI bar chart.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for evaluation."
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=50,
        help="Number of batches to evaluate (0 = entire dataset).",
    )
    parser.add_argument(
        "--num_t_samples",
        type=int,
        default=8,
        help="Number of diffusion-timestep samples per data sample.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")

    args = parser.parse_args()

    # ── Setup ─────────────────────────────────────────────────────────
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    rollout_policy, ckpt_dict, config = _load_policy_and_config(args.agent, device)
    inner_policy = rollout_policy.policy
    inner_policy.set_eval()

    # Observation normalisation stats (numpy).
    obs_norm_stats = ckpt_dict.get("obs_normalization_stats", None)
    if obs_norm_stats is not None:
        for m in obs_norm_stats:
            for k in obs_norm_stats[m]:
                obs_norm_stats[m][k] = np.array(obs_norm_stats[m][k])

    # ── Dataset ───────────────────────────────────────────────────────
    dataset = _build_dataset(config, ckpt_dict, dataset_path=args.dataset)
    print(f"Dataset length: {len(dataset)}")

    # ── Run analysis ──────────────────────────────────────────────────
    num_batches = args.num_batches if args.num_batches > 0 else None
    cmi_all, cmi_by_t = run_cmi_analysis(
        inner_policy,
        dataset,
        obs_norm_stats,
        batch_size=args.batch_size,
        num_batches=num_batches,
        num_t_samples=args.num_t_samples,
    )

    # ── Print summary ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Conditional Mutual Information estimates")
    print("  I(action ; modality_k | remaining_modalities)")
    print("  ∝  E[ MSE_without_k  −  MSE_full ]")
    print("=" * 60)
    for k in sorted(cmi_all):
        vals = cmi_all[k]
        if vals:
            mean = np.mean(vals)
            se = np.std(vals) / np.sqrt(len(vals))
            print(f"  {k:30s}  CMI = {mean:+.6f}  (±{se:.6f},  n={len(vals)})")
        else:
            print(f"  {k:30s}  (no samples)")

    # ── Plots ─────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.plot_path)) or ".", exist_ok=True)
    plot_cmi_bar(cmi_all, args.plot_path)
    plot_cmi_vs_timestep(cmi_by_t, args.plot_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
