"""
Saliency analysis: run policy rollouts until success, and (on the successful
rollout) compute gradient-norm saliency per observation key over time.

Behavior:
  1) Saliency computation is **decoupled** from env action selection.  The env
     is always stepped with the standard rollout_policy action (which applies
     action denormalization, rot_6d conversion, and action chunking), keeping
     the trajectory faithful to the trained agent.
  2) Allows overriding diffusion inference steps *for saliency only* via
     --saliency_inference_steps (default: 20).
  3) Optionally computes saliency only every N steps via --saliency_every.
  4) Repeats rollouts until success (or max attempts). Only saves plot/video
     for the successful rollout.

Notes:
  - On saliency steps, an extra forward + backward pass is run to compute
    gradients; the env action still comes from the rollout_policy.
  - When saliency is skipped (step_i % saliency_every != 0), no extra
    computation is done.
"""

import argparse
from collections import OrderedDict

import imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_obs_encoder(policy):
    """Return the ObservationEncoder that sits inside the policy."""
    nets = policy.nets
    if hasattr(policy, "ema") and policy.ema is not None:
        nets = policy.ema.averaged_model
    obs_group_enc = nets["policy"]["obs_encoder"]
    return obs_group_enc.nets["obs"]


def _prepare_obs_with_grad(rollout_policy, ob):
    """Preprocess observation like RolloutPolicy._prepare_observation, but keep grads."""
    ob = TensorUtils.to_tensor(ob)
    ob = TensorUtils.to_batch(ob)
    ob = TensorUtils.to_device(ob, rollout_policy.policy.device)
    ob = TensorUtils.to_float(ob)

    # Process images (divide by 255, channel-first, etc.)
    for k in ob:
        if ObsUtils.key_is_obs_modality(key=k, obs_modality="rgb") or \
           ObsUtils.key_is_obs_modality(key=k, obs_modality="depth"):
            ob[k] = ObsUtils.process_obs(obs=ob[k], obs_key=k)

    # Normalize
    if rollout_policy.obs_normalization_stats is not None:
        stats = TensorUtils.to_float(
            TensorUtils.to_device(
                TensorUtils.to_tensor(rollout_policy.obs_normalization_stats),
                rollout_policy.policy.device,
            )
        )
        ob = {k: ob[k] for k in rollout_policy.policy.global_config.all_obs_keys}
        ob = ObsUtils.normalize_dict(ob, normalization_stats=stats)

    # Enable gradients
    for k in ob:
        if ob[k].is_floating_point():
            ob[k] = ob[k].detach().requires_grad_(True)

    return ob


def _encode_per_key(obs_encoder, obs_dict):
    """Run the ObservationEncoder but return per-key feature tensors (before concat)."""
    features = OrderedDict()
    feats_list = []
    for k in obs_encoder.obs_shapes:
        x = obs_dict[k]
        for rand in obs_encoder.obs_randomizers[k]:
            if rand is not None:
                x = rand.forward_in(x)
        if obs_encoder.obs_nets[k] is not None:
            x = obs_encoder.obs_nets[k](x)
            if obs_encoder.activation is not None:
                x = obs_encoder.activation(x)
        for rand in reversed(obs_encoder.obs_randomizers[k]):
            if rand is not None:
                x = rand.forward_out(x)
        x = TensorUtils.flatten(x, begin_axis=1)

        if x.requires_grad:
            x.retain_grad()

        features[k] = x
        feats_list.append(x)

    combined = torch.cat(feats_list, dim=-1)
    return features, combined


def _run_diffusion_with_cond(policy, obs_cond, num_steps_override=None):
    """Run diffusion denoising loop given a pre-computed obs_cond vector."""
    algo = policy  # DiffusionPolicyUNet
    To = algo.algo_config.horizon.observation_horizon
    Ta = algo.algo_config.horizon.action_horizon
    Tp = algo.algo_config.horizon.prediction_horizon
    action_dim = algo.ac_dim

    if algo.algo_config.ddpm.enabled:
        num_steps = algo.algo_config.ddpm.num_inference_timesteps
    elif algo.algo_config.ddim.enabled:
        num_steps = algo.algo_config.ddim.num_inference_timesteps
    else:
        raise ValueError("Neither DDPM nor DDIM is enabled.")

    if num_steps_override is not None:
        num_steps = int(num_steps_override)

    nets = algo.nets
    if algo.ema is not None:
        nets = algo.ema.averaged_model

    B = obs_cond.shape[0]
    naction = torch.randn((B, Tp, action_dim), device=algo.device)

    algo.noise_scheduler.set_timesteps(num_steps)
    for k in algo.noise_scheduler.timesteps:
        noise_pred = nets["policy"]["noise_pred_net"](
            sample=naction, timestep=k, global_cond=obs_cond
        )
        naction = algo.noise_scheduler.step(
            model_output=noise_pred, timestep=k, sample=naction
        ).prev_sample

    start = To - 1
    end = start + Ta
    return naction[:, start:end]


def compute_saliency_step(
    obs_dict_with_grad,
    obs_encoder,
    inner_policy,
    saliency_inference_steps=10,
    saliency_time_index=-1,
):
    """Compute per-key gradient saliency for one env step.

    Returns:
        saliency (dict): per-key gradient norm values.
    """
    # Ensure time dimension exists if horizon/frame-stack > 1
    obs_input = {}
    for k in inner_policy.obs_shapes:
        t = obs_dict_with_grad[k]
        if t.ndim - 1 == len(inner_policy.obs_shapes[k]):
            t = t.unsqueeze(1)
        obs_input[k] = t

    B, T = next(iter(obs_input.values())).shape[:2]

    time_features_list = []
    time_combined_list = []
    for t_idx in range(T):
        obs_t = {k: obs_input[k][:, t_idx] for k in obs_input}
        feats_t, combined_t = _encode_per_key(obs_encoder, obs_t)
        time_features_list.append(feats_t)
        time_combined_list.append(combined_t)

    obs_features = torch.stack(time_combined_list, dim=1)  # [B, T, D]
    obs_cond = obs_features.flatten(start_dim=1)  # [B, T*D]

    # deterministic denoising for stable gradients
    devices = [inner_policy.device] if inner_policy.device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(0)
        action_chunk = _run_diffusion_with_cond(
            inner_policy,
            obs_cond,
            num_steps_override=saliency_inference_steps,
        )

    loss = action_chunk.norm()
    loss.backward()

    t_sel = saliency_time_index if saliency_time_index >= 0 else (T + saliency_time_index)
    t_sel = int(np.clip(t_sel, 0, T - 1))
    sel_feats = time_features_list[t_sel]

    saliency = {}
    for k, feat in sel_feats.items():
        g = feat.grad
        saliency[k] = float(g.detach().norm().cpu()) if g is not None else 0.0

    first_action = action_chunk[0, 0].detach().cpu().numpy()

    # best-effort grad cleanup (all model parameters, not just noise_pred_net)
    try:
        nets = inner_policy.nets
        if inner_policy.ema is not None:
            nets = inner_policy.ema.averaged_model
        for p in nets.parameters():
            p.grad = None
    except Exception:
        pass

    return saliency, first_action


# ---------------------------------------------------------------------------
# Video helper
# ---------------------------------------------------------------------------

def create_saliency_video_frame(env, saliency_history, step_i, img_h, img_w):
    """Render env + saliency time-series side-by-side."""
    try:
        env_img = env.render(mode="rgb_array", height=img_h, width=img_w)
    except Exception:
        return None
    if env_img is None:
        return None

    all_keys = set()
    for s in saliency_history:
        all_keys.update(k for k in s if k != "step")
    keys = sorted(all_keys)
    if not keys:
        return env_img

    steps = [s.get("step", i) for i, s in enumerate(saliency_history)]
    plot_w = max(300, img_w // 2)
    fig, ax = plt.subplots(figsize=(plot_w / 100, img_h / 100), dpi=100)
    for k in keys:
        series = []
        for s in saliency_history:
            vals = [abs(s.get(k2, 0.0)) for k2 in keys]
            total = float(sum(vals))
            series.append(abs(s.get(k, 0.0)) / total if total > 0 else 0.0)
        ax.plot(steps, series, label=k if len(k) <= 20 else k[:19] + "…", linewidth=1.5)
    ax.set_xlabel("Step", fontsize=8)
    ax.set_ylabel("Relative saliency (fraction)", fontsize=8)
    ax.set_title(f"Relative Gradient Saliency (step {step_i})", fontsize=9)
    ax.legend(loc="upper right", fontsize=6)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    plot_img = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    if plot_img.shape[0] != env_img.shape[0]:
        from PIL import Image as PILImage
        plot_img = np.array(
            PILImage.fromarray(plot_img).resize((plot_img.shape[1], env_img.shape[0]))
        )

    return np.concatenate([env_img, plot_img], axis=1)


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def run_saliency_rollout(
    rollout_policy,
    env,
    horizon,
    record_video=False,
    img_height=256,
    img_width=256,
    saliency_inference_steps=10,
    saliency_every=1,
    show_step_prints=False,
):
    """Run one rollout, computing saliency as configured."""
    policy = rollout_policy
    inner = policy.policy
    obs_encoder = _get_obs_encoder(inner)

    policy.start_episode()
    obs = env.reset()
    state_dict = env.get_state()
    obs = env.reset_to(state_dict)

    saliency_history = []
    frames = [] if record_video else None
    success_flag = False

    inner.set_eval()
    inner.reset()

    prev_sal = None

    for step_i in tqdm(range(horizon), desc="Saliency rollout"):
        do_sal = (saliency_every <= 1) or ((step_i % saliency_every) == 0)

        if do_sal:
            obs_g = _prepare_obs_with_grad(policy, obs)
            sal, _ = compute_saliency_step(
                obs_dict_with_grad=obs_g,
                obs_encoder=obs_encoder,
                inner_policy=inner,
                saliency_inference_steps=saliency_inference_steps,
                saliency_time_index=-1,
            )
            prev_sal = sal
            if show_step_prints:
                print("step", step_i, "saliency", {k: f"{v:.4f}" for k, v in sal.items()})
        else:
            sal = dict(prev_sal) if prev_sal is not None else {}

        # Always get the env action through the RolloutPolicy so that
        # action denormalization, rot_6d conversion, and action chunking
        # are applied identically to run_trained_agent.py.
        act = policy(ob=obs)

        sal["step"] = step_i
        saliency_history.append(sal)

        if record_video:
            frame = create_saliency_video_frame(
                env, saliency_history, step_i, img_height, img_width
            )
            if frame is not None:
                frames.append(frame)

        next_obs, r, done, _ = env.step(act)
        success = env.is_success().get("task", False)

        if done or success:
            success_flag = bool(success)
            break

        obs = next_obs

    return saliency_history, success_flag, frames


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def smooth_series(series, window):
    if window is None or window <= 1:
        return np.asarray(series)
    w = int(window)
    kernel = np.ones(w, dtype=float) / float(w)
    return np.convolve(np.asarray(series, dtype=float), kernel, mode="same")


def make_saliency_plot(results, smoothing_window=1, save_path="saliency_vs_time.png"):
    all_keys = set()
    for r in results:
        all_keys.update(k for k in r if k != "step")
    keys = sorted(all_keys)
    steps = [r["step"] for r in results]

    plt.figure(figsize=(10, 6))
    for k in keys:
        series = []
        for r in results:
            vals = [abs(r.get(k2, 0.0)) for k2 in keys]
            total = float(sum(vals))
            series.append(abs(r.get(k, 0.0)) / total if total > 0 else 0.0)
        plt.plot(steps, smooth_series(series, smoothing_window), label=k)

    plt.xlabel("Step")
    plt.ylabel("Relative saliency (fraction)")
    plt.title("Relative Gradient Saliency vs Time")
    plt.legend(loc="upper right", fontsize="small")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved saliency plot to {save_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Gradient-based saliency analysis for a trained policy (repeat until success)."
    )
    parser.add_argument("--agent", type=str, required=True, help="Path to checkpoint (.pth)")
    parser.add_argument("--horizon", type=int, default=700)
    parser.add_argument("--env", type=str, default=None, help="Override env name")
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--plot_path", type=str, default="saliency_vs_time.png")
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--video_fps", type=int, default=20)
    parser.add_argument("--img_height", type=int, default=256)
    parser.add_argument("--img_width", type=int, default=256)
    parser.add_argument("--smoothing_window", type=int, default=11)

    # Speed controls
    parser.add_argument("--saliency_inference_steps", type=int, default=100)
    parser.add_argument("--saliency_every", type=int, default=1)

    # Repeat-until-success controls
    parser.add_argument("--max_rollouts", type=int, default=20,
                        help="Max attempts before giving up (0 = infinite)")
    parser.add_argument("--reset_seed_each_rollout", action="store_true",
                        help="If set, re-seed RNG each rollout using (seed + rollout_idx)")
    parser.add_argument("--print_step_saliency", action="store_true",
                        help="Print per-step saliency values (can be slow/noisy)")

    args = parser.parse_args()

    # Optional speed tweak (PyTorch 2+)
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # --- Load policy ---
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    rollout_policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=args.agent, device=device, verbose=True
    )

    config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
    rollout_horizon = args.horizon if args.horizon is not None else config.experiment.rollout.horizon

    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict,
        env_name=args.env,
        render=False,
        render_offscreen=False,
        verbose=True,
    )

    base_seed = args.seed

    # --- Repeat rollouts until success ---
    attempt = 0
    best_failure_len = -1
    best_failure_results = None

    while True:
        attempt += 1
        if args.max_rollouts > 0 and attempt > args.max_rollouts:
            print(f"\nReached max_rollouts={args.max_rollouts} without success.")
            # No plot saved (as requested). We can optionally report best failure stats.
            if best_failure_results is not None:
                print(f"Longest failure rollout length: {best_failure_len} steps.")
            return

        # Seeding
        if base_seed is not None:
            if args.reset_seed_each_rollout:
                seed_i = int(base_seed) + (attempt - 1)
            else:
                seed_i = int(base_seed)
            np.random.seed(seed_i)
            torch.manual_seed(seed_i)

        collect_video = args.video_path is not None

        print(f"\n=== Rollout attempt {attempt} ===")
        results, success, frames = run_saliency_rollout(
            rollout_policy=rollout_policy,
            env=env,
            horizon=rollout_horizon,
            record_video=collect_video,
            img_height=args.img_height,
            img_width=args.img_width,
            saliency_inference_steps=args.saliency_inference_steps,
            saliency_every=args.saliency_every,
            show_step_prints=args.print_step_saliency,
        )
        print(f"Attempt {attempt}: {'SUCCESS' if success else 'FAIL'} (steps={len(results)})")

        if not results:
            print("No results collected on this attempt (error or immediate termination).")
            continue

        if not success:
            # track longest failure for debugging prints
            if len(results) > best_failure_len:
                best_failure_len = len(results)
                best_failure_results = results
            continue

        # ---- Success: print average saliency and save outputs ----
        all_keys = set()
        for r in results:
            all_keys.update(k for k in r if k != "step")

        print("\n=== Average Gradient Saliency (raw grad-norm) [successful rollout] ===")
        for k in sorted(all_keys):
            vals = [r.get(k, 0.0) for r in results]
            print(f"  {k}: {np.mean(vals):.6f}")

        make_saliency_plot(results, args.smoothing_window, args.plot_path)

        if args.video_path is not None and frames:
            w = imageio.get_writer(args.video_path, fps=args.video_fps)
            for fr in frames:
                w.append_data(fr)
            w.close()
            print(f"Saved saliency video to {args.video_path}")

        print(f"\nDone. Saved plot/video from attempt {attempt}.")
        return


if __name__ == "__main__":
    main()
