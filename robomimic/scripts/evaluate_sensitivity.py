"""
Script to evaluate sensitivity of a trained policy to observation perturbations.
"""
import argparse
import json
import numpy as np
import copy
import torch
import matplotlib.pyplot as plt
import imageio
from io import BytesIO

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.envs.env_base import EnvBase
from robomimic.envs.wrappers import EnvWrapper
from robomimic.algo import RolloutPolicy
from tqdm import tqdm


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def create_perturbed_observation(obs, key, obs_norm_stats):
    """Create a perturbed version of the observation by setting values so normalized inputs equal zero.

    This sets observation values such that after preprocessing and normalization the policy
    will receive zero for that input channel. For image-like modalities (rgb/depth) this
    uses the normalization offset and multiplies by 255 to get raw values; for others it
    uses the offset directly. If normalization stats are absent it falls back to zeros.

    Args:
        obs: Original observation dict
        key: Key to perturb
        obs_norm_stats: Normalization statistics (offset and scale)

    Returns:
        Perturbed observation dict
    """
    obs_perturbed = copy.deepcopy(obs)
    val = obs[key]

    # Always zero inputs by setting them to the normalization offset

    if obs_norm_stats is not None and key in obs_norm_stats:
        offset_stat = np.array(obs_norm_stats[key].get("offset", 0.0))
        offset_stat = offset_stat[0] if offset_stat.ndim > 0 else offset_stat
    else:
        # throw an error
        raise ValueError(f"No normalization stats for key {key}")

    if ObsUtils.key_is_obs_modality(key, "rgb") or ObsUtils.key_is_obs_modality(key, "depth"):
        preproc_mult = 255.0
        # for images, the offset is of shape (channels, h, w), need to expand to match val shape
        # but the input is of shape (h, w, channels)
        perturbed_val = offset_stat.transpose(1, 2, 0) * preproc_mult
    else:
        perturbed_val = offset_stat

    # broadcast the dims and shape to match original
    perturbed_val = np.broadcast_to(perturbed_val, val.shape).astype(val.dtype)

    obs_perturbed[key] = perturbed_val

    return obs_perturbed


def create_sensitivity_video_frame(env, step_sensitivities, step_i, img_height, img_width):
    """Create a video frame combining environment render and sensitivity time series plot.
    
    Args:
        env: Environment instance
        step_sensitivities: List of sensitivity dicts (history up to current step)
        step_i: Current step number
        img_height: Target image height
        img_width: Target image width
        
    Returns:
        Combined image array or None if rendering fails
    """
    try:
        env_img = env.render(mode="rgb_array", height=img_height, width=img_width)
    except Exception:
        return None

    if env_img is None:
        return None

    # Extract all unique keys from sensitivity history
    all_keys = set()
    for s in step_sensitivities:
        all_keys.update(s.keys())
    all_keys.discard("step")
    keys = sorted(list(all_keys))

    if len(keys) == 0:
        return env_img

    # Build time series data
    steps = [s.get("step", i) for i, s in enumerate(step_sensitivities)]
    
    # Create line plot of sensitivities vs time
    plot_w = max(300, img_width // 2)
    fig = plt.figure(figsize=(plot_w / 100.0, img_height / 100.0), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    
    for k in keys:
        series = [s.get(k, 0.0) for s in step_sensitivities]
        ax.plot(steps, series, label=k if len(k) <= 15 else k[:14] + "…", linewidth=1.5)
    
    ax.set_xlabel("Step", fontsize=8)
    ax.set_ylabel("Sensitivity", fontsize=8)
    ax.set_title(f"Sensitivities vs Time (Step {step_i})", fontsize=9)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="upper right", fontsize=6)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # Convert figure to image
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    plot_img = buf.reshape((h, w, 3))
    plt.close(fig)

    # Ensure plot_img height matches env_img height
    if plot_img.shape[0] != env_img.shape[0]:
        try:
            from PIL import Image
            plot_pil = Image.fromarray(plot_img)
            plot_pil = plot_pil.resize((plot_img.shape[1], env_img.shape[0]))
            plot_img = np.array(plot_pil)
        except Exception:
            # Fallback: pad or crop
            if plot_img.shape[0] > env_img.shape[0]:
                plot_img = plot_img[:env_img.shape[0], ...]
            else:
                pad_h = env_img.shape[0] - plot_img.shape[0]
                pad = np.zeros((pad_h, plot_img.shape[1], 3), dtype=plot_img.dtype)
                plot_img = np.concatenate([plot_img, pad], axis=0)

    # Concatenate horizontally
    try:
        combined = np.concatenate([env_img, plot_img], axis=1)
    except Exception:
        # Fallback to just writing env_img
        combined = env_img

    return combined 


def run_sensitivity_analysis(policy, env, horizon, record_video=False, img_height=256, img_width=256):
    """Run one rollout and compute sensitivity analysis by zeroing out all inputs.

    Args:
        policy (RolloutPolicy): loaded policy
        env (EnvBase): loaded environment
        horizon (int): max steps
        record_video (bool): whether to collect frames in-memory for later writing
    Returns:
        (step_sensitivities, success_flag, frames)
    """
    assert isinstance(policy, RolloutPolicy)
    print("Starting sensitivity analysis rollout (zeroing all inputs in normalized space)...")

    policy.start_episode()
    obs = env.reset()
    state_dict = env.get_state()
    # helpful for deterministic playback if needed
    obs = env.reset_to(state_dict)
    
    # Get normalization stats if available
    obs_norm_stats = policy.obs_normalization_stats
    used_keys = policy.policy.global_config.all_obs_keys
    step_sensitivities = []

    frames = [] if record_video else None
    success_flag = False

    try:
        for step_i in tqdm(range(horizon)):
            # Get original action
            act_original = policy(ob=obs)
            
            # Compute sensitivity for each observation key
            l2_diffs = {}
            
            
            for k in used_keys:
                # Create perturbed observation (set so normalized input is zero)
                obs_perturbed = create_perturbed_observation(
                    obs, k, obs_norm_stats
                )
                
                # Get perturbed action and compute L2 difference
                act_perturbed = policy(ob=obs_perturbed)
                diff = act_perturbed - act_original
                l2_diffs[k] = np.linalg.norm(diff)

            # Compute softmax over L2 norms and record results
            keys = list(l2_diffs.keys())
            if len(keys) > 0:
                vals = np.array([l2_diffs[k] for k in keys])
                softmax_vals = softmax(vals)

                step_data = {k: float(v) for k, v in zip(keys, softmax_vals)}
                step_data["step"] = step_i
                step_sensitivities.append(step_data)

                # Create and collect video frame if requested
                if record_video:
                    frame = create_sensitivity_video_frame(
                        env, step_sensitivities, step_i, img_height, img_width
                    )
                    if frame is not None:
                        frames.append(frame)

            # Step environment
            next_obs, r, done, _ = env.step(act_original)
            success = env.is_success()["task"]
            
            if done or success:
                # record whether the rollout ended with a success
                success_flag = bool(success)
                break
                
            obs = next_obs

    except Exception as e:
        print(f"Error during rollout: {e}")
        import traceback
        traceback.print_exc()

    return step_sensitivities, success_flag, frames


def evaluate_sensitivity(args):
    # relative path to agent
    ckpt_path = args.agent
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    # restore policy
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=True)

    # read rollout settings
    config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
    rollout_horizon = config.experiment.rollout.horizon
    if args.horizon is not None:
        rollout_horizon = args.horizon

    # create environment from saved checkpoint
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict, 
        env_name=args.env, 
        render=False, 
        render_offscreen=False, 
        verbose=True,
    )

    # set seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # Run Analysis; collect frames in-memory only if a video path is requested
    collect_video = args.video_path is not None

    # Retry rollouts until success (or until max attempts reached if provided)
    attempts = 0
    max_rollouts = int(getattr(args, "max_rollouts", 0))
    results = []
    success = False
    frames = None
    while True:
        attempts += 1
        print(f"Starting rollout attempt {attempts}...")
        results, success, frames = run_sensitivity_analysis(
            policy, env, rollout_horizon,
            record_video=collect_video, img_height=args.img_height, img_width=args.img_width,
        )
        if success:
            print(f"Rollout attempt {attempts} succeeded.")
            break
        else:
            print(f"Rollout attempt {attempts} did not succeed.")
            if max_rollouts > 0 and attempts >= max_rollouts:
                print(f"Reached max attempts ({max_rollouts}); stopping retries.")
                break
            print("Retrying rollout...")

    # Compute average sensitivity across the trajectory
    if len(results) > 0:
        avg_sensitivity = {}
        # Union of all keys
        all_keys = set().union(*[r.keys() for r in results])
        all_keys.discard("step")

        print("\n=== Sensitivity Analysis Results (Average Softmaxed L2 Norm) ===")
        for k in all_keys:
            vals = [r.get(k, 0.0) for r in results]
            avg_val = np.mean(vals)
            avg_sensitivity[k] = avg_val
            print(f"{k}: {avg_val:.4f}")

        # Plot sensitivities vs time for each key
        # Build time series (results are recorded per step)
        steps = [int(r.get("step", i)) for i, r in enumerate(results)]
        # sort results by step just in case
        sorted_idx = np.argsort(steps)
        steps_sorted = [steps[i] for i in sorted_idx]

        def smooth_series(series, window):
            if window is None or window <= 1:
                return np.asarray(series)
            w = int(window)
            if w <= 1:
                return np.asarray(series)
            kernel = np.ones(w, dtype=float) / float(w)
            return np.convolve(np.asarray(series, dtype=float), kernel, mode="same")

        keys = sorted(list(all_keys))
        plt.figure(figsize=(10, 6))
        for k in keys:
            series = [results[i].get(k, 0.0) for i in sorted_idx]
            series_s = smooth_series(series, args.smoothing_window)
            plt.plot(steps_sorted, series_s, label=k)

        plt.xlabel("Step")
        plt.ylabel("Softmaxed sensitivity")
        plt.title("Sensitivities vs Time")
        plt.legend(loc="upper right", fontsize="small")
        plt.tight_layout()

        save_path = args.plot_path if hasattr(args, "plot_path") and args.plot_path is not None else "sensitivity_vs_time.png"

        # Only save outputs if the rollout succeeded
        if success:
            plt.savefig(save_path)
            print(f"Saved sensitivity plot to {save_path}")

            if args.video_path is not None and frames is not None:
                w = imageio.get_writer(args.video_path, fps=args.video_fps)
                for fr in frames:
                    w.append_data(fr)
                w.close()
                print(f"Saved sensitivity video to {args.video_path}")
        else:
            print("Task did not succeed; skipping saving plot and video.")

    else:
        print("No results collected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent",
        type=str,
        default="/home/jiuzl/robomimic_suite/logs/diffusion_policy_tool_hang/high_frequency_contact_force_vision/20260127133123/last.pth",
        help="path to saved checkpoint pth file",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=800,
        help="maximum horizon of rollout",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="override env name",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed",
    )
    parser.add_argument(
        "--plot_path",
        type=str,
        default="/home/jiuzl/robomimic_suite/logs/diffusion_policy_tool_hang/high_frequency_contact_force_vision/20260127133123/sensitivity_vs_time.png",
        help="path to save sensitivities vs time plot (default: sensitivity_vs_time.png)",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default="/home/jiuzl/robomimic_suite/logs/diffusion_policy_tool_hang/high_frequency_contact_force_vision/20260127133123/sensitivity_video.mp4",
        help="path to save combined video of obs and sensitivities (mp4 or gif)",
    )
    parser.add_argument(
        "--video_fps",
        type=int,
        default=20,
        help="frames per second for saved video",
    )
    parser.add_argument(
        "--img_height",
        type=int,
        default=256,
        help="height for rendered env image",
    )
    parser.add_argument(
        "--img_width",
        type=int,
        default=256,
        help="width for rendered env image",
    )
    parser.add_argument(
        "--smoothing_window",
        type=int,
        default=11,
        help="moving-average smoothing window (odd int). 1 means no smoothing.",
    )
    parser.add_argument(
        "--max_rollouts",
        type=int,
        default=0,
        help="Maximum number of rollout attempts to try; 0 means keep trying until success",
    )

    # Always zero all inputs in normalized space; no CLI flag required

    args = parser.parse_args()
    evaluate_sensitivity(args)
