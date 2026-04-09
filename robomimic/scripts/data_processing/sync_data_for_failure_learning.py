"""Synchronize image observations with low-dimensional robot data
with optional image resizing and frame skipping for failure-learning datasets.

This is a lightweight variant based on sync_image_low_dim.py but adds
`--resize_h`/`--resize_w` to resize images before storing and `--skip_n`
to skip N frames after saving a frame (useful to downsample data).
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Iterable, List, Tuple

import h5py
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync images with low-dim data with optional resize/skip"
    )
    parser.add_argument("--image-h5", required=True, help="Path to the image HDF5 file")
    parser.add_argument("--lowdim-h5", required=True, help="Path to the low-dimensional HDF5 file")
    parser.add_argument("--output-h5", required=True, help="Destination path for the synchronized HDF5")
    parser.add_argument(
        "--image-timestamp-key",
        default="rs_timestamp",
        help="Dataset key holding timestamps inside the image obs group",
    )
    parser.add_argument(
        "--lowdim-timestamp-key",
        default="timestamp",
        help="Dataset key holding timestamps inside the low-dimensional obs group",
    )
    parser.add_argument(
        "--image-keys",
        nargs="*",
        help="Optional list of image observation keys to copy (defaults to all datasets except the timestamp)",
    )
    parser.add_argument(
        "--lowdim-keys",
        nargs="*",
        help="Optional list of low-dimensional observation keys to sync (defaults to all datasets except the timestamp)",
    )
    parser.add_argument("--resize_h", type=int, default=None, help="Resize image height (pixels)")
    parser.add_argument("--resize_w", type=int, default=None, help="Resize image width (pixels)")
    parser.add_argument(
        "--skip_n",
        type=int,
        default=0,
        help="Skip N frames after saving one frame (0 means keep all frames)",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip demos that miss required keys instead of raising an error",
    )
    parser.add_argument(
        "--pos-key",
        default="robot0_eef_pos",
        help="Low-dim key for end-effector position used in action computation (default: robot0_eef_pos)",
    )
    parser.add_argument(
        "--quat-key",
        default="robot0_eef_quat",
        help="Low-dim key for end-effector quaternion (x,y,z,w) used in action computation (default: robot0_eef_quat)",
    )
    parser.add_argument(
        "--gripper-key",
        default="robot0_gripper_qpos",
        help="Low-dim key for gripper joint position used in action computation (default: robot0_gripper_qpos)",
    )
    return parser.parse_args()


def validate_files(*paths: str) -> None:
    missing = [path for path in paths if not os.path.exists(path)]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(f"Missing required file(s): {joined}")


def resolve_dataset_keys(group: h5py.Group, timestamp_key: str, explicit: Iterable[str] | None) -> List[str]:
    def _filter(keys: Iterable[str]) -> List[str]:
        return [k for k in keys if "timestamp" not in k.lower()]

    if explicit:
        explicit = list(explicit)
        missing = [k for k in explicit if k not in group]
        if missing:
            raise KeyError(f"Group {group.name} missing requested keys: {missing}")
        filtered = _filter(explicit)
        if not filtered:
            raise KeyError("No valid keys remain after removing timestamp datasets")
        return filtered
    keys: List[str] = []
    for key, item in group.items():
        if key == timestamp_key or "timestamp" in key.lower():
            continue
        if isinstance(item, h5py.Dataset):
            keys.append(key)
    if not keys:
        raise KeyError(f"Group {group.name} has no datasets besides timestamp '{timestamp_key}'")
    return keys


def find_nearest_idx(array: np.ndarray, value: float) -> int:
    idx = int(np.searchsorted(array, value, side="left"))
    if idx == 0:
        return 0
    if idx >= len(array):
        return len(array) - 1
    prev_diff = abs(value - array[idx - 1])
    next_diff = abs(array[idx] - value)
    return idx - 1 if prev_diff <= next_diff else idx


def detect_timestamp_jump(timestamps: np.ndarray, threshold: float = 1.0) -> int:
    if len(timestamps) < 2:
        return 0
    diffs = np.diff(timestamps)
    jump_indices = np.where(diffs > threshold)[0]
    if jump_indices.size > 0:
        return int(jump_indices[-1] + 1)
    return 0


def resize_image_array(img_arr: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    # img_arr expected shape: (H, W) or (H, W, C)
    mode = None
    if img_arr.ndim == 2:
        mode = "L"
    elif img_arr.ndim == 3 and img_arr.shape[2] == 3:
        mode = "RGB"
    elif img_arr.ndim == 3 and img_arr.shape[2] == 4:
        mode = "RGBA"
    else:
        # Fallback: treat as generic array and resize channel-wise
        pil = Image.fromarray(img_arr.astype(np.uint8))
        pil = pil.resize((new_w, new_h), Image.BILINEAR)
        return np.asarray(pil)

    pil = Image.fromarray(img_arr.astype(np.uint8), mode=mode)
    pil = pil.resize((new_w, new_h), Image.BILINEAR)
    return np.asarray(pil)


def sync_demo(
    demo: str,
    image_obs: h5py.Group,
    lowdim_obs: h5py.Group,
    image_ts_key: str,
    lowdim_ts_key: str,
    image_keys: List[str],
    lowdim_keys: List[str],
    resize_h: int | None,
    resize_w: int | None,
    skip_n: int,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], np.ndarray] | Tuple[None, np.ndarray]:
    if image_ts_key not in image_obs:
        raise KeyError(f"Image timestamps '{image_ts_key}' missing in {image_obs.name}")
    if lowdim_ts_key not in lowdim_obs:
        raise KeyError(f"Low-dim timestamps '{lowdim_ts_key}' missing in {lowdim_obs.name}")

    master_timestamps = np.asarray(image_obs[image_ts_key][:], dtype=np.float64)
    follower_timestamps = np.asarray(lowdim_obs[lowdim_ts_key][:], dtype=np.float64)

    if master_timestamps.size == 0:
        raise ValueError(f"Demo {demo} has no image timestamps to drive synchronization")
    if follower_timestamps.size == 0:
        raise ValueError(f"Demo {demo} has no low-dimensional timestamps")

    master_cache = {key: image_obs[key][:] for key in image_keys}
    follower_cache = {key: lowdim_obs[key][:] for key in lowdim_keys}

    non_zero_mask = follower_timestamps > 1e-6
    if not np.all(non_zero_mask):
        print(f"Warning: Discarding {np.sum(~non_zero_mask)} zero-valued timestamps from low-dim data for demo {demo}")
        follower_timestamps = follower_timestamps[non_zero_mask]
        for k in follower_cache:
            follower_cache[k] = follower_cache[k][non_zero_mask]
        if follower_timestamps.size == 0:
            raise ValueError(f"Demo {demo} has only zero-valued low-dimensional timestamps")

    jump_idx = detect_timestamp_jump(follower_timestamps, threshold=0.5)
    if jump_idx > 0:
        print(f"Warning: Sudden jump detected in low-dim timestamps for demo {demo} at index {jump_idx}. Discarding {jump_idx} samples before the jump.")
        follower_timestamps = follower_timestamps[jump_idx:]
        for k in follower_cache:
            follower_cache[k] = follower_cache[k][jump_idx:]
        if follower_timestamps.size == 0:
            print(f"Warning: Discarding all low-dim timestamps due to jump for demo {demo}; skipping demo")
            return None, follower_timestamps

    low_start, low_end = np.min(follower_timestamps), np.max(follower_timestamps)
    img_start, img_end = np.min(master_timestamps), np.max(master_timestamps)
    overlap_start = max(img_start, low_start)
    overlap_end = min(img_end, low_end)

    print(f"Demo {demo} timestamp overlap: [{overlap_start:.3f}, {overlap_end:.3f}]")

    if overlap_start > overlap_end:
        print(f"Warning: No timestamp overlap between image and low-dim for demo {demo}; skipping demo")
        return None, follower_timestamps

    candidates_mask = (master_timestamps >= overlap_start) & (master_timestamps <= overlap_end)
    candidate_indices = np.where(candidates_mask)[0]
    if candidate_indices.size == 0:
        print(f"Warning: No image timestamps fall within the overlap interval for demo {demo}; skipping demo")
        return None, follower_timestamps

    start_idx = candidate_indices[0]
    end_idx = candidate_indices[-1]
    master_indices = np.arange(start_idx, end_idx + 1)
    master_cache_sliced = {k: v[master_indices] for k, v in master_cache.items()}

    synced_images: Dict[str, List[np.ndarray]] = {key: [] for key in image_keys}
    synced_lowdim: Dict[str, List[np.ndarray]] = {key: [] for key in lowdim_keys}

    master_in_ts = master_timestamps[master_indices]
    skip_counter = 0
    for local_idx, timestamp in enumerate(master_in_ts):
        if skip_counter > 0:
            skip_counter -= 1
            continue
        timestamp = float(timestamp)
        follower_idx = find_nearest_idx(follower_timestamps, timestamp)
        time_diff = abs(follower_timestamps[follower_idx] - timestamp)
        if time_diff > 0.1:
            raise ValueError(
                f"Timestamp mismatch at master idx {master_indices[local_idx]} (master ts: {timestamp}, nearest follower ts: {follower_timestamps[follower_idx]}, diff: {time_diff})"
            )

        for key in image_keys:
            img_sample = master_cache_sliced[key][local_idx]
            if resize_h is not None and resize_w is not None:
                img_sample = resize_image_array(img_sample, resize_h, resize_w)
            synced_images[key].append(img_sample)

        for key in lowdim_keys:
            follower_sample = follower_cache[key][follower_idx]
            synced_lowdim[key].append(follower_sample)

        # after saving this frame, skip next `skip_n` frames
        if skip_n > 0:
            skip_counter = skip_n

    image_arrays = {key: np.stack(values, axis=0) for key, values in synced_images.items()}
    lowdim_arrays = {key: np.stack(values, axis=0) for key, values in synced_lowdim.items()}

    # Ensure timestamps align with number of produced image frames.
    if image_arrays:
        first_key = next(iter(image_arrays))
        n_frames = image_arrays[first_key].shape[0]
        timestamps_out = master_in_ts[:n_frames]
    else:
        timestamps_out = master_in_ts

    return (
        {
            "timestamps": timestamps_out,
            "image_obs": image_arrays,
            "lowdim_obs": lowdim_arrays,
        },
        follower_timestamps,
    )


def write_demo(demo: str, out_root: h5py.Group, synced: Dict[str, Dict[str, np.ndarray]], image_ts_key: str, actions: np.ndarray | None) -> None:
    g_demo = out_root.create_group(demo)
    g_obs = g_demo.create_group("obs")

    if actions is not None:
        g_demo.create_dataset("actions", data=actions)
    g_obs.create_dataset(image_ts_key, data=synced["timestamps"])
    for key, arr in synced["image_obs"].items():
        g_obs.create_dataset(key, data=arr)
    for key, arr in synced["lowdim_obs"].items():
        g_obs.create_dataset(key, data=arr)

    g_demo.attrs["num_samples"] = synced["timestamps"].shape[0]


def generate_actions(
    lowdim_obs: Dict[str, np.ndarray],
    pos_key: str = "robot0_eef_pos",
    quat_key: str = "robot0_eef_quat",
    gripper_key: str = "robot0_gripper_qpos",
) -> "np.ndarray | None":
    """
    Compute a 7-D action vector for every synced timestep from robot EEF state.

    Action layout: [d_pos (3) | d_rotvec (3) | gripper (1)]
      - d_pos:    pos[t+1] - pos[t]
      - d_rotvec: rotation vector of R_{t+1} * R_t^{-1}
      - gripper:  -1.0 (open) or +1.0 (close) from gripper_qpos[t+1]

    The last timestep is padded by repeating the T-2 action so the returned
    array has the same length T as the input observations.

    Returns None when any required key is missing or the demo is too short.
    """
    for k in (pos_key, quat_key, gripper_key):
        if k not in lowdim_obs:
            print(f"  Missing key '{k}' for action computation; skipping action generation")
            return None

    pos = lowdim_obs[pos_key]       # (T, 3)
    quat = lowdim_obs[quat_key]     # (T, 4)  scipy convention: (x, y, z, w)
    gripper = lowdim_obs[gripper_key]  # (T,) or (T, k)
    if gripper.ndim > 1:
        gripper = gripper[:, 0]

    T = pos.shape[0]
    if T < 2:
        print("  Demo has fewer than 2 frames; cannot compute relative actions")
        return None

    # Gripper open/close threshold: midpoint of observed range.
    # High value → open (−1), low value → close (+1), matching synced_data_to_robotmimic.py.
    g_mid = (gripper.min() + gripper.max()) / 2.0

    actions: List[np.ndarray] = []
    for i in range(T - 1):
        d_pos = pos[i + 1] - pos[i]

        r_curr = Rotation.from_quat(quat[i])
        r_next = Rotation.from_quat(quat[i + 1])
        d_rotvec = (r_next * r_curr.inv()).as_rotvec()

        g_action = -1.0 if float(gripper[i + 1]) > g_mid else 1.0

        if np.any(np.abs(d_pos) > 1.0) or np.any(np.abs(d_rotvec) > 1.0):
            print(
                f"\033[93mWarning: large action values at t={i}: "
                f"d_pos={d_pos}, d_rotvec={d_rotvec}\033[0m"
            )

        actions.append(np.concatenate([d_pos, d_rotvec, [g_action]]))

    actions_arr = np.array(actions, dtype=np.float64)  # (T-1, 7)
    # Repeat the last computed action for the final observation so that the
    # length matches T and every observation has a corresponding action entry.
    actions_arr = np.concatenate([actions_arr, actions_arr[-1:]], axis=0)  # (T, 7)
    return actions_arr


def resample_sequence(sequence: np.ndarray, follower_ts: np.ndarray, master_ts: np.ndarray) -> np.ndarray:
    if sequence.shape[0] != follower_ts.shape[0]:
        raise ValueError(
            "Sequence length does not match low-dimensional timestamp count for resampling"
        )
    indices = [find_nearest_idx(follower_ts, t) for t in master_ts]
    return sequence[indices]


def main() -> None:
    args = parse_args()
    validate_files(args.image_h5, args.lowdim_h5)
    if os.path.abspath(args.image_h5) == os.path.abspath(args.output_h5):
        raise ValueError("Output file must differ from the image input file")
    if os.path.abspath(args.lowdim_h5) == os.path.abspath(args.output_h5):
        raise ValueError("Output file must differ from the low-dimensional input file")
    if args.skip_n < 0:
        raise ValueError("--skip_n must be non-negative")

    with h5py.File(args.image_h5, "r") as f_image, h5py.File(args.lowdim_h5, "r") as f_lowdim:
        if "data" not in f_image or "data" not in f_lowdim:
            raise KeyError("Both HDF5 files must contain a top-level 'data' group")
        demos = sorted(set(f_image["data"].keys()) & set(f_lowdim["data"].keys()))
        if not demos:
            raise ValueError("No overlapping demos found between the provided files")

        os.makedirs(os.path.dirname(os.path.abspath(args.output_h5)) or ".", exist_ok=True)
        with h5py.File(args.output_h5, "w") as f_out:
            g_out = f_out.create_group("data")
            processed = 0
            for demo in demos:
                print(f"Processing demo {demo}...")
                try:
                    image_obs = f_image["data"][demo]["obs"]
                    lowdim_demo = f_lowdim["data"][demo]
                    lowdim_obs = lowdim_demo["obs"]

                    image_keys = resolve_dataset_keys(image_obs, args.image_timestamp_key, args.image_keys)
                    lowdim_keys = resolve_dataset_keys(lowdim_obs, args.lowdim_timestamp_key, args.lowdim_keys)
                    result = sync_demo(
                        demo,
                        image_obs,
                        lowdim_obs,
                        args.image_timestamp_key,
                        args.lowdim_timestamp_key,
                        image_keys,
                        lowdim_keys,
                        args.resize_h,
                        args.resize_w,
                        args.skip_n,
                    )
                    if result[0] is None:
                        continue
                    synced, follower_ts = result
                except Exception as exc:
                    if args.allow_missing:
                        print(f"Skipping {demo}: {exc}")
                        continue
                    raise

                # Compute 7-D actions from synced EEF state (d_pos, d_rotvec, gripper).
                actions_data = generate_actions(
                    synced["lowdim_obs"],
                    pos_key=args.pos_key,
                    quat_key=args.quat_key,
                    gripper_key=args.gripper_key,
                )
                # Fall back to copying a pre-computed actions dataset when EEF keys are missing.
                if actions_data is None and "actions" in lowdim_demo:
                    try:
                        actions_source = lowdim_demo["actions"][:]
                        actions_data = resample_sequence(actions_source, follower_ts, synced["timestamps"])
                    except ValueError as exc:
                        print(f"Skipping actions for {demo}: {exc}")
                write_demo(demo, g_out, synced, args.image_timestamp_key, actions_data)
                processed += 1
                print(f"Synchronized {demo}: {synced['timestamps'].shape[0]} frames")

            f_out.attrs["source_image_h5"] = os.path.abspath(args.image_h5)
            f_out.attrs["source_lowdim_h5"] = os.path.abspath(args.lowdim_h5)
            f_out.attrs["num_synced_demos"] = processed
            print(f"Finished syncing {processed} demo(s) to {args.output_h5}")


if __name__ == "__main__":
    main()
