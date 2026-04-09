import h5py
import numpy as np
import argparse
import os
import shutil
import cv2
import json
from scipy.spatial.transform import Rotation as R



'''
example usage: 
python synced_data_to_robotmimic.py --input /home/jiuzl/robomimic_suite/robot_camera_control/data/synced_image_low_dim.hdf5 \
    --out /home/jiuzl/robomimic_suite/robot_camera_control/data/franka_robomimic.hdf5 --skip_n 4
'''

#>>>>> Action Generation
def generate_actions(robot_obs, gripper_qpos):
    """
    Generate actions: relative pose (d_pos, d_rot) and binary gripper action.
    
    Args:
        robot_obs (dict): Dictionary containing 'robot0_eef_pos', 'robot0_eef_quat' arrays.
        gripper_qpos (np.array): Gripper joint positions.
    
    Returns:
        np.array: Actions array of shape (N-1, 7).
    """
    pos_data = robot_obs['robot0_eef_pos']
    quat_data = robot_obs['robot0_eef_quat']
    
    num_samples = len(pos_data)
    if num_samples < 2:
        return None

    actions = []
    
    # Determine gripper range to decide open/close threshold
    g_min = np.min(gripper_qpos)
    g_max = np.max(gripper_qpos)
    g_mid = (g_min + g_max) / 2.0
    
    # Heuristic: If max - min is small, gripper might be static.
    # We assume 'High' value means Open (like width), 'Low' means Closed.
    # Action: -1 for Open, 1 for Close.
    # If standard Robotiq (joint), 0=Open, High=Closed. In that case, we might need to invert.
    # For now, default to High=Open (Width). 
    # Logic: if next_val > mid -> Open (-1). Else -> Close (1).
    
    for i in range(num_samples - 1):
        # 1. Delta Pose
        curr_pos = pos_data[i]
        next_pos = pos_data[i+1]
        d_pos = next_pos - curr_pos
        
        # 2. Delta Rotation (Rotation Vector)
        curr_quat = quat_data[i]
        next_quat = quat_data[i+1]
        
        r_curr = R.from_quat(curr_quat)
        r_next = R.from_quat(next_quat)
        
        # Delta rotation: R_next * R_curr_inv
        # Convert to rotation vector
        r_diff = r_next * r_curr.inv()
        d_rot = r_diff.as_rotvec()
        
        # 3. Gripper Action (Binary State of Next Step)
        g_next_val = gripper_qpos[i+1]
        
        # Using 0.04 as a common threshold if range isn't clear, or midpoint.
        # Assuming High Values = Open (Width).
        # Open -> -1, Close -> 1
        if g_next_val > g_mid:
            g_action = -1.0
        else:
            g_action = 1.0
        
        # if any value of d_pos, d_rot, and g_action out of the range [-1, 0, 1.0], print a warning in red (but still include the action)
        if np.any(np.abs(d_pos) > 1.0) or np.any(np.abs(d_rot) > 1.0) or np.abs(g_action) > 1.0:
            print(f"\033[91mWarning: Action values out of range at index {i}: d_pos={d_pos}, d_rot={d_rot}, g_action={g_action}\033[0m")
            print(f"  Current Quat: {curr_quat}, Next Quat: {next_quat}")
            print(f"  Current rotvec: {r_curr.as_rotvec()}, Next rotvec: {r_next.as_rotvec()}")
            # print the euler angles of current and next rotation for debugging
            print(f"  Current Euler: {r_curr.as_euler('xyz', degrees=True)}, Next Euler: {r_next.as_euler('xyz', degrees=True)}")


        action = np.concatenate([d_pos, d_rot, [g_action]])
        actions.append(action)
        
    return np.array(actions)

def pad_with_history(data, history_len):
    """
    Pad the data with previous measurements.
    """
    n_samples = data.shape[0]
    
    # Handle 1D array
    if data.ndim == 1:
        data = data.reshape(-1, 1)
        
    dim = data.shape[1]
    
    padded_data = np.zeros((n_samples, history_len, dim), dtype=data.dtype)
    
    for i in range(n_samples):
        # Determine start index (inclusive) and end index (inclusive) for slicing
        # We want [i-H+1, ..., i]
        start_idx = i - history_len + 1
        
        if start_idx < 0:
            # Need padding
            valid_len = i + 1
            pad_len = history_len - valid_len
            
            # Fill valid part at the end
            padded_data[i, pad_len:, :] = data[0:valid_len]
            
            # Fill padding part with the first available value (data[0])
            # np.tile to repeat data[0] pad_len times
            padded_data[i, :pad_len, :] = np.tile(data[0], (pad_len, 1))
        else:
            # Full history available
            padded_data[i, :, :] = data[start_idx : i+1]
    # Flatten history and dim
    return padded_data.reshape(n_samples, -1)

#>>>>>>>>>>>>>>>>>>> Demo Processing Functions <<<<<<<<<<<<<<<<<<
def process_demo(demo_name, demo_grp, out_data_grp, args, new_demo_name):
    print(f"Processing {demo_name} -> {new_demo_name}")
    
    obs_grp = demo_grp["obs"]
    
    # 1. Identify valid keys (excluding timestamp)
    keys = []
    for k in obs_grp.keys():
        if "timestamp" not in k:
            keys.append(k)
            
    # Load data for processing
    data_cache = {}
    num_frames = 0
    
    # Determine output num_samples (T-1)
    # Check length of a reference key (e.g., 'robot0_eef_pos')
    ref_key = 'robot0_eef_pos'
    if ref_key not in obs_grp:
         # Fallback to any key
         ref_key = keys[0]
         
    num_frames = obs_grp[ref_key].shape[0]
    
    if args.exclude_demos and demo_name in args.exclude_demos:
        # print in green to indicate skipping
        print(f"\033[92m  Skipping {demo_name} (excluded).\033[0m")
        return 0

    # Subsampling
    # Keep indices: 0, 1+skip, 1+2*(1+skip)...
    # Wait, "skip n frames". means keep 1, skip n. Step = n+1.
    step = args.skip_n + 1
    indices = np.arange(0, num_frames, step)
    
    if len(indices) < 2:
        print(f"\033[91m  Skipping {demo_name}: Too few frames after subsampling.\033[0m")
        return 0

    # Load and subsample all data
    for k in keys:
        d = obs_grp[k][:]
        if args.skip_n > 0:
            d = d[indices]
        
        # Image Resizing
        if "image" in k and args.resize_w is not None and args.resize_h is not None:
            # Check if likely image (3 dimensions or 4? N,H,W,C)
            if d.ndim == 4:
                new_imgs = []
                for img in d:
                    new_img = cv2.resize(img, (args.resize_w, args.resize_h), interpolation=cv2.INTER_AREA)
                    new_imgs.append(new_img)
                d = np.array(new_imgs)
        
        # If the key looks like a depth image, scale values (e.g., from millimeters to meters)
        if "depth" in k.lower():
            # Convert to float32 before scaling to avoid integer truncation
            d = d.astype(np.float32) * args.depth_scale
            # resize depth to (args.resize_w, args.resize_h) if specified
            if args.resize_w is not None and args.resize_h is not None:
                new_depths = []
                for depth in d:
                    new_depth = cv2.resize(depth, (args.resize_w, args.resize_h), interpolation=cv2.INTER_NEAREST)
                    new_depths.append(new_depth)
                d = np.array(new_depths)

        data_cache[k] = d

    # Recalculate num_frames after subsampling
    num_frames_sub = len(data_cache[ref_key])
    
    # 2. Generate Actions
    # Requires robot pose and gripper info
    required_action_keys = ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']
    can_compute_action = all(k in data_cache for k in required_action_keys)
    
    actions = None
    if can_compute_action:
        robot_obs = {
            'robot0_eef_pos': data_cache['robot0_eef_pos'],
            'robot0_eef_quat': data_cache['robot0_eef_quat']
        }
        actions = generate_actions(robot_obs, data_cache['robot0_gripper_qpos'])
    else:
        print(f"\033[91m  Warning: Missing keys for action generation in {demo_name}. Included keys: {list(data_cache.keys())}\033[0m")
        return 0

    if actions is None:
        return 0
    
    # Apply History Padding
    # Apply to keys containing "force", "torque" or in args.pad_keys
    for k in list(data_cache.keys()):
        should_pad = False
        if "force" in k or "torque" in k:
            should_pad = True
        
        if args.pad_keys and k in args.pad_keys:
            should_pad = True
            
        if should_pad:
            d = data_cache[k]
            # Only pad low-dim data (rank <= 2)
            if isinstance(d, np.ndarray) and d.ndim <= 2:
                # print(f"  Padding key '{k}' with history_len={args.history_len}")
                data_cache[k] = pad_with_history(d, args.history_len)

    # Check lengths
    # Actions len: T-1
    # Obs len: T-1 (stop at second last)
    # Next obs len: T-1 (start at second)
    
    # Create Output Groups
    g_demo = out_data_grp.create_group(new_demo_name)
    g_demo.attrs["num_samples"] = len(actions)
    
    g_demo.create_dataset("actions", data=actions)
    
    g_obs = g_demo.create_group("obs")
    g_next_obs = g_demo.create_group("next_obs")
    
    # Slice obs (0 to T-1) and next_obs (1 to T)
    for k, v in data_cache.items():
        # Obs: exclude last frame
        obs_data = v[:-1]
        # Next Obs: exclude first frame
        next_obs_data = v[1:]
        
        g_obs.create_dataset(k, data=obs_data)
        g_next_obs.create_dataset(k, data=next_obs_data)
        
        # If image, allow compression? Robomimic usually likes uncompressed or gzip.
        # h5py default is uncompressed.

    return len(actions)

#>>>>>>>>>>>>>>>>>>>>>>>>>>> Main Execution<<<<<<<<<<<<<<<<<<<<<<<<<<<

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input synced HDF5 file (from sync_image_low_dim.py)")
    parser.add_argument("--out", type=str, required=True, help="Output robomimic-compatible HDF5 file")
    
    parser.add_argument("--resize_w", type=int, default=224)
    parser.add_argument("--resize_h", type=int, default=224)
    parser.add_argument("--skip_n", type=int, default=0)
    parser.add_argument("--exclude_demos", nargs='*', default=[])
    parser.add_argument("--pad_keys", nargs='*', default=[], help="List of keys to always apply history padding")
    parser.add_argument("--history_len", type=int, default=10, help="History length for padding (default: 10)")
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=0.001,
        help="Scale factor applied to datasets whose key contains 'depth' (default: 0.001)",
    )
    parser.add_argument(
        "--train-percent",
        type=float,
        default=0.9,
        help="Fraction of demos to use for training (default: 0.9)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for shuffling demo assignment (default: None)",
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist.")
        return

    f_in = h5py.File(args.input, 'r')
    f_out = h5py.File(args.out, 'w')
    
    if "data" not in f_in:
        print("Error: Input file missing 'data' group.")
        f_in.close()
        f_out.close()
        return

    g_data_out = f_out.create_group("data")
    
    # Sort demos
    demos = sorted(list(f_in["data"].keys()), key=lambda x: int(x.split('_')[1]) if '_' in x else x)
    
    total_samples = 0
    valid_demo_count = 0
    output_demo_names = []
    
    for demo_name in demos:
        new_demo_name = f"demo_{valid_demo_count}"
        
        n_samples = process_demo(demo_name, f_in["data"][demo_name], g_data_out, args, new_demo_name)
        
        if n_samples > 0:
            total_samples += n_samples
            valid_demo_count += 1
            output_demo_names.append(new_demo_name)
            
    # Global Attributes
    g_data_out.attrs["total"] = total_samples
    
    # Env Args (Try to copy from input or set default)
    # Robomimic expects env_args as json string in data attrs
    env_args = {
        "env_name": "Unknown",
        "env_kwargs": {}
    }
                 
    g_data_out.attrs["env_args"] = json.dumps(env_args)
    
    # Create mask group listing demo names for train/valid split
    print("Creating train/validation split...")
    # Create top-level mask group (same level as 'data') for compatibility
    
    mask_grp = f_out.create_group("mask")
    # Clamp train-percent to [0,1]
    p = float(args.train_percent)
    p = max(0.0, min(1.0, p))

    N = len(output_demo_names)
    n_train = int(np.floor(p * N))

    # Randomly assign demos to train/valid. Use seed if provided for reproducibility.
    if N > 0:
        if args.seed is not None:
            rng = np.random.RandomState(args.seed)
            perm = rng.permutation(N)
        else:
            perm = np.random.permutation(N)

        train_idx = perm[:n_train]
        valid_idx = perm[n_train:]

        train_names = [output_demo_names[i] for i in train_idx]
        valid_names = [output_demo_names[i] for i in valid_idx]
    else:
        train_names = []
        valid_names = []
    print(f"Assigned {len(train_names)} demos to train and {len(valid_names)} demos to validation.")
    print(f"Train demos: {train_names}")
    print(f"Validation demos: {valid_names}")
    dt = h5py.string_dtype(encoding='utf-8')
    # Write train/valid lists into the top-level mask group
    mask_grp.create_dataset("train", data=np.array(train_names, dtype=dt), dtype=dt)
    mask_grp.create_dataset("valid", data=np.array(valid_names, dtype=dt), dtype=dt)
    

    f_in.close()
    f_out.close()
    print(f"Processing complete. {valid_demo_count} demos, {total_samples} total samples.")

if __name__ == "__main__":
    main()
