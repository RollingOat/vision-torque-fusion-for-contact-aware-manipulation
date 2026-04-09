import h5py
import cv2
import numpy as np
import argparse
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

'''
example usage:
python visualize_synced_data.py /home/jiuzl/robomimic_suite/robot_camera_control/data/franka_robomimic.hdf5 \
--out_dir /home/jiuzl/robomimic_suite/robot_camera_control/video --fps 10
'''
def create_plot_image(data, title, width, height, color=None):
    """
    Creates a plot of shape (N, 3) data (x, y, z) and returns it as an RGB image.
    """
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    
    # Plot x, y, z
    labels = ['x', 'y', 'z']
    colors = ['r', 'g', 'b']
    
    steps = np.arange(len(data))
    
    for i in range(3):
        ax.plot(steps, data[:, i], label=labels[i], color=colors[i], linewidth=1)
        
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.set_xlim(0, len(data))
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Remove margins
    plt.tight_layout()
    
    # Draw to canvas
    fig.canvas.draw()
    
    # Convert to image
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    
    # Convert RGB to BGR for OpenCV
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    return img

def visualize_demo(
    demo_grp,
    demo_name,
    output_folder,
    image_key,
    fps=30,
    force_hist_len=10,
    force_key="robot0_force_ee",
    torque_key="robot0_torque_ee",
):
    print(f"Visualizing {demo_name} ({image_key})...")

    obs_grp = demo_grp["obs"]

    if image_key not in obs_grp:
        print(f"  Skipping {demo_name}: '{image_key}' not found in obs group.")
        return

    images = obs_grp[image_key][:]
    n_frames = len(images)
    if n_frames == 0:
        print(f"  Skipping {demo_name}: No frames for key '{image_key}'.")
        return

    video_path = os.path.join(output_folder, f"{demo_name}.mp4")

    # Force visualization target size to 480x680 regardless of input image shape
    target_h = 480
    target_w = 680
    total_w = target_w

    # --- Force/Torque Plot Preparation ---
    def locate_signal(keywords):
        for key in obs_grp:
            lowered = key.lower()
            if any(keyword in lowered for keyword in keywords):
                return key, obs_grp[key][:]
        return None, None

    def fetch_signal(preferred_key):
        if preferred_key and preferred_key in obs_grp:
            return preferred_key, obs_grp[preferred_key][:]
        else:
            return locate_signal(["force", "torque"])


    def extract_latest_vector(data):
        arr = np.asarray(data)
        if arr.ndim == 3:
            return arr[:, -1, :]
        if arr.ndim == 2:
            cols = arr.shape[1]
            if cols % force_hist_len == 0 and cols // force_hist_len >= 3:
                feature_dim = cols // force_hist_len
                arr = arr.reshape(-1, force_hist_len, feature_dim)
                return arr[:, -1, :]
            return arr
        if arr.ndim == 1:
            return arr[:, None]
        return arr

    force_key, force_raw = fetch_signal(force_key)
    torque_key, torque_raw = fetch_signal(torque_key)

    force_data = extract_latest_vector(force_raw) if force_raw is not None else None
    torque_data = extract_latest_vector(torque_raw) if torque_raw is not None else None

    plot_h = 200
    ft_img_base = None
    trq_img_base = None

    if force_data is not None:
        ft_img_base = create_plot_image(force_data, f"Force ({force_key})", total_w, plot_h)
        ft_img_base = cv2.resize(ft_img_base, (total_w, plot_h))

    if torque_data is not None:
        trq_img_base = create_plot_image(torque_data, f"Torque ({torque_key})", total_w, plot_h)
        trq_img_base = cv2.resize(trq_img_base, (total_w, plot_h))

    total_h = target_h
    if force_data is not None:
        total_h += plot_h
    if torque_data is not None:
        total_h += plot_h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video = cv2.VideoWriter(video_path, fourcc, fps, (total_w, total_h))

    pos_data = obs_grp["robot0_eef_pos"][:] if "robot0_eef_pos" in obs_grp else None

    for i in range(n_frames):
        frame = images[i]
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.resize(frame, (target_w, target_h))
        cv2.putText(frame, image_key, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Frame: {i}/{n_frames}", (10, target_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if pos_data is not None and i < len(pos_data):
            pos = pos_data[i]
            text = f"Pos: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]"
            cv2.putText(frame, text, (10, target_h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Overlay numeric force/torque values on the image
        if force_data is not None and i < len(force_data):
            fvec = force_data[i]
            # Support vectors of length >=3 (show first 3) or scalar
            if getattr(fvec, 'ndim', None) is None:
                fstr = f"F: {float(fvec):.2f}"
            else:
                # Ensure 1D
                f1d = np.asarray(fvec).reshape(-1)
                if f1d.size >= 3:
                    fstr = f"F: [{f1d[0]:.2f}, {f1d[1]:.2f}, {f1d[2]:.2f}]"
                else:
                    fstr = "F: [" + ", ".join(f"{v:.2f}" for v in f1d) + "]"
            # Put top-right corner
            org = (max(10, target_w - 10 - len(fstr) * 10), 30)
            cv2.putText(frame, fstr, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        if torque_data is not None and i < len(torque_data):
            tvec = torque_data[i]
            t1d = np.asarray(tvec).reshape(-1)
            if t1d.size >= 3:
                tstr = f"T: [{t1d[0]:.2f}, {t1d[1]:.2f}, {t1d[2]:.2f}]"
            else:
                tstr = "T: [" + ", ".join(f"{v:.2f}" for v in t1d) + "]"
            org2 = (max(10, target_w - 10 - len(tstr) * 10), 60)
            cv2.putText(frame, tstr, org2, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2)

        final_frame = frame

        if ft_img_base is not None:
            f_img = ft_img_base.copy()
            margin_l = 0.12 * total_w
            margin_r = 0.95 * total_w
            plot_w_pixels = margin_r - margin_l
            x_pos = int(margin_l + (i / max(1, n_frames - 1)) * plot_w_pixels)
            cv2.line(f_img, (x_pos, 0), (x_pos, plot_h), (0, 0, 0), 2)
            final_frame = np.vstack([final_frame, f_img])

        if trq_img_base is not None:
            t_img = trq_img_base.copy()
            margin_l = 0.12 * total_w
            margin_r = 0.95 * total_w
            plot_w_pixels = margin_r - margin_l
            x_pos = int(margin_l + (i / max(1, n_frames - 1)) * plot_w_pixels)
            cv2.line(t_img, (x_pos, 0), (x_pos, plot_h), (0, 0, 0), 2)
            final_frame = np.vstack([final_frame, t_img])

        if final_frame.shape[0] != total_h or final_frame.shape[1] != total_w:
            final_frame = cv2.resize(final_frame, (total_w, total_h))
        out_video.write(final_frame)

    out_video.release()
    print(f"  Saved video to {video_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize synced HDF5 dataset and save videos.")
    parser.add_argument("file", nargs="?", default="synced_data.hdf5", help="Path to synced HDF5 file")
    parser.add_argument("--out_dir", default="/home/jiuzl/robomimic_suite/temp", help="Output directory for videos")
    parser.add_argument("--fps", type=int, default=30, help="FPS for output video")
    parser.add_argument("--image-key", default="agentview_image", help="Observation image key to visualize")
    parser.add_argument(
        "--force-hist-len",
        type=int,
        default=10,
        help="History length used when reshaping flattened force/torque datasets",
    )
    parser.add_argument(
        "--force-key",
        default="robot0_force_ee",
        help="Preferred force dataset key (fallbacks to any 'force' key if missing)",
    )
    parser.add_argument(
        "--torque-key",
        default="robot0_torque_ee",
        help="Preferred torque dataset key (fallbacks to any 'torque' key if missing)",
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        print(f"File not found: {args.file}")
        return
        
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
        print(f"Created output directory: {args.out_dir}")
        
    f = h5py.File(args.file, 'r')
    
    if "data" not in f:
        print("Invalid file structure: no 'data' group.")
        return
        
    def get_sort_key(s):
        # Improved sorting logic to handle variable key formats.
        # It tries to find a trailing number (e.g. 'demo_10' -> 10)
        # If not, it falls back to string sorting.
        # Priority 0 is for successfully parsed numbers, 1 for others.
        parts = s.split('_')
        # Check if any part is a digit, preferably the last one or the designated ID slot
        for part in reversed(parts):
            if part.isdigit():
                return (0, int(part))
        return (1, s)

    demos = sorted(list(f["data"].keys()), key=get_sort_key)
    
    print(f"Found {len(demos)} demos.")
    
    for demo_name in demos:
        visualize_demo(
            f["data"][demo_name],
            demo_name,
            args.out_dir,
            args.image_key,
            args.fps,
            args.force_hist_len,
            args.force_key,
            args.torque_key,
        )
        
    f.close()
    print("Done.")

if __name__ == "__main__":
    main()
