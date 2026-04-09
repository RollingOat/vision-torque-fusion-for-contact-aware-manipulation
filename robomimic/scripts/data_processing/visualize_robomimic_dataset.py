import h5py
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio
import warnings
import os

warnings.filterwarnings("ignore", message=".*macro_block_size.*", module="imageio")


def _extract_joint_ext_torque(obs_group, key="robot0_joint_ext_torque"):
    if key not in obs_group:
        return None
    data = np.asarray(obs_group[key][()])
    if data.ndim == 3 and data.shape[-1] == 70:
        data = data[:, -1, :]
    elif data.ndim == 2 and data.shape[1] == 70:
        data = data
    else:
        data = data.reshape((data.shape[0], -1)) if data.ndim >= 2 else data[:, None]
    if data.shape[1] >= 7:
        return data[:, -7:]
    return None


def _extract_gripper_width(obs_group):
    for key in ("robot0_gripper_width", "robot0_gripper_qpos", "robot0_gripper_pos"):
        if key in obs_group:
            data = np.asarray(obs_group[key][()])
            if data.ndim == 1:
                return data
            if data.ndim >= 2:
                vals = data.reshape((data.shape[0], -1))
                if vals.shape[1] == 1:
                    return vals[:, 0]
                if vals.shape[1] >= 2:
                    return np.sum(vals[:, :2], axis=1)
                return vals[:, 0]
    return None


def visualize_dataset(dataset_path, output_dir=None, camera_names="agentview_image", fps=10):
    print(f"Opening dataset: {dataset_path}")
    f = h5py.File(dataset_path, "r")
    
    # Parse camera names if comma-separated
    if isinstance(camera_names, str):
        camera_names = [c.strip() for c in camera_names.split(",")]

    # Get list of demos (ensure string names, not bytes)
    raw_demos = list(f["data"].keys())
    demos = [d.decode() if isinstance(d, bytes) else d for d in raw_demos]
    # Sort demos by number (demo_0, demo_1, ...)
    try:
        inds = np.argsort([int(elem.replace("demo_", "")) for elem in demos])
        demos = [demos[i] for i in inds]
    except Exception:
        print("Warning: Could not sort demos numerically. Using default order.")

    print(f"Found {len(demos)} demonstrations.")

    # Iterate through demos
    for ep in demos:
        print(f"Playing {ep}...")
        
        # Load observations for this episode
        data_grp = f["data"]
        ep_grp = data_grp[ep]
        obs_group = ep_grp["obs"]
        
        # Check Cameras
        valid_cameras = []
        for cam in camera_names:
            if cam in obs_group:
                valid_cameras.append(cam)
            else:
                print(f"Camera '{cam}' not found in {ep}. Available keys: {list(obs_group.keys())}")
        
        if not valid_cameras:
            # Try to auto-detect an image key if none found from requests
            candidates = [k for k in obs_group.keys() if "image" in k]
            if candidates:
                # Just pick the first candidate as fallback
                print(f"-> No valid requested cameras found. Switching to fallback '{candidates[0]}'")
                valid_cameras = [candidates[0]]
            else:
                print("No image data found. Skipping visualization for this episode.")
                continue

        # Load and pre-process all camera images vectorized (CHW→HWC + normalize)
        camera_images = {}
        T = 0
        for cam in valid_cameras:
            imgs = obs_group[cam][()]
            if imgs.ndim == 4 and imgs.shape[1] == 3:
                imgs = np.transpose(imgs, (0, 2, 3, 1))
            if imgs.dtype != np.uint8:
                if np.issubdtype(imgs.dtype, np.floating):
                    imgs = np.clip(imgs, 0.0, 1.0) * 255.0
                imgs = np.clip(imgs, 0, 255).astype(np.uint8)
            camera_images[cam] = imgs
            T = max(T, len(imgs))
        
        actions = ep_grp["actions"][()] if "actions" in ep_grp else None
        torque_data = _extract_joint_ext_torque(obs_group)
        if torque_data is None:
            print("robot0_joint_ext_torque not found or malformed. Skipping episode.")
            continue

        pos_data = obs_group["robot0_eef_pos"][()] if "robot0_eef_pos" in obs_group else None
        gripper_width = _extract_gripper_width(obs_group)

        timesteps = np.arange(T)

        n_cams = len(valid_cameras)
        # Create subplots: 1 row, n_cams + 1 columns
        # n_cams for images, 1 for plot
        fig, axes = plt.subplots(1, n_cams + 1, figsize=(7 * (n_cams + 1), 6), dpi=100)
        
        # If n_cams + 1 == 1, axes is not a list, but we always have at least 1 cam + 1 plot = 2 axes
        # So axes is always indexable.
        
        img_axes = axes[:n_cams]
        ax_plot = axes[n_cams]

        for i, ax in enumerate(img_axes):
            ax.set_axis_off()
            ax.set_title(f"{valid_cameras[i]}")

        # Plot setup
        ax_plot.set_xlim(0, T)
        min_v = np.min(torque_data)
        max_v = np.max(torque_data)
        pad = (max_v - min_v) * 0.1 if max_v > min_v else 1.0
        ax_plot.set_ylim(min_v - pad, max_v + pad)
        ax_plot.set_xlabel("Time Step")
        ax_plot.set_ylabel("robot0_joint_ext_torque (last 7)")
        ax_plot.grid(True, linestyle="--", alpha=0.7)
        ax_plot.set_title(f"Episode: {ep} Torque")

        # Initialize images for each camera (already HWC uint8 after pre-processing)
        img_displays = []
        for i, cam in enumerate(valid_cameras):
            disp = img_axes[i].imshow(camera_images[cam][0])
            img_displays.append(disp)

        lines = []
        colors = ["r", "g", "b", "c", "m", "y", "k"]
        for i in range(7):
            line, = ax_plot.plot([], [], color=colors[i % len(colors)], linewidth=1.5, label=f"J{i+1}")
            lines.append(line)
        ax_plot.legend(loc="upper right", fontsize=8)

        # Text overlay on the FIRST camera view
        ax_text = img_axes[0].text(0.02, 0.95, "", transform=img_axes[0].transAxes, color="lime", fontsize=10, 
                              verticalalignment='top', bbox=dict(facecolor='black', alpha=0.5))

        def update_frame(t):
            for i, cam in enumerate(valid_cameras):
                imgs = camera_images[cam]
                img_displays[i].set_data(imgs[min(t, len(imgs) - 1)])

            for i, line in enumerate(lines):
                if i < torque_data.shape[1]:
                    line.set_data(timesteps[:t + 1], torque_data[:t + 1, i])

            title_parts = [f"Ep: {ep}"]
            if pos_data is not None and t < len(pos_data):
                pos = pos_data[t]
                if np.asarray(pos).size >= 3:
                    title_parts.append(f"Pos: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
            if gripper_width is not None and t < len(gripper_width):
                title_parts.append(f"Gripper: {float(gripper_width[t]):.3f}")
            img_axes[0].set_title(f"{valid_cameras[0]}\n" + " | ".join(title_parts), fontsize=9)

            if actions is not None and t < len(actions):
                rv = actions[t, 3:6]
                rv_norm = np.linalg.norm(rv)
                ax_text.set_text(f"RotVec: [{rv[0]:+.3f}, {rv[1]:+.3f}, {rv[2]:+.3f}]\nWait:  {rv_norm:.3f}")
            else:
                ax_text.set_text("")

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(output_dir, f"{ep}.mp4")
            with imageio.get_writer(out_path, fps=fps, codec="libx264", format="ffmpeg") as writer:
                for t in range(T):
                    update_frame(t)
                    fig.canvas.draw()
                    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                    w, h = fig.canvas.get_width_height()
                    frame = buf.reshape(h, w, 3)
                    writer.append_data(frame)
            plt.close(fig)
            print(f"Episode {ep} saved to {out_path}")
        else:
            for t in range(T):
                update_frame(t)
                plt.pause(1.0 / max(1, fps))
            plt.show()
            plt.close(fig)
            print(f"Episode {ep} finished.")

    print("Visualization finished.")
    f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Animate or save per-demo MP4 videos from a Robomimic dataset")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the HDF5 dataset")
    parser.add_argument("--output_dir", type=str, default="/home/jiuzl/robomimic_suite/temp", help="Directory to save per-demo MP4s (omit to just play)")
    parser.add_argument("--camera", type=str, default="agentview_image", help="Camera observation key(s), comma-separated (default: agentview_image)")
    parser.add_argument("--fps", type=int, default=60, help="FPS for animation (default: 30)")

    args = parser.parse_args()

    visualize_dataset(args.dataset, args.output_dir, args.camera, args.fps)
