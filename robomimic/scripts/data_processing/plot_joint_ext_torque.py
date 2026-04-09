#!/usr/bin/env python3
"""
Plot smoothed joint external torques vs. time for a specified demo in an HDF5 dataset.

Usage:
    python plot_joint_ext_torque.py \
        --hdf5 /media/jiuzl/PortableSSD/egg_boiler/same_start_pose/low_dim_data.hdf5 \
        --demo demo_0

    # Save to file instead of displaying:
    python plot_joint_ext_torque.py \
        --hdf5 /media/.../low_dim_data.hdf5 \
        --demo demo_0 \
        --save torque_plot.png

    # Adjust smoothing window (default 51 samples):
    python plot_joint_ext_torque.py ... --window 101
"""

import argparse

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter


JOINT_LABELS = [f"Joint {i+1}" for i in range(7)]
COLORS = plt.cm.tab10.colors


def load_data(hdf5_path, demo_name):
    key_base = f"data/{demo_name}/obs"
    with h5py.File(hdf5_path, "r") as f:
        available = list(f["data"].keys())
        if demo_name not in available:
            raise KeyError(f"Demo '{demo_name}' not found. Available: {available}")
        torque = f[f"{key_base}/robot0_joint_ext_torque"][:]   # (T, 7)
        timestamp = f[f"{key_base}/timestamp"][:]               # (T,)
    return timestamp, torque


def smooth(signal, window):
    """Savitzky-Golay smoothing; falls back to raw if window > signal length."""
    if window >= len(signal):
        return signal
    # window must be odd
    w = window if window % 2 == 1 else window + 1
    return savgol_filter(signal, window_length=w, polyorder=3)


def plot(timestamp, torque, demo_name, window, save_path=None):
    t = timestamp - timestamp[0]   # start from t=0

    # 1920x1080 @ 100 dpi
    fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100)
    fig.suptitle(f"Smoothed Joint External Torques — {demo_name}", fontsize=20, fontweight="bold")

    for j in range(7):
        raw = torque[:, j]
        sm = smooth(raw, window)
        ax.plot(t, raw, color=COLORS[j], alpha=0.2, linewidth=0.8)
        ax.plot(t, sm,  color=COLORS[j], linewidth=1.8, label=JOINT_LABELS[j])

    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    ax.set_xlabel("Time (s)", fontsize=16, fontweight="bold")
    ax.set_ylabel("External Torque (Nm)", fontsize=16, fontweight="bold")
    ax.legend(fontsize=14, loc="upper right", prop={"weight": "bold"})
    ax.grid(True, linewidth=0.4, alpha=0.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100)
        print(f"Plot saved to '{save_path}'.")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot smoothed joint external torques from an HDF5 demo.")
    parser.add_argument("--hdf5", required=True, help="Path to HDF5 file.")
    parser.add_argument("--demo", required=True, help="Demo name, e.g. 'demo_0'.")
    parser.add_argument("--window", type=int, default=51,
                        help="Savitzky-Golay smoothing window size (odd integer, default: 51).")
    parser.add_argument("--save", default=None,
                        help="If given, save the figure to this path instead of displaying.")
    args = parser.parse_args()

    timestamp, torque = load_data(args.hdf5, args.demo)
    print(f"Loaded '{args.demo}': {len(timestamp)} timesteps, "
          f"{torque.shape[1]} joints, "
          f"duration {timestamp[-1] - timestamp[0]:.2f}s")

    plot(timestamp, torque, args.demo, args.window, args.save)


if __name__ == "__main__":
    main()
