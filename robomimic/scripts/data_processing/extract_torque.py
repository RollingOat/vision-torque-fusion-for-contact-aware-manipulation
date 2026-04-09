#!/usr/bin/env python3
"""
Extract robot0_joint_ext_torque from an HDF5 dataset and save as a .npy file.

Output file (saved next to the HDF5 or to --out_dir):
    <demo>_torque.npy     shape (T, 7)   – raw external joint torques (Nm)

Usage:
    # Single demo
    python extract_torque.py --hdf5 /path/to/low_dim_data.hdf5 --demo demo_0

    # Multiple demos
    python extract_torque.py --hdf5 /path/to/low_dim_data.hdf5 --demo demo_0 demo_1 demo_5

    # All demos in the file
    python extract_torque.py --hdf5 /path/to/low_dim_data.hdf5 --all

    # Write to a custom directory
    python extract_torque.py --hdf5 /path/to/low_dim_data.hdf5 --all --out_dir ./npy_data
"""

import argparse
import os

import h5py
import numpy as np


def extract_demo(hdf5_path: str, demo_name: str, out_dir: str) -> None:
    key_base = f"data/{demo_name}/obs"
    with h5py.File(hdf5_path, "r") as f:
        available = list(f["data"].keys())
        if demo_name not in available:
            raise KeyError(
                f"Demo '{demo_name}' not found in file.\n"
                f"Available demos: {available}"
            )
        torque = f[f"{key_base}/robot0_joint_ext_torque"][:]  # (T, 7)

    torque_path = os.path.join(out_dir, f"{demo_name}_torque.npy")
    np.save(torque_path, torque)

    print(f"[{demo_name}] T={torque.shape[0]}, joints={torque.shape[1]}  →  {torque_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract joint external torques from an HDF5 dataset and save as .npy files."
    )
    parser.add_argument("--hdf5", required=True, help="Path to the HDF5 file.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--demo", nargs="+", metavar="DEMO",
        help="One or more demo names to extract (e.g. demo_0 demo_1)."
    )
    group.add_argument(
        "--all", action="store_true",
        help="Extract every demo found in data/."
    )
    parser.add_argument(
        "--out_dir", default=None,
        help="Output directory for .npy files. Defaults to the directory of the HDF5 file."
    )
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.hdf5))
    os.makedirs(out_dir, exist_ok=True)

    if args.all:
        with h5py.File(args.hdf5, "r") as f:
            demos = sorted(f["data"].keys())
        print(f"Found {len(demos)} demos in '{args.hdf5}'.")
    else:
        demos = args.demo

    for demo in demos:
        extract_demo(args.hdf5, demo, out_dir)

    print(f"\nDone. Files saved to: {out_dir}")


if __name__ == "__main__":
    main()
