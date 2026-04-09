#!/usr/bin/env python3
"""
Extract oak_image frames from a specified demo in an HDF5 dataset.
Every 10th frame is saved (1 frame kept, 9 skipped).

Usage:
    python extract_oak_images.py \
        --hdf5 /path/to/dataset.hdf5 \
        --demo demo_0 \
        --output_dir ./output_frames
"""

import argparse
import os

import cv2
import h5py
import numpy as np


def extract_oak_images(hdf5_path, demo_name, output_dir, skip=9):
    """
    Extract oak_image frames from a demo, keeping 1 out of every (skip+1) frames.

    Args:
        hdf5_path:  Path to the HDF5 file.
        demo_name:  Demo key, e.g. "demo_0".
        output_dir: Directory where images will be saved.
        skip:       Number of frames to skip between saved frames (default: 9).
    """
    os.makedirs(output_dir, exist_ok=True)

    with h5py.File(hdf5_path, "r") as f:
        key = f"data/{demo_name}/obs/oak_image"
        if key not in f:
            raise KeyError(f"Key '{key}' not found in {hdf5_path}. "
                           f"Available demos: {list(f['data'].keys())}")

        images = f[key]  # shape: (T, H, W, 3), dtype uint8
        total_frames = images.shape[0]
        frame_indices = range(0, total_frames, skip + 1)

        print(f"Demo '{demo_name}': {total_frames} total frames, "
              f"saving {len(frame_indices)} frames (every {skip+1} frames).")

        for save_idx, frame_idx in enumerate(frame_indices):
            img_rgb = images[frame_idx]                    # (H, W, 3) RGB uint8
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            out_path = os.path.join(output_dir, f"frame_{frame_idx:05d}.png")
            cv2.imwrite(out_path, img_bgr)

        print(f"Saved {len(frame_indices)} images to '{output_dir}'.")


def main():
    parser = argparse.ArgumentParser(description="Extract oak images from HDF5 demo.")
    parser.add_argument("--hdf5", required=True,
                        help="Path to HDF5 dataset file.")
    parser.add_argument("--demo", required=True,
                        help="Demo name, e.g. 'demo_0'.")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to save extracted images.")
    parser.add_argument("--skip", type=int, default=9,
                        help="Number of frames to skip between saved frames (default: 9).")
    args = parser.parse_args()

    extract_oak_images(
        hdf5_path=args.hdf5,
        demo_name=args.demo,
        output_dir=args.output_dir,
        skip=args.skip,
    )


if __name__ == "__main__":
    main()
