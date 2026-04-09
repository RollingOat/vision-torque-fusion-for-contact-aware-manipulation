#!/usr/bin/env python3
"""
Brighten all images in a folder using gamma correction.

Usage:
    python brighten_images.py --input_dir ./output_frames --gamma 1.8
    python brighten_images.py --input_dir ./output_frames --output_dir ./bright_frames --gamma 2.0
"""

import argparse
import os

import cv2
import numpy as np


def build_gamma_lut(gamma):
    table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                      for i in range(256)], dtype=np.uint8)
    return table


def brighten_images(input_dir, output_dir, gamma):
    os.makedirs(output_dir, exist_ok=True)
    lut = build_gamma_lut(gamma)

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    files = sorted(f for f in os.listdir(input_dir)
                   if os.path.splitext(f)[1].lower() in exts)

    if not files:
        print(f"No image files found in '{input_dir}'.")
        return

    for fname in files:
        img = cv2.imread(os.path.join(input_dir, fname))
        if img is None:
            print(f"  Skipping unreadable file: {fname}")
            continue
        bright = cv2.LUT(img, lut)
        cv2.imwrite(os.path.join(output_dir, fname), bright)

    print(f"Brightened {len(files)} images (gamma={gamma}) -> '{output_dir}'.")


def main():
    parser = argparse.ArgumentParser(description="Brighten images in a folder via gamma correction.")
    parser.add_argument("--input_dir", required=True, help="Folder containing input images.")
    parser.add_argument("--output_dir", default=None,
                        help="Folder to save brightened images. Defaults to overwriting input_dir.")
    parser.add_argument("--gamma", type=float, default=1.8,
                        help="Gamma value >1 brightens, <1 darkens (default: 1.8).")
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir else args.input_dir
    brighten_images(args.input_dir, output_dir, args.gamma)


if __name__ == "__main__":
    main()
