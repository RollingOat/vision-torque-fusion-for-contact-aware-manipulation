"""Synchronize image observations with low-dimensional robot data.

This utility uses image timestamps as the master timeline and aligns the
low-dimensional datapoints (e.g., joint states, forces) by nearest timestamp.
It also stores short force/torque histories to facilitate downstream policies.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Iterable, List, Optional, Tuple

import h5py
import numpy as np


FORCE_HIST_LEN = 10


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Synchronize an image HDF5 file with a low-dimensional HDF5 file"
	)
	parser.add_argument("--image-h5", required=True, help="Path to the image HDF5 file")
	parser.add_argument("--lowdim-h5", required=True, help="Path to the low-dimensional HDF5 file")
	parser.add_argument("--output-h5", required=True, help="Destination path for the synchronized HDF5")
	parser.add_argument(
		"--image-timestamp-key",
		default="timestamp",
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
	parser.add_argument(
		"--force-hist-len",
		type=int,
		default=FORCE_HIST_LEN,
		help="Number of historical samples to keep for force/torque signals",
	)
	parser.add_argument(
		"--ft-mode",
		choices=["single_raw_ft", "mean_raw_ft"],
		default="single_raw_ft",
		help=(
			"How to represent force/torque at each image frame. "
			"'single_raw_ft': nearest single raw F/T sample (default). "
			"'mean_raw_ft': mean of the past --ft-mean-window raw F/T samples "
			"(at 1 kHz, window=30 covers ~30 ms, roughly one action step at 30 Hz)."
		),
	)
	parser.add_argument(
		"--ft-mean-window",
		type=int,
		default=30,
		help=(
			"Number of raw F/T samples to average when --ft-mode=mean_raw_ft. "
			"Defaults to 30 (30 ms at 1 kHz)."
		),
	)
	parser.add_argument(
		"--allow-missing",
		action="store_true",
		help="Skip demos that miss required keys instead of raising an error",
	)
	parser.add_argument(
		"--exclude-demo-index",
		nargs="*",
		type=int,
		help=(
			"Indices (zero-based) of demos to exclude from processing "
			"(based on the sorted overlapping demos list)"
		),
	)
	parser.add_argument(
		"--skip-n",
		type=int,
		default=0,
		dest="skip_n",
		help=(
			"Keep every (skip_n + 1)-th frame and discard the rest. "
			"E.g. --skip-n 2 keeps frames 0, 3, 6, … (default: 0 = keep all frames)."
		),
	)

	# ------------------------------------------------------------------
	# Point-cloud generation options
	# ------------------------------------------------------------------
	parser.add_argument(
		"--pointcloud",
		action="store_true",
		default=False,
		help=(
			"Compute a colored point cloud from agentview_image + agentview_depth "
			"for every synced frame and store it as obs/agentview_pointcloud "
			"(shape T × N × 6, channels [x, y, z, r, g, b]) in the output HDF5."
		),
	)
	parser.add_argument(
		"--calib",
		default=None,
		help="Path to the RealSense calibration JSON (required when --pointcloud is set).",
	)
	parser.add_argument(
		"--pc-config",
		default=None,
		help=(
			"Path to a pointcloud processing config JSON (e.g. pointcloud_config.json). "
			"Values in the file act as defaults; explicit CLI flags take priority."
		),
	)
	parser.add_argument(
		"--depth-trunc",
		type=float,
		default=None,
		help="Discard depth points beyond this distance in metres (config/default: 3.0).",
	)
	parser.add_argument(
		"--pc-voxel-size",
		type=float,
		default=None,
		help=(
			"Voxel size in metres for point-cloud downsampling before fixed-N sampling; "
			"0 = no voxel downsampling (config/default: 0.005)."
		),
	)
	parser.add_argument(
		"--n-points",
		type=int,
		default=None,
		help=(
			"Number of points to sample per cloud (with replacement when the cloud is "
			"smaller).  The output dataset will have shape (T, N, 6).  (config/default: 1024)"
		),
	)
	parser.add_argument(
		"--pc-use-fps",
		action="store_true",
		default=None,
		help=(
			"Use Farthest Point Sampling (FPS) to select the fixed N points. "
			"When absent falls back to random uniform sampling. (config/default: true)"
		),
	)
	parser.add_argument(
		"--pc-center",
		type=float,
		nargs=3,
		metavar=("X", "Y", "Z"),
		default=None,
		help=(
			"Only keep points within --pc-radius metres of this XYZ centre point "
			"(colour-camera frame) before sampling.  Example: --pc-center 0.0 0.0 0.8"
		),
	)
	parser.add_argument(
		"--pc-radius",
		type=float,
		default=None,
		help="Radius in metres for the centre-point filter (config/default: 0.5).",
	)
	parser.add_argument(
		"--pc-depth-aligned",
		action="store_true",
		default=None,
		help=(
			"Indicate that the depth image is already registered (aligned) to the "
			"color image frame.  When set, color intrinsics are used directly for "
			"back-projection and the extrinsic rigid transform is skipped. "
			"(config/default: false)"
		),
	)
	return parser.parse_args()


def validate_files(*paths: str) -> None:
	missing = [path for path in paths if not os.path.exists(path)]
	if missing:
		joined = ", ".join(missing)
		raise FileNotFoundError(f"Missing required file(s): {joined}")


def resolve_dataset_keys(
	group: h5py.Group, timestamp_key: str, explicit: Iterable[str] | None
) -> List[str]:
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


def is_force_key(key: str) -> bool:
	lower = key.lower()
	return "force" in lower or "torque" in lower


def gather_history(data: np.ndarray, end_idx: int, hist_len: int) -> np.ndarray:
	start_idx = max(0, end_idx - hist_len + 1)
	history = data[start_idx : end_idx + 1]
	if history.shape[0] == hist_len:
		return history
	pad_len = hist_len - history.shape[0]
	pad_value = history[0:1]
	padding = np.repeat(pad_value, pad_len, axis=0)
	return np.concatenate([padding, history], axis=0)


def resample_sequence(
	sequence: np.ndarray, follower_ts: np.ndarray, master_ts: np.ndarray
) -> np.ndarray:
	if sequence.shape[0] != follower_ts.shape[0]:
		raise ValueError(
			"Sequence length does not match low-dimensional timestamp count for resampling"
		)
	indices = [find_nearest_idx(follower_ts, t) for t in master_ts]
	return sequence[indices]


def detect_timestamp_jump(timestamps: np.ndarray, threshold: float = 1.0) -> int:
	"""
	Detects if there is a sudden jump in timestamps and returns the index
	of the start of the valid segment after the last jump.
	"""
	if len(timestamps) < 2:
		return 0

	diffs = np.diff(timestamps)
    # Using 1.0 as a safe threshold for "sudden jump". 
    # Normal dt for 100Hz is 0.01, 500Hz is 0.002.
	jump_indices = np.where(diffs > threshold)[0]

	if jump_indices.size > 0:
		# The jump occurs between index i and i+1.
		# We want the index of the start of the valid segment (i+1).
		# We take the *last* detected jump to preserve the most recent valid segment.
		return int(jump_indices[-1] + 1)

	return 0


# ---------------------------------------------------------------------------
# Point-cloud helpers (pure NumPy — no Open3D required)
# ---------------------------------------------------------------------------

def load_calib(json_path: str) -> dict:
	"""Load a RealSense calibration JSON and return a flat dict with keys:
	  ``"color"``       – color-camera intrinsics (fx, fy, cx, cy, width, height)
	  ``"depth"``       – depth-camera intrinsics (fx, fy, cx, cy, width, height)
	  ``"R"``           – (3, 3) float64 rotation  depth → color
	  ``"t"``           – (3,)   float64 translation depth → color
	  ``"depth_scale"`` – scalar float; raw uint16 × scale = metres
	"""
	with open(json_path, "r") as fh:
		raw = json.load(fh)
	c   = raw["color_intrinsics"]
	d   = raw["depth_intrinsics"]
	ext = raw["extrinsics_depth_to_color"]
	return {
		"color": {
			"fx": c["fx"], "fy": c["fy"], "cx": c["cx"], "cy": c["cy"],
			"width": c["width"], "height": c["height"],
		},
		"depth": {
			"fx": d["fx"], "fy": d["fy"], "cx": d["cx"], "cy": d["cy"],
			"width": d["width"], "height": d["height"],
		},
		"R":           np.array(ext["R"], dtype=np.float64),   # (3, 3)
		"t":           np.array(ext["t"], dtype=np.float64),   # (3,)
		"depth_scale": raw["depth_scale_m_per_unit"],
	}


def _scale_intrinsics(intr: dict, img_h: int, img_w: int) -> dict:
	"""Scale intrinsics to match the actual image resolution.

	The calibration JSON is recorded at a fixed resolution (e.g. 640×480).
	If the images stored in the HDF5 were captured or resized to a different
	resolution the focal lengths and principal point must be scaled accordingly.
	"""
	sx = img_w / intr["width"]
	sy = img_h / intr["height"]
	return {
		"fx": intr["fx"] * sx, "fy": intr["fy"] * sy,
		"cx": intr["cx"] * sx, "cy": intr["cy"] * sy,
		"width": img_w, "height": img_h,
	}


def compute_pointcloud_np(
	color_rgb: np.ndarray,
	depth_raw: np.ndarray,
	calib: dict,
	depth_trunc: float = 3.0,
	aligned: bool = False,
) -> np.ndarray:
	"""Project depth + RGB into a colored point cloud.

	Args:
		color_rgb:   uint8 RGB image (H_c, W_c, 3).
		depth_raw:   uint16 raw depth image; units = raw sensor counts.
		             Unaligned: (H_d, W_d) in depth-camera frame.
		             Aligned:   (H_c, W_c) already registered to color frame —
		             each pixel (u, v) in depth corresponds to (u, v) in color.
		calib:       dict from :func:`load_calib`.
		depth_trunc: Discard points beyond this distance in metres.
		aligned:     If True, depth is already registered to the color frame and
		             color intrinsics are used directly for back-projection,
		             skipping the extrinsic rigid transform.

	Returns:
		float32 array of shape (N, 6) with columns [x, y, z, r, g, b].
		xyz are in the color-camera frame (metres); rgb are in [0, 1].
	"""
	scale  = calib["depth_scale"]
	h_c, w_c = color_rgb.shape[:2]

	if aligned:
		# Depth is already registered to the color frame.  Each depth pixel
		# (u, v) corresponds directly to color pixel (u, v).  Back-project
		# using color intrinsics (scaled to the depth image resolution).
		c_intr = _scale_intrinsics(calib["color"], *depth_raw.shape[:2])
		h_d, w_d = depth_raw.shape

		z = depth_raw.flatten().astype(np.float64) * scale
		u = np.tile(np.arange(w_d, dtype=np.float64), h_d)
		v = np.repeat(np.arange(h_d, dtype=np.float64), w_d)

		valid = (z > 0.0) & np.isfinite(z) & (z < depth_trunc)
		z = z[valid];  u = u[valid];  v = v[valid]

		if z.size == 0:
			return np.zeros((0, 6), dtype=np.float32)

		X = (u - c_intr["cx"]) / c_intr["fx"] * z
		Y = (v - c_intr["cy"]) / c_intr["fy"] * z

		# Map depth pixel coordinates to color image coordinates (handles the
		# case where depth and color images differ slightly in resolution).
		u_c = np.clip((u * w_c / w_d).astype(np.int32), 0, w_c - 1)
		v_c = np.clip((v * h_c / h_d).astype(np.int32), 0, h_c - 1)

		xyz = np.stack([X, Y, z], axis=1).astype(np.float32)
		rgb = color_rgb[v_c, u_c].astype(np.float32) / 255.0
		return np.concatenate([xyz, rgb], axis=1)   # (N, 6)

	# --- Unaligned path: depth and color are in separate camera frames. ---
	d_intr = _scale_intrinsics(calib["depth"], *depth_raw.shape[:2])
	c_intr = _scale_intrinsics(calib["color"], *color_rgb.shape[:2])
	R, t   = calib["R"], calib["t"]

	h_d, w_d = depth_raw.shape

	# Flatten and convert to metres.
	z_d = depth_raw.flatten().astype(np.float64) * scale
	u_d = np.tile(np.arange(w_d, dtype=np.float64), h_d)
	v_d = np.repeat(np.arange(h_d, dtype=np.float64), w_d)

	valid = (z_d > 0.0) & np.isfinite(z_d) & (z_d < depth_trunc)
	z_d = z_d[valid];  u_d = u_d[valid];  v_d = v_d[valid]

	if z_d.size == 0:
		return np.zeros((0, 6), dtype=np.float32)

	# Back-project to 3-D in the depth-camera frame.
	X_d = (u_d - d_intr["cx"]) / d_intr["fx"] * z_d
	Y_d = (v_d - d_intr["cy"]) / d_intr["fy"] * z_d
	P_d = np.stack([X_d, Y_d, z_d], axis=0)   # (3, N)

	# Rigid-transform depth → color frame.
	P_c            = R @ P_d + t[:, np.newaxis]   # (3, N)
	X_c, Y_c, Z_c = P_c[0], P_c[1], P_c[2]

	front = Z_c > 0.0
	X_c = X_c[front];  Y_c = Y_c[front];  Z_c = Z_c[front]

	# Project onto the color image.
	u_c = (X_c * c_intr["fx"] / Z_c + c_intr["cx"]).astype(np.int32)
	v_c = (Y_c * c_intr["fy"] / Z_c + c_intr["cy"]).astype(np.int32)

	in_bounds = (u_c >= 0) & (u_c < w_c) & (v_c >= 0) & (v_c < h_c)
	u_c = u_c[in_bounds];  v_c = v_c[in_bounds]
	X_c = X_c[in_bounds];  Y_c = Y_c[in_bounds];  Z_c = Z_c[in_bounds]

	xyz = np.stack([X_c, Y_c, Z_c], axis=1).astype(np.float32)
	rgb = color_rgb[v_c, u_c].astype(np.float32) / 255.0
	return np.concatenate([xyz, rgb], axis=1)   # (N, 6)


def _voxel_downsample_np(pc: np.ndarray, voxel_size: float) -> np.ndarray:
	"""Keep one point per voxel cell (the first point encountered)."""
	if len(pc) == 0 or voxel_size <= 0.0:
		return pc
	voxel_keys = np.floor(pc[:, :3] / voxel_size).astype(np.int64)
	# Pack three int64 indices into a single int64 key via a Cantor-style hash.
	# Use large prime multipliers so that nearby voxels rarely collide.
	keys = (voxel_keys[:, 0] * 1_000_003
			+ voxel_keys[:, 1] * 1_009
			+ voxel_keys[:, 2])
	_, first_idx = np.unique(keys, return_index=True)
	return pc[first_idx]


def _fps_np(pc: np.ndarray, n_points: int) -> np.ndarray:
	"""Farthest Point Sampling (FPS) — pure NumPy fallback.

	Iteratively selects the point whose minimum distance to the already-selected
	set is largest, guaranteeing spatially uniform coverage.  When the input has
	fewer points than requested the cloud is padded by repeating the first point.

	Args:
		pc:       (N, 6) float32 point cloud [x, y, z, r, g, b].
		n_points: Number of points to select.

	Returns:
		(n_points, 6) float32 array.
	"""
	N = len(pc)
	if N == 0:
		return np.zeros((n_points, 6), dtype=np.float32)
	if N <= n_points:
		pad = np.tile(pc[0:1], (n_points - N, 1))
		return np.concatenate([pc, pad], axis=0).astype(np.float32)

	selected  = np.empty(n_points, dtype=np.int64)
	selected[0] = 0
	min_dist  = np.full(N, np.inf, dtype=np.float64)

	for i in range(1, n_points):
		last      = pc[selected[i - 1], :3].astype(np.float64)
		d         = np.sum((pc[:, :3].astype(np.float64) - last) ** 2, axis=1)
		min_dist  = np.minimum(min_dist, d)
		selected[i] = np.argmax(min_dist)

	return pc[selected].astype(np.float32)


def _fps_torch(pc: np.ndarray, n_points: int) -> np.ndarray:
	"""Farthest Point Sampling via PyTorch3D (GPU-accelerated).

	Uses ``pytorch3d.ops.sample_farthest_points`` which runs on CUDA when
	available, matching the method used in 3D-Diffusion-Policy.  Falls back
	gracefully to :func:`_fps_np` if PyTorch3D is not importable.

	Args:
		pc:       (N, 6) float32 point cloud [x, y, z, r, g, b].
		n_points: Number of points to select.

	Returns:
		(n_points, 6) float32 array.
	"""
	if not _HAS_PYTORCH3D:
		return _fps_np(pc, n_points)

	N = len(pc)
	if N == 0:
		return np.zeros((n_points, 6), dtype=np.float32)
	if N <= n_points:
		pad = np.tile(pc[0:1], (n_points - N, 1))
		return np.concatenate([pc, pad], axis=0).astype(np.float32)

	device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
	pts    = _torch.from_numpy(pc[:, :3]).float().unsqueeze(0).to(device)  # (1, N, 3)
	_, idx = _sample_farthest_points(pts, K=n_points)                      # (1, n_points)
	idx    = idx.squeeze(0).cpu().numpy()
	return pc[idx].astype(np.float32)


try:
	import torch as _torch
	from pytorch3d.ops import sample_farthest_points as _sample_farthest_points
	_HAS_PYTORCH3D = True
	_FPS_BACKEND = "pytorch3d"
except ImportError:
	_HAS_PYTORCH3D = False
	_FPS_BACKEND = "numpy"


def _fps(pc: np.ndarray, n_points: int) -> np.ndarray:
	"""Select *n_points* via FPS, using PyTorch3D when available."""
	if _FPS_BACKEND == "pytorch3d":
		return _fps_torch(pc, n_points)
	return _fps_np(pc, n_points)


def _filter_by_radius(pc: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
	"""Keep only points within *radius* metres of *center* (xyz)."""
	if len(pc) == 0:
		return pc
	dist2 = np.sum((pc[:, :3] - center) ** 2, axis=1)
	return pc[dist2 <= radius ** 2]


def _sample_fixed(pc: np.ndarray, n_points: int) -> np.ndarray:
	"""Random uniform sampling to exactly *n_points*; pads by repeating if too few."""
	N = len(pc)
	if N == 0:
		return np.zeros((n_points, 6), dtype=np.float32)
	if N <= n_points:
		pad = np.tile(pc[0:1], (n_points - N, 1))
		return np.concatenate([pc, pad], axis=0).astype(np.float32)
	idx = np.random.choice(N, n_points, replace=False)
	return pc[idx].astype(np.float32)


def build_pointcloud_batch(
	color_frames: np.ndarray,
	depth_frames: np.ndarray,
	calib: dict,
	depth_trunc: float = 3.0,
	voxel_size: float = 0.005,
	n_points: int = 1024,
	use_fps: bool = True,
	center: Optional[np.ndarray] = None,
	radius: float = 0.5,
	aligned: bool = False,
) -> np.ndarray:
	"""Compute colored point clouds for every frame in a demo.

	Pipeline per frame:
	  1. Back-project depth to 3-D, colour from RGB image            → (N, 6)
	     (unaligned: uses extrinsic R,t; aligned: uses color intrinsics directly)
	  2. Radius filter around *center* (optional)                     → (K, 6)
	  3. Voxel downsample (optional, reduces N before sampling)       → (M, 6)
	  4. Fixed-N sampling: FPS when *use_fps* else random             → (n_points, 6)

	Args:
		color_frames: (T, H_c, W_c, 3) uint8 RGB.
		depth_frames: (T, H_d, W_d)    uint16 raw depth (unaligned), or
		              (T, H_c, W_c)    uint16 raw depth already registered to
		              the color frame when *aligned* is True.
		calib:        dict from :func:`load_calib`.
		depth_trunc:  Discard points beyond this distance in metres.
		voxel_size:   Voxel size for pre-sampling downsampling (metres); 0 = skip.
		n_points:     Fixed number of points per frame in the output.
		use_fps:      Use Farthest Point Sampling; falls back to random sampling if False.
		center:       If given, keep only points within *radius* m of this xyz point.
		radius:       Radius in metres for the centre-point filter.
		aligned:      If True, depth is already registered to the color frame and
		              color intrinsics are used directly for back-projection.

	Returns:
		float32 array of shape (T, n_points, 6) with columns [x, y, z, r, g, b].
	"""
	print(f"  [pointcloud] depth_rgb_aligned={aligned}  "
		  f"sampling={'FPS (' + _FPS_BACKEND + ')' if use_fps else 'random'}")
	if center is not None:
		print(f"  [pointcloud] radius filter: center={center.tolist()}  radius={radius} m")
	T   = color_frames.shape[0]
	out = np.zeros((T, n_points, 6), dtype=np.float32)
	for i in range(T):
		pc = compute_pointcloud_np(color_frames[i], depth_frames[i], calib, depth_trunc, aligned)
		if center is not None:
			pc = _filter_by_radius(pc, center, radius)
		if voxel_size > 0.0:
			pc = _voxel_downsample_np(pc, voxel_size)
		out[i] = _fps(pc, n_points) if use_fps else _sample_fixed(pc, n_points)
	return out


def sync_demo(
	demo: str,
	image_obs: h5py.Group,
	lowdim_obs: h5py.Group,
	image_ts_key: str,
	lowdim_ts_key: str,
	image_keys: List[str],
	lowdim_keys: List[str],
	hist_len: int,
	ft_mode: str = "single_raw_ft",
	ft_mean_window: int = 30,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], np.ndarray]:
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

	# Guard against timestamp array being longer than the actual image data
	if master_cache:
		min_cache_len = min(v.shape[0] for v in master_cache.values())
		if master_timestamps.size > min_cache_len:
			print(f"Warning: master_timestamps has {master_timestamps.size} entries but image cache has {min_cache_len}; truncating timestamps for demo {demo}.")
			master_timestamps = master_timestamps[:min_cache_len]

	# Remove any zero timestamp entries from low-dim data (often initialization artifacts)
	non_zero_mask = follower_timestamps > 1e-6
	if not np.all(non_zero_mask):
		print(f"Warning: Discarding {np.sum(~non_zero_mask)} zero-valued timestamps from low-dim data for demo {demo}")
		follower_timestamps = follower_timestamps[non_zero_mask]
		for k in follower_cache:
			follower_cache[k] = follower_cache[k][non_zero_mask]
		
		if follower_timestamps.size == 0:
			raise ValueError(f"Demo {demo} has only zero-valued low-dimensional timestamps")

	# Detect sudden jumps (gaps > 0.5s) in low-dimensional timestamps and discard data before the last jump.
	# Typically gaps are tiny (0.01s for 100Hz). A sudden jump implies separate recording segments.
	jump_idx = detect_timestamp_jump(follower_timestamps, threshold=0.5)
	if jump_idx > 0:
		print(f"Warning: Sudden jump detected in low-dim timestamps for demo {demo} at index {jump_idx}. Discarding {jump_idx} samples before the jump.")
		follower_timestamps = follower_timestamps[jump_idx:]
		for k in follower_cache:
			follower_cache[k] = follower_cache[k][jump_idx:]
		if follower_timestamps.size == 0:
			print(f"Warning: Discarding all low-dim timestamps due to jump for demo {demo}; skipping demo")
			return None, follower_timestamps

	# Determine overlap interval between master (image) and follower (low-dim) timestamps.
	# We strictly intersect the ranges.
	low_start, low_end = np.min(follower_timestamps), np.max(follower_timestamps)
	img_start, img_end = np.min(master_timestamps), np.max(master_timestamps)
	
	overlap_start = max(img_start, low_start)
	overlap_end = min(img_end, low_end)
	
	print(f"Demo {demo} timestamp overlap: [{overlap_start:.3f}, {overlap_end:.3f}]")

	if overlap_start > overlap_end:
		print(f"Warning: No timestamp overlap between image and low-dim for demo {demo}; skipping demo")
		return None, follower_timestamps

	# Find master (image) timestamps that fall within the overlap interval based on closest logic.
    # 1. Start: Closest image timestamp to overlap_start that is >= overlap_start
	# 2. End: Closest image timestamp to overlap_end that is <= overlap_end
	
	# Find candidate indices roughly
	candidates_mask = (master_timestamps >= overlap_start) & (master_timestamps <= overlap_end)
	candidate_indices = np.where(candidates_mask)[0]
	
	if candidate_indices.size == 0:
		print(f"Warning: No image timestamps fall within the overlap interval for demo {demo}; skipping demo")
		return None, follower_timestamps

	# Refine start: First index in candidates is by definition closest >= overlap_start since array is sorted
	start_idx = candidate_indices[0]
	print(f"Demo {demo} master start idx: {start_idx}, timestamp: {master_timestamps[start_idx]:.3f}")
	# Refine end: Last index in candidates is by definition closest <= overlap_end
	end_idx = candidate_indices[-1]
	
	master_indices = np.arange(start_idx, end_idx + 1)

	# Slice caches to only include overlapping master frames to simplify indexing below.
	master_cache_sliced = {k: v[master_indices] for k, v in master_cache.items()}

	synced_images: Dict[str, List[np.ndarray]] = {key: [] for key in image_keys}
	synced_lowdim: Dict[str, List[np.ndarray]] = {key: [] for key in lowdim_keys}

	# Iterate only over image timestamps inside the overlap and find nearest low-dim samples.
	master_in_ts = master_timestamps[master_indices]
	for local_idx, timestamp in enumerate(master_in_ts):
		timestamp = float(timestamp)
		follower_idx = find_nearest_idx(follower_timestamps, timestamp)
		time_diff = abs(follower_timestamps[follower_idx] - timestamp)
		if time_diff > 0.1:
			raise ValueError(
				f"Timestamp mismatch at master idx {master_indices[local_idx]} (master ts: {timestamp}, nearest follower ts: {follower_timestamps[follower_idx]}, diff: {time_diff})"
			)

		for key in image_keys:
			synced_images[key].append(master_cache_sliced[key][local_idx])
		for key in lowdim_keys:
			if ft_mode == "mean_raw_ft" and is_force_key(key):
				# Average the past `ft_mean_window` raw F/T samples up to (and
				# including) the nearest follower index.  At 1 kHz a window of
				# 30 spans ~30 ms, aligning with one action step at 30 Hz.
				start = max(0, follower_idx - ft_mean_window + 1)
				follower_sample = follower_cache[key][start : follower_idx + 1].mean(axis=0)
			else:
				follower_sample = follower_cache[key][follower_idx]
			synced_lowdim[key].append(follower_sample)

	image_arrays = {key: np.stack(values, axis=0) for key, values in synced_images.items()}
	lowdim_arrays = {key: np.stack(values, axis=0) for key, values in synced_lowdim.items()}

	return (
		{
			"timestamps": master_in_ts,
			"image_obs": image_arrays,
			"lowdim_obs": lowdim_arrays,
		},
		follower_timestamps,
	)


def write_demo(
	demo: str,
	out_root: h5py.Group,
	synced: Dict[str, Dict[str, np.ndarray]],
	image_ts_key: str,
	actions: Optional[np.ndarray],
	pointcloud: Optional[np.ndarray] = None,
) -> None:
	g_demo = out_root.create_group(demo)
	g_obs = g_demo.create_group("obs")

	if actions is not None:
		g_demo.create_dataset("actions", data=actions)
	g_obs.create_dataset(image_ts_key, data=synced["timestamps"])
	for key, arr in synced["image_obs"].items():
		g_obs.create_dataset(key, data=arr)
	for key, arr in synced["lowdim_obs"].items():
		g_obs.create_dataset(key, data=arr)
	if pointcloud is not None:
		g_obs.create_dataset("agentview_pointcloud", data=pointcloud)

	g_demo.attrs["num_samples"] = synced["timestamps"].shape[0]


_PC_DEFAULTS = {
	"depth_trunc":      3.0,
	"voxel_size":       0.005,
	"n_points":         1024,
	"use_fps":          True,
	"center":           None,
	"radius":           0.5,
	"depth_rgb_aligned": False,
}


def _resolve_pc_args(args: argparse.Namespace) -> dict:
	"""Merge CLI args > pointcloud config file > hardcoded defaults."""
	cfg: dict = {}
	if getattr(args, "pc_config", None):
		with open(args.pc_config) as fh:
			cfg = json.load(fh)

	def _pick(cli_val, key):
		if cli_val is not None:
			return cli_val
		return cfg.get(key, _PC_DEFAULTS[key])

	center_raw = _pick(getattr(args, "pc_center", None), "center")
	return {
		"depth_trunc":      _pick(args.depth_trunc,    "depth_trunc"),
		"voxel_size":       _pick(args.pc_voxel_size,  "voxel_size"),
		"n_points":         _pick(args.n_points,        "n_points"),
		"use_fps":          _pick(getattr(args, "pc_use_fps", None), "use_fps"),
		"center":           np.array(center_raw, dtype=np.float64) if center_raw is not None else None,
		"radius":           _pick(getattr(args, "pc_radius", None), "radius"),
		"depth_rgb_aligned": _pick(getattr(args, "pc_depth_aligned", None), "depth_rgb_aligned"),
	}


def main() -> None:
	args = parse_args()

	# ------------------------------------------------------------------
	# Resolve point-cloud processing parameters (CLI > config > defaults)
	# ------------------------------------------------------------------
	pc = _resolve_pc_args(args)

	# Validate point-cloud arguments before doing any file I/O.
	calib = None
	if args.pointcloud:
		if not args.calib:
			raise ValueError("--calib is required when --pointcloud is set")
		if not os.path.exists(args.calib):
			raise FileNotFoundError(f"Calibration file not found: {args.calib}")
		calib = load_calib(args.calib)
		print(
			f"[pointcloud] calib={args.calib}  depth_trunc={pc['depth_trunc']} m  "
			f"voxel={pc['voxel_size']} m  n_points={pc['n_points']}  "
			f"use_fps={pc['use_fps']}  center={pc['center']}  radius={pc['radius']} m  "
			f"depth_rgb_aligned={pc['depth_rgb_aligned']}"
		)

	validate_files(args.image_h5, args.lowdim_h5)
	if os.path.abspath(args.image_h5) == os.path.abspath(args.output_h5):
		raise ValueError("Output file must differ from the image input file")
	if os.path.abspath(args.lowdim_h5) == os.path.abspath(args.output_h5):
		raise ValueError("Output file must differ from the low-dimensional input file")

	with h5py.File(args.image_h5, "r") as f_image, h5py.File(args.lowdim_h5, "r") as f_lowdim:
		if "data" not in f_image or "data" not in f_lowdim:
			raise KeyError("Both HDF5 files must contain a top-level 'data' group")
		demos = sorted(set(f_image["data"].keys()) & set(f_lowdim["data"].keys()))
		# Apply demo-index exclusion if requested. Indices refer to the
		# zero-based positions in the sorted overlapping demos list.
		if getattr(args, "exclude_demo_index", None):
			exclude_set = set(args.exclude_demo_index)
			# Warn about any out-of-range indices and ignore them
			invalid = [i for i in exclude_set if i < 0 or i >= len(demos)]
			if invalid:
				print(f"Warning: --exclude-demo-index contains out-of-range indices {invalid}; ignoring them")
			exclude_set = {i for i in exclude_set if 0 <= i < len(demos)}
			if exclude_set:
				demos = [d for idx, d in enumerate(demos) if idx not in exclude_set]
				if not demos:
					raise ValueError("No demos left after applying --exclude-demo-index filter")
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
				
					image_keys = resolve_dataset_keys(
						image_obs, args.image_timestamp_key, args.image_keys
					)
					lowdim_keys = resolve_dataset_keys(
						lowdim_obs, args.lowdim_timestamp_key, args.lowdim_keys
					)
					result = sync_demo(
						demo,
						image_obs,
						lowdim_obs,
						args.image_timestamp_key,
						args.lowdim_timestamp_key,
						image_keys,
						lowdim_keys,
						args.force_hist_len,
						ft_mode=args.ft_mode,
						ft_mean_window=args.ft_mean_window,
					)
					if result[0] is None:
						# sync_demo already printed a warning; skip this demo.
						continue
					synced, follower_ts = result
				except Exception as exc:
					if args.allow_missing:
						print(f"Skipping {demo}: {exc}")
						continue
					raise

				# Apply frame skipping: keep every (skip_n+1)-th frame.
				if args.skip_n > 0:
					step = args.skip_n + 1
					indices = np.arange(0, synced["timestamps"].shape[0], step)
					if len(indices) < 2:
						print(f"  Skipping {demo}: too few frames after --skip-n {args.skip_n} subsampling.")
						continue
					synced["timestamps"] = synced["timestamps"][indices]
					for k in synced["image_obs"]:
						synced["image_obs"][k] = synced["image_obs"][k][indices]
					for k in synced["lowdim_obs"]:
						synced["lowdim_obs"][k] = synced["lowdim_obs"][k][indices]
					print(f"  [skip_n={args.skip_n}] {len(indices)} frames kept (step={step})")

				actions_data = None
				if "actions" in lowdim_demo:
					try:
						actions_source = lowdim_demo["actions"][:]
						actions_data = resample_sequence(
							actions_source, follower_ts, synced["timestamps"]
						)
					except ValueError as exc:
						print(f"Skipping actions for {demo}: {exc}")

				pc_data = None
				if calib is not None:
					img_obs = synced["image_obs"]
					if "agentview_image" in img_obs and "agentview_depth" in img_obs:
						print(f"  [pointcloud] building clouds for {demo} "
							  f"({img_obs['agentview_image'].shape[0]} frames)…")
						pc_data = build_pointcloud_batch(
							img_obs["agentview_image"],
							img_obs["agentview_depth"],
							calib,
							depth_trunc=pc["depth_trunc"],
							voxel_size=pc["voxel_size"],
							n_points=pc["n_points"],
							use_fps=pc["use_fps"],
							center=pc["center"],
							radius=pc["radius"],
							aligned=pc["depth_rgb_aligned"],
						)
						print(f"  [pointcloud] agentview_pointcloud "
							  f"shape={pc_data.shape} dtype={pc_data.dtype}")
					else:
						print(f"  [pointcloud] skipping {demo}: "
							  "agentview_image or agentview_depth not in synced obs")

				write_demo(demo, g_out, synced, args.image_timestamp_key, actions_data, pc_data)
				processed += 1
				print(f"Synchronized {demo}: {synced['timestamps'].shape[0]} frames")

			f_out.attrs["source_image_h5"] = os.path.abspath(args.image_h5)
			f_out.attrs["source_lowdim_h5"] = os.path.abspath(args.lowdim_h5)
			f_out.attrs["num_synced_demos"] = processed
			print(f"Finished syncing {processed} demo(s) to {args.output_h5}")


if __name__ == "__main__":
	main()
