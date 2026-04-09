#!/usr/bin/env python3
"""
hdf5_pointcloud_viewer.py

Reads unaligned RGB and depth frames from a robomimic HDF5 file, aligns the
depth stream to the color camera using RealSense calibration parameters, builds
a colored Open3D point cloud per frame, and displays it in a real-time 3-D viewer.

The key difference from ``realsense_aligned_viewer.py`` (which assumes images are
already spatially aligned) is that here the depth and color images live in
*different* camera coordinate frames.  Alignment is done mathematically:

    1. Back-project each valid depth pixel to a 3-D point in the depth-camera frame.
    2. Rigid-transform that point to the color-camera frame using the
       extrinsic rotation R and translation t stored in the calibration JSON.
    3. Project the transformed point onto the color image plane to sample its RGB.

Expected HDF5 layout (robomimic format)
----------------------------------------
    data/
      demo_0/
        obs/
          agentview_image  (T, H, W, 3)  uint8   – RGB color frames
          agentview_depth  (T, H, W)     uint16  – raw depth in mm (unaligned)
      demo_1/
        ...

Usage
-----
    python hdf5_pointcloud_viewer.py \\
        --hdf5  /path/to/test_pc.hdf5 \\
        --calib /path/to/realsense_calib_405622072966.json \\
        [--demo   demo_0]   \\
        [--fps    10]       \\
        [--depth-trunc 3.0] \\
        [--voxel-size  0.005]
"""

import argparse
import importlib
import json
import time
from typing import Optional

import h5py
import numpy as np
import open3d as o3d


# ---------------------------------------------------------------------------
# Calibration loader
# ---------------------------------------------------------------------------

def load_calib(json_path: str) -> dict:
    """Load RealSense calibration from the JSON file produced by the capture rig.

    Returns a flat dict with:
        "color"       – color-camera intrinsics dict (fx, fy, cx, cy, width, height)
        "depth"       – depth-camera intrinsics dict (fx, fy, cx, cy, width, height)
        "R"           – (3, 3) float64 rotation:    depth-camera → color-camera
        "t"           – (3,)   float64 translation: depth-camera → color-camera
        "depth_scale" – scalar float; raw_uint16 × scale = metres
    """
    with open(json_path, "r") as fh:
        raw = json.load(fh)

    c = raw["color_intrinsics"]
    d = raw["depth_intrinsics"]
    ext = raw["extrinsics_depth_to_color"]

    return {
        "color": {
            "fx": c["fx"], "fy": c["fy"],
            "cx": c["cx"], "cy": c["cy"],
            "width": c["width"], "height": c["height"],
        },
        "depth": {
            "fx": d["fx"], "fy": d["fy"],
            "cx": d["cx"], "cy": d["cy"],
            "width": d["width"], "height": d["height"],
        },
        "R": np.array(ext["R"], dtype=np.float64),   # (3, 3)
        "t": np.array(ext["t"], dtype=np.float64),   # (3,)
        "depth_scale": raw["depth_scale_m_per_unit"],
    }


# ---------------------------------------------------------------------------
# Core: unaligned depth + RGB → (N, 6) numpy array
# ---------------------------------------------------------------------------

def compute_pointcloud_np(
    color_rgb: np.ndarray,
    depth_raw: np.ndarray,
    calib: dict,
    depth_trunc: float = 3.0,
) -> np.ndarray:
    """Back-project unaligned depth + RGB into a (N, 6) float32 array [x,y,z,r,g,b].

    Args:
        color_rgb:   uint8 RGB image (H_c, W_c, 3).
        depth_raw:   uint16 raw depth image (H_d, W_d); raw counts × depth_scale = m.
        calib:       dict returned by :func:`load_calib`.
        depth_trunc: Discard points beyond this distance in metres.

    Returns:
        float32 array of shape (N, 6); xyz in colour-camera frame (metres),
        rgb in [0, 1].  Returns shape (0, 6) when no valid depth exists.
    """
    d_intr = calib["depth"]
    c_intr = calib["color"]
    R      = calib["R"]
    t      = calib["t"]
    scale  = calib["depth_scale"]

    h_d, w_d = depth_raw.shape
    h_c, w_c = color_rgb.shape[:2]

    z_d = depth_raw.flatten().astype(np.float64) * scale
    u_d = np.tile(np.arange(w_d, dtype=np.float64), h_d)
    v_d = np.repeat(np.arange(h_d, dtype=np.float64), w_d)

    valid = (z_d > 0.0) & np.isfinite(z_d) & (z_d < depth_trunc)
    z_d = z_d[valid];  u_d = u_d[valid];  v_d = v_d[valid]

    if z_d.size == 0:
        return np.zeros((0, 6), dtype=np.float32)

    X_d = (u_d - d_intr["cx"]) / d_intr["fx"] * z_d
    Y_d = (v_d - d_intr["cy"]) / d_intr["fy"] * z_d
    P_d = np.stack([X_d, Y_d, z_d], axis=0)

    P_c            = R @ P_d + t[:, np.newaxis]
    X_c, Y_c, Z_c = P_c[0], P_c[1], P_c[2]

    front = Z_c > 0.0
    X_c = X_c[front];  Y_c = Y_c[front];  Z_c = Z_c[front]

    u_c = (X_c * c_intr["fx"] / Z_c + c_intr["cx"]).astype(np.int32)
    v_c = (Y_c * c_intr["fy"] / Z_c + c_intr["cy"]).astype(np.int32)

    in_bounds = (u_c >= 0) & (u_c < w_c) & (v_c >= 0) & (v_c < h_c)
    u_c = u_c[in_bounds];  v_c = v_c[in_bounds]
    X_c = X_c[in_bounds];  Y_c = Y_c[in_bounds];  Z_c = Z_c[in_bounds]

    xyz = np.stack([X_c, Y_c, Z_c], axis=1).astype(np.float32)
    rgb = color_rgb[v_c, u_c].astype(np.float32) / 255.0
    return np.concatenate([xyz, rgb], axis=1)   # (N, 6)


def _np_to_o3d(pc: np.ndarray) -> o3d.geometry.PointCloud:
    """Convert a (N, 6) float32 [x,y,z,r,g,b] array to an Open3D PointCloud."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc[:, :3].astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(pc[:, 3:].astype(np.float64))
    return pcd


# ---------------------------------------------------------------------------
# FPS sampling helpers
# ---------------------------------------------------------------------------

def _voxel_downsample_np(pc: np.ndarray, voxel_size: float) -> np.ndarray:
    """Keep one point per voxel cell (first encountered) — pure NumPy."""
    if len(pc) == 0 or voxel_size <= 0.0:
        return pc
    keys = np.floor(pc[:, :3] / voxel_size).astype(np.int64)
    hkeys = keys[:, 0] * 1_000_003 + keys[:, 1] * 1_009 + keys[:, 2]
    _, first = np.unique(hkeys, return_index=True)
    return pc[first]


def _fps_np(pc: np.ndarray, n_points: int) -> np.ndarray:
    """Farthest Point Sampling — pure NumPy fallback.

    Pads by repeating the first point when the cloud has fewer than n_points.
    """
    N = len(pc)
    if N == 0:
        return np.zeros((n_points, 6), dtype=np.float32)
    if N <= n_points:
        pad = np.tile(pc[0:1], (n_points - N, 1))
        return np.concatenate([pc, pad], axis=0).astype(np.float32)

    selected    = np.empty(n_points, dtype=np.int64)
    selected[0] = 0
    min_dist    = np.full(N, np.inf, dtype=np.float64)

    for i in range(1, n_points):
        last         = pc[selected[i - 1], :3].astype(np.float64)
        d            = np.sum((pc[:, :3].astype(np.float64) - last) ** 2, axis=1)
        min_dist     = np.minimum(min_dist, d)
        selected[i]  = np.argmax(min_dist)

    return pc[selected].astype(np.float32)


def _fps_torch(pc: np.ndarray, n_points: int) -> np.ndarray:
    """Farthest Point Sampling via PyTorch3D (GPU-accelerated when CUDA is available).

    Falls back to :func:`_fps_np` if PyTorch3D is not installed.
    """
    try:
        import torch
        from pytorch3d.ops import sample_farthest_points
    except ImportError:
        return _fps_np(pc, n_points)

    N = len(pc)
    if N == 0:
        return np.zeros((n_points, 6), dtype=np.float32)
    if N <= n_points:
        pad = np.tile(pc[0:1], (n_points - N, 1))
        return np.concatenate([pc, pad], axis=0).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pts    = torch.from_numpy(pc[:, :3]).float().unsqueeze(0).to(device)
    _, idx = sample_farthest_points(pts, K=n_points)
    idx    = idx.squeeze(0).cpu().numpy()
    return pc[idx].astype(np.float32)


_FPS_BACKEND = (
    "pytorch3d"
    if (importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("pytorch3d") is not None)
    else "numpy"
)


def _fps(pc: np.ndarray, n_points: int) -> np.ndarray:
    """Dispatch FPS to PyTorch3D or NumPy depending on availability."""
    if _FPS_BACKEND == "pytorch3d":
        return _fps_torch(pc, n_points)
    return _fps_np(pc, n_points)


def _filter_by_radius(pc: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    """Keep only points within *radius* metres of *center* (xyz)."""
    if len(pc) == 0:
        return pc
    dist2 = np.sum((pc[:, :3] - center) ** 2, axis=1)
    return pc[dist2 <= radius ** 2]


# ---------------------------------------------------------------------------
# Real-time viewer
# ---------------------------------------------------------------------------

def run_viewer(
    hdf5_path: str,
    calib_path: str,
    demo_key: str = "demo_0",
    fps: float = 10.0,
    depth_trunc: float = 3.0,
    voxel_size: float = 0.005,
    n_points: int = 1024,
    use_fps: bool = True,
    center: Optional[np.ndarray] = None,
    radius: float = 0.5,
    point_size: float = 2.0,
) -> None:
    """Iterate over frames in an HDF5 demo and show a live point cloud.

    Pipeline per frame:
      1. Back-project unaligned depth to 3-D, colour from RGB  → (N, 6) numpy
      2. Optional radius filter around *center*                 → (K, 6) numpy
      3. Voxel downsample (optional, speeds up FPS)            → (M, 6) numpy
      4. Optional Farthest Point Sampling to exactly n_points   → (n_points, 6)
      5. Display in Open3D
    """

    calib = load_calib(calib_path)
    print(f"[hdf5_pointcloud_viewer] Loaded calibration: {calib_path}")
    print(f"  color intrinsics : {calib['color']}")
    print(f"  depth intrinsics : {calib['depth']}")
    print(f"  depth scale      : {calib['depth_scale']} m/unit")
    print(f"  FPS backend      : {_FPS_BACKEND}")
    if center is not None:
        print(f"  radius filter    : center={center.tolist()}  radius={radius} m")
    if use_fps:
        print(f"  FPS sampling     : enabled  n_points={n_points}")
    else:
        print(f"  FPS sampling     : disabled")

    hdf5_file = h5py.File(hdf5_path, "r")
    obs      = hdf5_file[f"data/{demo_key}/obs"]
    color_ds = obs["agentview_image"]   # (T, H, W, 3) uint8
    depth_ds = obs["agentview_depth"]   # (T, H, W)    uint16
    n_frames = color_ds.shape[0]

    print(f"[hdf5_pointcloud_viewer] demo={demo_key}, frames={n_frames}, "
          f"color={color_ds.shape[1:]}, depth={depth_ds.shape[1:]}")
    print(f"[hdf5_pointcloud_viewer] depth_trunc={depth_trunc} m, "
          f"voxel_size={voxel_size} m, playback_fps={fps}")

    title = "HDF5 Point Cloud"
    if use_fps:
        title += f" — FPS {n_points} pts ({_FPS_BACKEND})"
    vis = o3d.visualization.Visualizer()
    vis.create_window(title, width=1280, height=720)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.zeros((1, 3), dtype=np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.zeros((1, 3), dtype=np.float64))
    vis.add_geometry(pcd)
    vis.get_render_option().point_size = point_size

    frame_dt    = 1.0 / fps
    first_frame = True

    try:
        for i in range(n_frames):
            t0 = time.time()

            color_rgb = color_ds[i]   # (H, W, 3) uint8
            depth_raw = depth_ds[i]   # (H, W)    uint16

            # 1. Raw point cloud as numpy array.
            pc = compute_pointcloud_np(color_rgb, depth_raw, calib, depth_trunc)

            # 2. Optional radius filter around the specified center point.
            if center is not None:
                pc = _filter_by_radius(pc, center, radius)

            # 3. Optional voxel pre-downsampling (reduces input size for FPS).
            if voxel_size > 0.0:
                pc = _voxel_downsample_np(pc, voxel_size)

            # 4. Optional FPS to fixed n_points.
            if use_fps:
                pc = _fps(pc, n_points)

            # 5. Convert to Open3D and display.
            new_pcd = _np_to_o3d(pc)
            n_pts   = len(new_pcd.points)

            if n_pts == 0:
                n_valid = int(np.count_nonzero(depth_raw))
                print(f"\n[frame {i:04d}] empty cloud  "
                      f"(non-zero depth px={n_valid}, "
                      f"range=[{depth_raw.min()}, {depth_raw.max()}])")
            else:
                pcd.points = new_pcd.points
                pcd.colors = new_pcd.colors
                vis.update_geometry(pcd)

                if first_frame:
                    vis.reset_view_point(True)
                    first_frame = False

            print(f"[frame {i+1:04d}/{n_frames}]  {n_pts:>6} pts  "
                  f"t={time.time()-t0:.3f}s",
                  end="\r", flush=True)

            if vis.poll_events() is False:
                print("\n[hdf5_pointcloud_viewer] Window closed by user.")
                break
            vis.update_renderer()

            elapsed = time.time() - t0
            wait = frame_dt - elapsed
            if wait > 0:
                time.sleep(wait)

    finally:
        hdf5_file.close()
        vis.destroy_window()
        print("\n[hdf5_pointcloud_viewer] Done.")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

_PC_DEFAULTS = {
    "depth_trunc": 3.0,
    "voxel_size":  0.005,
    "n_points":    1024,
    "use_fps":     False,
    "center":      None,
    "radius":      0.5,
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

    center_raw = _pick(args.center, "center")
    return {
        "depth_trunc": _pick(args.depth_trunc, "depth_trunc"),
        "voxel_size":  _pick(args.voxel_size,  "voxel_size"),
        "n_points":    _pick(args.n_points,     "n_points"),
        "use_fps":     _pick(args.use_fps,      "use_fps"),
        "center":      np.array(center_raw, dtype=np.float64) if center_raw is not None else None,
        "radius":      _pick(args.radius,       "radius"),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Real-time colored point cloud viewer from a robomimic HDF5 file"
    )
    parser.add_argument(
        "--hdf5", required=True,
        help="Path to the .hdf5 dataset file  (e.g. datasets/test_pc.hdf5)",
    )
    parser.add_argument(
        "--calib", required=True,
        help="Path to the RealSense calibration JSON  "
             "(e.g. realsense_calib_405622072966.json)",
    )
    parser.add_argument(
        "--demo", default="demo_0",
        help="Which demo to play back (default: demo_0)",
    )
    parser.add_argument(
        "--fps", type=float, default=10.0,
        help="Playback speed in frames-per-second (default: 10)",
    )
    parser.add_argument(
        "--point-size", type=float, default=10.0,
        help="Rendered point size in pixels (default: 10.0)",
    )
    # ------------------------------------------------------------------
    # Point-cloud processing args (all default to None so the config
    # file can supply values; explicit CLI flags always take priority)
    # ------------------------------------------------------------------
    parser.add_argument(
        "--pc-config", default=None,
        help="Path to a pointcloud processing config JSON (e.g. pointcloud_config.json). "
             "Values in the file act as defaults; explicit CLI flags take priority.",
    )
    parser.add_argument(
        "--depth-trunc", type=float, default=None,
        help="Discard 3-D points farther than this distance in metres (config/default: 3.0)",
    )
    parser.add_argument(
        "--voxel-size", type=float, default=None,
        help="Voxel pre-downsampling before sampling in metres; 0 = skip (config/default: 0.005)",
    )
    parser.add_argument(
        "--n-points", type=int, default=None,
        help="Number of points after FPS sampling (config/default: 1024)",
    )
    parser.add_argument(
        "--use-fps", action="store_true", default=None,
        help="Enable Farthest Point Sampling to --n-points after voxel downsampling",
    )
    parser.add_argument(
        "--center", type=float, nargs=3, metavar=("X", "Y", "Z"), default=None,
        help="Only keep points within --radius metres of this XYZ centre point "
             "(colour-camera frame).  Example: --center 0.0 0.0 0.8",
    )
    parser.add_argument(
        "--radius", type=float, default=None,
        help="Radius in metres for the centre-point filter (config/default: 0.5)",
    )
    args = parser.parse_args()

    pc = _resolve_pc_args(args)

    run_viewer(
        hdf5_path=args.hdf5,
        calib_path=args.calib,
        demo_key=args.demo,
        fps=args.fps,
        depth_trunc=pc["depth_trunc"],
        voxel_size=pc["voxel_size"],
        n_points=pc["n_points"],
        use_fps=pc["use_fps"],
        center=pc["center"],
        radius=pc["radius"],
        point_size=args.point_size,
    )


if __name__ == "__main__":
    main()
