import numpy as np
import cv2

def align_depth_to_color_offline(depth_image, color_image, depth_intrin, color_intrin, extrinsics):
    """
    Mathematically warps a depth image to align with a color image.
    """
    height, width = depth_image.shape
    aligned_depth = np.zeros((color_image.shape[0], color_image.shape[1]), dtype=depth_image.dtype)

    # 1. Create a grid of all (u_d, v_d) coordinates
    u_d, v_d = np.meshgrid(np.arange(width), np.arange(height))
    
    # Flatten the arrays for vectorized processing
    u_d = u_d.flatten()
    v_d = v_d.flatten()
    z_d = depth_image.flatten() * 0.001 # Assuming 1000.0 depth scale (meters)

    # Filter out valid depths (Z > 0)
    valid = z_d > 0
    u_d, v_d, z_d = u_d[valid], v_d[valid], z_d[valid]

    # 2. Deprojection (Depth 2D -> Depth 3D)
    x_d = (u_d - depth_intrin['cx']) * z_d / depth_intrin['fx']
    y_d = (v_d - depth_intrin['cy']) * z_d / depth_intrin['fy']
    P_d = np.vstack((x_d, y_d, z_d)) # Shape: (3, N)

    # 3. Rigid Transformation (Depth 3D -> Color 3D)
    R = np.array(extrinsics['rotation']).reshape(3, 3)
    t = np.array(extrinsics['translation']).reshape(3, 1)
    P_c = (R @ P_d) + t # Matrix multiplication broadcasting

    # Extract X_c, Y_c, Z_c
    x_c, y_c, z_c = P_c[0, :], P_c[1, :], P_c[2, :]

    # 4. Projection (Color 3D -> Color 2D)
    u_c = np.round((x_c * color_intrin['fx'] / z_c) + color_intrin['cx']).astype(int)
    v_c = np.round((y_c * color_intrin['fy'] / z_c) + color_intrin['cy']).astype(int)

    # 5. Filter bounds to ensure we don't map outside the color image dimensions
    color_h, color_w = color_image.shape[:2]
    valid_color = (u_c >= 0) & (u_c < color_w) & (v_c >= 0) & (v_c < color_h)

    # 6. Assign depth values to the new aligned coordinates in the color frame
    # Note: We reverse the depth scaling back to 16-bit integers
    aligned_depth[v_c[valid_color], u_c[valid_color]] = (z_c[valid_color] * 1000.0).astype(depth_image.dtype)

    return aligned_depth