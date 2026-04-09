"""Estimate gripper mass and center of mass using pose sweeps on the Franka Research 3.

This script drives the robot through several orientations (down, up, left, right,
tilted) while holding a fixed position. At each pose it holds for a specified
duration, logs external force/torque at the stiffness frame, and then estimates
the gripper mass and COM using the relation tau = r x F under static gravity.

Assumptions
-----------
- The `franky_control` package exposes a controller with `move_to_pose`,
  `get_ee_pose`, and `get_external_wrench` methods.
- Gravity is aligned with the world +Z axis (downward force in tool frame will
  depend on orientation). Update GRAVITY if needed.
- The robot is collision-free for the small orientation changes around the
  initial position.
"""

from __future__ import annotations

import argparse
import time
import json
import pickle
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from frankar3_control.franka_r3_robot import FR3_ROBOT
from data_processing.utils import estimate_payload_with_bias, compensate_weight, optimize_payload_with_bias


'''
Configuration	Euler or axis interpretation	Quaternion (w,x,y,z)
Down	flip +Z to −Z	(0, 1, 0, 0)
Down +20° Left	tilt left	~ (0.94, 0.34, 0, 0)
Down +20° Right	tilt right	~ (0.94, -0.34, 0, 0)
Down +20° Forward	forward tilt	~ (0.94, 0, 0.34, 0)
Down +20° Back	backward tilt	~ (0.94, 0, -0.34, 0)
Up	neutral	(1, 0, 0, 0)

'''

NEUTRAL = [7.75992220e-01, -1.74063230e-01,  9.16369350e-03, -2.20974650e+00, 1.98854642e-03,  2.03519185e+00,  1.56922985e+00-np.pi/4]
J1 = NEUTRAL.copy() + np.array([0, 0, 0, 0, 0, +20.0/180*np.pi, 0])  # tilt up
J2 = NEUTRAL.copy() + np.array([0, 0, 0, 0, 0, -20.0/180*np.pi, 0])  # tilt down
J3 = NEUTRAL.copy() + np.array([0, 0, 0, 0, 20.0/180*np.pi, 0, 0])  # tilt left
J4 = NEUTRAL.copy() + np.array([0, 0, 0, 0, -20.0/180*np.pi, 0, 0])
J5 = NEUTRAL.copy() + np.array([0, 0, 0, 0, np.pi/2, 0, 0])  # tilt right
J6 = NEUTRAL.copy() + np.array([0, 0, 0, 0, -np.pi/2, 0, 0])
J7 = NEUTRAL.copy() + np.array([0, -20.0/180*np.pi, 0, 0, 0, 0, 0])
J8 = NEUTRAL.copy() + np.array([0, 0, 20.0/180*np.pi, 0, 0, 0, 0])
J9 = NEUTRAL.copy() + np.array([0, 0, -20.0/180*np.pi, 0, 0, 0, 0])
J10 = NEUTRAL.copy() + np.array([0, 20.0/180*np.pi, 0, 0, 0, 0, 0])
J11 = NEUTRAL.copy() + np.array([0, 0, 0, 0, 0, 90.0/180*np.pi, 10/180*np.pi])
joint_configs = {
    "neutral": NEUTRAL,
    "tilt_forward_20": J1,
    "tilt_backward_20": J2,
    "tilt_left_20": J3,
    "tilt_left_90": J5,
    "tilt_right_20": J4,
    "tilt_right_90": J6,
	"J7": J7,
	"J8": J8,
	"J9": J9,
	"J10": J10,
	"J11": J11
}

TEST_J1 = NEUTRAL.copy() + np.array([0, -20.0/180*np.pi, 0, 10/180*np.pi, 0, 0, 0])
TEST_J2 = NEUTRAL.copy() + np.array([0, 0, 20.0/180*np.pi, 0, 0, 10/180*np.pi, 0])
TEST_J3 = NEUTRAL.copy() + np.array([0, 0, -20.0/180*np.pi, 0, 0, 0, 10/180*np.pi])
TEST_J4 = NEUTRAL.copy() + np.array([30.0/180*np.pi, 20.0/180*np.pi, 0, 0, 10/180*np.pi, 0, 0])
test_joint_configs = [TEST_J1, TEST_J2, TEST_J3, TEST_J4]

GRAVITY = 9.81  # m/s^2


@dataclass
class PoseSample:
    name: str
    forces: np.ndarray  # (N, 3)
    torques: np.ndarray  # (N, 3)
    orientation: R      # Rotation from  sensor to base frame
    pose: np.ndarray    # Affine matrix of EE pose in world frame


def split_wrench(wrench: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
	flat = np.asarray(wrench).reshape(-1)
	if flat.size < 6:
		raise ValueError(f"Expected 6D wrench, got shape {flat.shape}")
	return flat[:3], flat[3:]


def record_wrench(robot: FR3_ROBOT, hold_s: float, sample_hz: float) -> Tuple[np.ndarray, np.ndarray]:
	forces: List[np.ndarray] = []
	torques: List[np.ndarray] = []
	period = 1.0 / sample_hz
	end_time = time.time() + hold_s
	while time.time() < end_time:
		wrench = robot.get_end_effector_wrench()
		f, t = split_wrench(wrench)
		forces.append(f)
		torques.append(t)
		time.sleep(period)
	return np.stack(forces, axis=0), np.stack(torques, axis=0)


def pose_to_pos_quat(affine) -> Tuple[np.ndarray, np.ndarray]:
	"""Best-effort extraction of position and quaternion from franky Affine-like object."""
	return affine.translation, affine.quaternion


def move_and_hold(
    robot: FR3_ROBOT,
    joint_pos: np.ndarray,
    hold_s: float,
    sample_hz: float,
    label: str,
) -> PoseSample:
    robot.move_to_joints(joint_pos)
    # Record actual pose after move
    pose = robot.get_end_effector_pose()
    pose_matrix = robot.get_end_effector_pose().matrix
    # Need orientation for gravity vector direction in sensor frame
    _, quat = pose_to_pos_quat(pose)
    orientation = R.from_quat(quat)
    time.sleep(2.0)  # Settle after move
    forces, torques = record_wrench(robot, hold_s, sample_hz)
    return PoseSample(label, forces, torques, orientation, pose_matrix)


def estimate_mass_and_com(samples: List[PoseSample], optimize: bool = True) -> Dict[str, np.ndarray]:
    """Estimate mass and COM using nonlinear optimization with bounds."""
    
    
    		
    if optimize:
        data = []
        for i in range(len(samples)):
            sample = samples[i]
            R_sw = sample.orientation.as_matrix()  # sensor to world rotation
            forces = sample.forces
            torques = sample.torques
            data.append({'R_sw': R_sw, 'force': forces, 'torque': torques})
        init_guess_mass = 0.2
        init_guess_com = np.array([0.0, 0.0, 0.2])
        init_guess_force_bias = np.array([0.0, 0.0, 0.0])
        init_guess_torque_bias = np.array([0.0, 0.0, 0.0])
        init_guess = {
            'mass': init_guess_mass,
            'com': init_guess_com,
            'force_bias': init_guess_force_bias,
            'torque_bias': init_guess_torque_bias}
        result = optimize_payload_with_bias(data, init_guess, g=GRAVITY)
    else:
        data = []
        for i in range(len(samples)):
            sample = samples[i]
            R_sw = sample.orientation.as_matrix()  # sensor to world rotation
            mean_force = np.mean(sample.forces, axis=0)
            mean_torque = np.mean(sample.torques, axis=0)
            data.append({'R_sw': R_sw, 'force': mean_force, 'torque': mean_torque})
        result = estimate_payload_with_bias(data, g=GRAVITY)
    return result


def save_calibration_data(prefix: str, samples: List[PoseSample]):
    """Save calibration results to JSON and raw samples to Pickle."""
    
    
    # Save raw samples (useful for offline re-optimization or debugging)
    pkl_path = f"{prefix}_samples.pkl"
    with open(pkl_path, 'wb') as f:
        pickle.dump(samples, f)
    print(f"Saved raw samples to {pkl_path}")

    # Save raw wrench data to .npz for easier access without Custom Classes
    data_dict = {}
    for sample in samples:
         data_dict[f"{sample.name}_forces"] = sample.forces
         data_dict[f"{sample.name}_torques"] = sample.torques
         data_dict[f"{sample.name}_quat"] = sample.orientation.as_quat()
         data_dict[f"{sample.name}_pose"] = sample.pose

    npz_path = f"{prefix}_wrenches.npz"
    np.savez(npz_path, **data_dict)
    print(f"Saved raw wrench arrays to {npz_path}")




def plot_wrenches(samples: List[PoseSample], output_file: str = "wrench_plot.png"):
    """Plot recorded forces and torques for each pose."""
    num_poses = len(samples)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Create a concatenated timeline just for plotting
    current_idx = 0
    ticks = []
    tick_labels = []
    
    colors = ['r', 'g', 'b']
    labels = ['x', 'y', 'z']
    
    for sample in samples:
        n = len(sample.forces)
        x_indices = np.arange(current_idx, current_idx + n)
        
        # Plot Forces
        for i in range(3):
            axes[0].plot(x_indices, sample.forces[:, i], color=colors[i], label=labels[i] if current_idx == 0 else "")
            
        # Plot Torques
        for i in range(3):
            axes[1].plot(x_indices, sample.torques[:, i], color=colors[i], label=labels[i] if current_idx == 0 else "")
            
        # Separator line
        axes[0].axvline(x=current_idx+n, color='k', linestyle='--', alpha=0.3)
        axes[1].axvline(x=current_idx+n, color='k', linestyle='--', alpha=0.3)
        
        # Label position roughly in middle
        ticks.append(current_idx + n/2)
        tick_labels.append(sample.name)
        
        current_idx += n

    axes[0].set_ylabel("Force (N)")
    axes[0].set_title("External Forces at Stiffness Frame")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_ylabel("Torque (Nm)")
    axes[1].set_title("External Torques at Stiffness Frame")
    axes[1].grid(True, alpha=0.3)
    
    axes[1].set_xticks(ticks)
    axes[1].set_xticklabels(tick_labels, rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    plt.close(fig)


def run_validation_sequence(
    robot: FR3_ROBOT,
    test_poses: List[np.ndarray],
    mass: float,
    com: np.ndarray,
    bias_f: np.ndarray,
    bias_t: np.ndarray,
    hold_s: float,
    sample_hz: float,
    output_file: str = "validation_check.png"
):
    """
    Move robot to test poses, measure wrench, compute compensated wrench, and plot results.
    """
    print(f"\n--- Starting Validation Sequence on {len(test_poses)} poses ---")
    
    samples: List[PoseSample] = []
    
    # Collection Phase
    for i, jp in enumerate(test_poses):
        name = f"Test_{i+1}"
        print(f"Moving to validation pose: {name}")
        sample = move_and_hold(robot, jp, hold_s, sample_hz, label=name)
        samples.append(sample)
        
    save_calibration_data("validation_check", samples)  # Save raw data for offline analysis
    
    # Computation Phase
    residual_forces = []
    residual_torques = []
    names = []
    
    
    for i, pose_forces in enumerate(samples):
         forces = pose_forces.forces
         torques = pose_forces.torques
         pose = pose_forces.pose
         measurements = np.concatenate([forces, torques], axis=1)  # (N, 6)
         external_force_torques = compensate_weight(measurements, mass, com, bias_f, bias_t, pose)  # (N, 6)
         residual_forces.append(external_force_torques[:, :3])
         residual_torques.append(external_force_torques[:, 3:])
         names.extend([pose_forces.name] * len(forces))
        
    residual_forces = np.concatenate(residual_forces, axis=0)
    residual_torques = np.concatenate(residual_torques, axis=0)
    
    # Plotting Phase
    num_samples = len(residual_forces)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    x_indices = np.arange(num_samples)
    colors = ['r', 'g', 'b']
    labels = ['x', 'y', 'z']
    
    # Plot Residual Forces
    for i in range(3):
        axes[0].plot(x_indices, residual_forces[:, i], color=colors[i], label=f"Comp {labels[i]}", alpha=0.7)
    axes[0].set_ylabel("Compensated Force (N)")
    axes[0].set_title(f"Validation: Compensated Force (Expect ~0)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot Residual Torques
    for i in range(3):
        axes[1].plot(x_indices, residual_torques[:, i], color=colors[i], label=f"Comp {labels[i]}", alpha=0.7)
    axes[1].set_ylabel("Compensated Torque (Nm)")
    axes[1].set_title("Validation: Compensated Torque (Expect ~0)")
    axes[1].grid(True, alpha=0.3)
    
    # Mark transitions
    unique_names, idx = np.unique(names, return_index=True)
    sorted_stages = sorted(zip(idx, unique_names))
    
    ticks = []
    tick_labels = []
    for start_idx, name in sorted_stages:
        for ax in axes:
            ax.axvline(x=start_idx, color='k', linestyle=':', alpha=0.3)
        ticks.append(start_idx)
        tick_labels.append(name)
    
    axes[1].set_xticks(ticks)
    axes[1].set_xticklabels(tick_labels, rotation=45, ha='left')
    
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Validation plot saved to {output_file}")
    
    # Statistics
    rmse_f = np.sqrt(np.mean(residual_forces**2))
    rmse_t = np.sqrt(np.mean(residual_torques**2))
    print(f"\n--- Validation Sequence Stats (RMSE) ---")
    print(f"  Force:  {rmse_f:.4f} N")
    print(f"  Torque: {rmse_t:.4f} Nm")





def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Franka gripper weight compensation sweep")
    parser.add_argument("--robot-ip", default="192.168.2.12", help="IP address of the robot controller")
    parser.add_argument("--hold-s", type=float, default=5.0, help="Hold duration per pose (seconds)")
    parser.add_argument("--sample-hz", type=float, default=500.0, help="Wrench sampling rate during hold")
    parser.add_argument("--plot-output", type=str, default="wrench_plot.png", help="Filename for the output plot")
    parser.add_argument("--save-prefix", type=str, default=None, help="Prefix to save calibration data (results in .json, samples in .pkl)")
    parser.add_argument("--saved-sample-path", type=str, default="/home/jiuzl/robomimic_suite/robot_camera_control/data_processing/calib_data_samples.pkl", help="Path to saved samples .pkl to skip robot collection")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples: List[PoseSample] = []
    
    if args.saved_sample_path:
        print(f"Loading saved samples from: {args.saved_sample_path}")
        with open(args.saved_sample_path, 'rb') as f:
            samples = pickle.load(f)
        print(f"Loaded {len(samples)} pose samples from disk.")
    else:
        # Capture current pose and derive base orientation
        robot = FR3_ROBOT(args.robot_ip)
        robot.robot.relative_dynamics_factor = 0.15
        # Neutral configuration for Franka Research 3
        neutral_joints = [7.75992220e-01, -1.74063230e-01,  9.16369350e-03, -2.20974650e+00, 1.98854642e-03,  2.03519185e+00,  1.56922985e+00]
        print(f"Moving to neutral joint configuration: {neutral_joints}")
        robot.move_to_joints(neutral_joints)
        time.sleep(1.0)  # Settle after big move
        pose = robot.get_end_effector_pose()
        curr_pos, curr_quat = pose_to_pos_quat(pose)
        print(f"Current EE Position: {curr_pos}, Quaternion: {curr_quat}")
        base_rot_vector = R.from_quat(curr_quat).as_rotvec()
        print(f"Base rotation vector: {base_rot_vector}")

        for name, jp in joint_configs.items():
            print(f"Moving to pose: {name}")
            sample = move_and_hold(
                robot,
                joint_pos=jp,
                hold_s=args.hold_s,
                sample_hz=args.sample_hz,
                label=name,
            )
            samples.append(sample)
            print(f"  Collected {len(sample.forces)} wrench samples at {name}")
          
	

    # plot_wrenches(samples, output_file=args.plot_output)

    result = estimate_mass_and_com(samples, optimize=False)
    mass_kg = result['mass']
    com_m = result['com']
    bias_f = result['force_bias']
    bias_t = result['torque_bias']
    if args.save_prefix:
        save_calibration_data(args.save_prefix, samples)
    print("\nEstimated gripper parameters:")
    print(f"  Mass: {mass_kg:.3f} kg")
    print(f"  COM (stiffness frame): [{com_m[0]:.4f}, {com_m[1]:.4f}, {com_m[2]:.4f}] m")
    print(f"  Force Bias (N): [{bias_f[0]:.4f}, {bias_f[1]:.4f}, {bias_f[2]:.4f}]")
    print(f"  Torque Bias (Nm): [{bias_t[0]:.4f}, {bias_t[1]:.4f}, {bias_t[2]:.4f}]")
    print("\nSet these values in your controller/load parameters so external wrench at the stiffness frame trends to zero at static.")
    
    
    # if robot is not None:
    #     print("\nGenerating validation plot on TEST POSES...")
    #     run_validation_sequence(
    #         robot,
    #         test_joint_configs,
    #         mass_kg,
    #         com_m,
    #         bias_f,
    #         bias_t,
    #         hold_s=3.0,
    #         sample_hz=args.sample_hz,
    #         output_file="validation_check_" + args.plot_output
    #     )
    # else:
    #     print("Skipping validation sweep because --saved-sample-path was provided.")




if __name__ == "__main__":
    main()
