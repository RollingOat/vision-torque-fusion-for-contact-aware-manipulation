"""
Script for running a trained policy on a real robot.

This script loads a trained policy checkpoint and runs it to generate actions
based on real-time observations. Unlike run_trained_agent.py, this script is
designed for real robot deployment and excludes all simulation-specific code.

Args:
    agent (str): path to saved checkpoint pth file

    horizon (int): maximum number of steps for policy execution

    robot_ip (str): IP address of the UR5e robot

    save_obs (bool): if True, save observations (images) during rollouts

    obs_save_dir (str): directory to save observations

    seed (int): if provided, set seed for policy

Example usage:

    python run_policy_real_robot.py --agent /path/to/model.pth --robot_ip 10.125.145.89 --horizon 400
"""
import argparse
import json
import numpy as np
import os
import time
from copy import deepcopy
from collections import OrderedDict, deque
import cv2
import matplotlib.pyplot as plt

import torch

import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.algo import RolloutPolicy

# Import UR5e robot environment
import sys
sys.path.append(os.path.expanduser("~/robomimic_suite/ur_robot_control"))
from ur5e_control.ur5e_robot import UR5eRobotEnv
from frankar3_control.franka_r3_env import FR3RobotEnv

def plot_force_torque_data(save_dir, forces, torques):
    """
    Plot force and torque data over time and save to files
    
    Args:
        save_dir (str): Directory to save plots
        forces (list): List of force measurements (N, 3) [Fx, Fy, Fz]
        torques (list): List of torque measurements (N, 3) [Mx, My, Mz]
    """
    # Convert to numpy arrays
    forces_array = np.array(forces) if len(forces) > 0 else None
    torques_array = np.array(torques) if len(torques) > 0 else None
    
    # Create time axis (assuming 10Hz control frequency)
    if forces_array is not None:
        time_steps = np.arange(len(forces_array)) * 0.1  # 0.1 seconds per step
    elif torques_array is not None:
        time_steps = np.arange(len(torques_array)) * 0.1
    else:
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot forces
    if forces_array is not None:
        ax = axes[0]
        ax.plot(time_steps, forces_array[:, 0], label='Fx', linewidth=2)
        ax.plot(time_steps, forces_array[:, 1], label='Fy', linewidth=2)
        ax.plot(time_steps, forces_array[:, 2], label='Fz', linewidth=2)
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Force (N)', fontsize=12)
        ax.set_title('TCP Forces over Time', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, 'No Force Data', ha='center', va='center', fontsize=14)
        axes[0].set_title('TCP Forces over Time', fontsize=14, fontweight='bold')
    
    # Plot torques
    if torques_array is not None:
        ax = axes[1]
        ax.plot(time_steps, torques_array[:, 0], label='Mx', linewidth=2)
        ax.plot(time_steps, torques_array[:, 1], label='My', linewidth=2)
        ax.plot(time_steps, torques_array[:, 2], label='Mz', linewidth=2)
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Torque (Nm)', fontsize=12)
        ax.set_title('TCP Torques over Time', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'No Torque Data', ha='center', va='center', fontsize=14)
        axes[1].set_title('TCP Torques over Time', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(save_dir, "force_torque_plot.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved force/torque plot to {plot_path}")
    
    # Also create individual plots for better detail
    if forces_array is not None:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(time_steps, forces_array[:, 0], label='Fx', linewidth=2)
        ax.plot(time_steps, forces_array[:, 1], label='Fy', linewidth=2)
        ax.plot(time_steps, forces_array[:, 2], label='Fz', linewidth=2)
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Force (N)', fontsize=12)
        ax.set_title('TCP Forces over Time', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        force_plot_path = os.path.join(save_dir, "forces_plot.png")
        plt.savefig(force_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    if torques_array is not None:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(time_steps, torques_array[:, 0], label='Mx', linewidth=2)
        ax.plot(time_steps, torques_array[:, 1], label='My', linewidth=2)
        ax.plot(time_steps, torques_array[:, 2], label='Mz', linewidth=2)
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Torque (Nm)', fontsize=12)
        ax.set_title('TCP Torques over Time', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        torque_plot_path = os.path.join(save_dir, "torques_plot.png")
        plt.savefig(torque_plot_path, dpi=150, bbox_inches='tight')
        plt.close()


def denormalize_gripper_action(action, gripper_min=3, gripper_max=233):
    """
    Denormalize gripper action from [-1, 1] to [gripper_min, gripper_max].
    
    Args:
        action (np.ndarray): action array with gripper command in last dimension
        gripper_min (float): minimum gripper value
        gripper_max (float): maximum gripper value

    Returns:
        np.ndarray: action array with denormalized gripper command
    """
    denormalized_action = action.copy()
    # Assuming gripper command is the last element in the action array
    gripper_command = action[-1]
    # Denormalize from [-1, 1] to [gripper_min, gripper_max]
    denorm_gripper = ((gripper_command + 1) / 2) * (gripper_max - gripper_min) + gripper_min
    denormalized_action[-1] = denorm_gripper
    return denormalized_action

def denormalize_action(action):
    action_min = np.array([
    -0.01486160549426484,
    -0.019840912274050493,
    -0.01407155138189592,
    -0.0466539808736792,
    -0.033694738984652424,
    -0.03931859683804159,
    3.0
    ])
    action_max = np.array([
    0.011614200462954874,
    0.022786854745120544,
    0.028229215475976677,
    0.05877675446899153,
    0.05329308586177526,
    0.041267414984663256,
    233.0
    ])
    denormalized_action = ((action + 1) / 2) * (action_max - action_min) + action_min
    return denormalized_action


def rollout(policy, env, horizon, save_obs=False, obs_save_dir=None, rollout_idx=0, num_frames=2):
    """
    Execute one rollout with the policy on the real robot.
    
    Args:
        policy (RolloutPolicy): loaded policy from checkpoint
        env (UR5eRobotEnv): robot environment
        horizon (int): maximum number of steps
        save_obs (bool): if True, save observations during rollout
        obs_save_dir (str): directory to save observations
        rollout_idx (int): rollout index for naming saved files
        num_frames (int): number of observation frames to stack (for diffusion policy)
    
    Returns:
        dict: statistics for the rollout (actions taken, etc.)
    """
    # Start new episode
    policy.start_episode()
    
    # Get initial observation
    obs = env.reset()
    
    
    actions = []
    
    # Initialize video writer for combined view if saving observations
    combined_video_writer = None
    if save_obs and obs_save_dir is not None:
        rollout_dir = os.path.join(obs_save_dir, f"rollout_{rollout_idx:03d}")
        os.makedirs(rollout_dir, exist_ok=True)
        
        # Get image dimensions from observations
        # Combine agent_view and eye_in_hand_view side by side
        agent_key = "agentview_image"
        eye_key = "robot0_eye_in_hand_image"
        
        if agent_key in obs and eye_key in obs:
            h1, w1 = obs[agent_key].shape[:2]
            h2, w2 = obs[eye_key].shape[:2]
            
            # Make both images same height for side-by-side combination
            combined_height = max(h1, h2)
            combined_width = w1 + w2
            
            combined_video_path = os.path.join(rollout_dir, "combined_views.mp4")
            combined_video_writer = cv2.VideoWriter(
                combined_video_path, 
                cv2.VideoWriter_fourcc(*'mp4v'), 
                10,  # fps
                (combined_width, combined_height)
            )
            print(f"\nRecording combined video to {rollout_dir}")
            print(f"Combined frame size: {combined_width}x{combined_height}")
        else:
            print(f"\nWarning: Could not find both camera views. Available keys: {list(obs.keys())}")
    
    for step in range(horizon):
        print(f"Step {step + 1}/{horizon}")

        # # Combine both camera views side-by-side for display and recording
        # agent_key = "agentview_image"
        
        # if agent_key in obs:
        #     # Get current frames
        #     agent_frame = obs[agent_key]
            
        #     # Add labels to frames
        #     agent_labeled = agent_frame.copy()
        #     cv2.putText(agent_labeled, "Agent View", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
        #     # Show combined view
        #     cv2.imshow("Combined Camera Views", agent_labeled)
        #     cv2.waitKey(1)
            
            # # Write combined frame to video
            # if save_obs and combined_video_writer is not None:
            #     # Convert RGB to BGR for video writing
            #     combined_bgr = cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR)
            #     combined_video_writer.write(combined_bgr)
        
        # preprocess operation is done inside policy
        # Get action from policy, action denormalization is done inside policy
        action = policy(ob=obs)
        print(f"Action: {action}")
        
        # Execute action on robot and get next observation
        next_obs = env.step(action)
        
        # Store data
        actions.append(action)
        
        # Update observation
        obs = next_obs
    
    # Close video writer and save all data
    if save_obs and obs_save_dir is not None:
        if combined_video_writer is not None:
            combined_video_writer.release()
        
        rollout_dir = os.path.join(obs_save_dir, f"rollout_{rollout_idx:03d}")
        
        # Save actions
        actions_path = os.path.join(rollout_dir, "actions.npy")
        np.save(actions_path, np.array(actions))
        
        print(f"Saved combined video, actions, forces, and torques to {rollout_dir}")
    
    stats = {
        "horizon": len(actions),
        "num_actions": len(actions),
    }
    
    return stats


def run_policy_real_robot(args):
    """
    Run a trained policy for real robot control.
    
    Args:
        args: parsed command line arguments
    """
    # Load checkpoint
    ckpt_path = args.agent
    
    # Device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    
    # Restore policy from checkpoint
    print(f"\nLoading policy from: {ckpt_path}")
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=True)

    # print the key in the checkpoint dict
    print(f"Checkpoint keys: {list(ckpt_dict.keys())}")
    # Get config for horizon if not specified
    rollout_horizon = args.horizon
    config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
    if rollout_horizon is None:
        rollout_horizon = config.experiment.rollout.horizon
        print(f"Using horizon from checkpoint config: {rollout_horizon}")
    
    # Maybe set seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        print(f"Set random seed to: {args.seed}")

    # Get image shapes from checkpoint
    shape_meta = ckpt_dict.get("shape_metadata")
    if shape_meta is not None:
        # shape_meta["all_shapes"] is a dict mapping obs keys to their shapes
        # For image observations, shape is (C, H, W) after processing
        print("\nObservation shapes from checkpoint:")
        for obs_key, shape in shape_meta["all_shapes"].items():
            print(f"  {obs_key}: {shape}")
        
        # Extract image resolution from first image observation
        # Look for common image observation keys
        image_keys = ["agentview_image", "robot0_eye_in_hand_image", "image"]
        rgb_resolution = None
        for key in image_keys:
            if key in shape_meta["all_shapes"]:
                # Shape is (C, H, W), so extract (W, H) for cv2
                shape = shape_meta["all_shapes"][key]
                rgb_resolution = (shape[2], shape[1])  # (width, height)
                break
        
        if rgb_resolution is None:
            # Default fallback
            rgb_resolution = (224, 224)
            print(f"\nWarning: No image observations found in checkpoint, using default resolution {rgb_resolution}")
        else:
            print(f"\nUsing image resolution from checkpoint: {rgb_resolution}")
    else:
        rgb_resolution = (224, 224)
        print(f"\nWarning: No shape metadata in checkpoint, using default resolution {rgb_resolution}")
    
    # Initialize robot environment
    print(f"\nInitializing robot environment at {args.robot_ip}...")
    env = FR3RobotEnv(robot_ip=args.robot_ip, ckpt_dict=ckpt_dict, config=config)
    
    print("\n" + "="*50)
    print("Policy loaded and robot connected successfully")
    print("="*50)
    print(f"Horizon: {rollout_horizon}")
    print(f"Save observations: {args.save_obs}")
    if args.save_obs:
        print(f"Observation save directory: {args.obs_save_dir}")
    print("="*50 + "\n")
    
    # Create save directory if needed
    if args.save_obs:
        os.makedirs(args.obs_save_dir, exist_ok=True)
    
    # Execute rollouts with user confirmation
    
    all_stats = []
    rollout_count = 0
    
    while True:
        # Ask user if ready to start rollout
        if rollout_count == 0:
            response = input("\nReady to start rollout? (y/n): ").strip().lower()
        else:
            response = input("\nDo you want to perform another rollout? (y/n): ").strip().lower()
        
        if response != 'y':
            print("Rollout cancelled by user.")
            break
        
        print(f"\n{'='*50}")
        print(f"Starting rollout {rollout_count + 1}")
        print(f"{'='*50}")
        
        stats = rollout(
            policy=policy,
            env=env,
            horizon=rollout_horizon,
            save_obs=args.save_obs,
            obs_save_dir=args.obs_save_dir,
            rollout_idx=rollout_count
        )
        
        all_stats.append(stats)
        rollout_count += 1
        print(f"\nRollout {rollout_count} completed: {stats}")
    
    # Print summary statistics if any rollouts were performed
    if len(all_stats) > 0:
        print("\n" + "="*50)
        print(f"Completed {len(all_stats)} rollout(s)")
        print("="*50)
        avg_horizon = np.mean([s["horizon"] for s in all_stats])
        print(f"Average horizon: {avg_horizon:.2f}")
        print("="*50 + "\n")
    else:
        print("\nNo rollouts performed.\n")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Path to trained model
    parser.add_argument(
        "--agent",
        type=str,
        default="/home/jiuzl/robomimic_suite/logs/diffusion_policy_franka/vision_only/20260210150732/last.pth",
        help="path to saved checkpoint pth file",
    )
    
    # Robot IP address
    parser.add_argument(
        "--robot_ip",
        type=str,
        default="192.168.2.12",
        help="IP address of the robot",
    )
    
    # Maximum horizon of rollout
    parser.add_argument(
        "--horizon",
        type=int,
        default=200,
        help="(optional) maximum number of steps for policy execution, defaults to value in checkpoint config",
    )
    
    # Save observations
    parser.add_argument(
        "--save_obs",
        action="store_true",
        help="save observations (images) during rollouts",
    )
    
    # Directory to save observations
    parser.add_argument(
        "--obs_save_dir",
        type=str,
        default="./rollout_observations",
        help="directory to save observations",
    )
    
    # Seed for reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="(optional) set seed for policy",
    )
    
    args = parser.parse_args()
    
    # Run policy on real robot
    run_policy_real_robot(args)
