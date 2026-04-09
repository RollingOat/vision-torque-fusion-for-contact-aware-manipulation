# Adaptive Vision-Torque Fusion for Contact-Aware Manipulation

learning contact-aware robot manipulation policies from demonstration for adaptive vision-torque fusion. This project is implemented based on [robomimic](https://github.com/ARISE-Initiative/robomimic) with force/torque sensing integration, multi-modal fusion strategies.

## Features

- **Adaptive vision-torque fusion** — learn when to rely on visual input vs. force/torque feedback for contact-rich tasks
- **Multiple fusion strategies** — composition, gating, and auxiliary-task-based approaches for combining vision and force modalities
- **Real robot deployment** — end-to-end pipeline from data collection to policy deployment on UR5e and Franka R3 robots
- **Diffusion policy backbone** — built on top of diffusion policy for expressive multi-modal action distributions
- **Rich data processing tools** — utilities for syncing multi-modal data, weight compensation, depth-color alignment, and point cloud processing

## Installation (AI generated not verified yet)

```bash
git clone https://github.com/<your-org>/robomimic-suite.git
cd robomimic-suite/robomimic

# Create conda environment
conda create -n robomimic python=3.10
conda activate robomimic

# Install the package
pip install -e .
```

## Quick Start

### Training

**Train a diffusion policy with vision + force/torque on UR5e data:**

```bash
python robomimic/scripts/train_real_robot.py \
    --config robomimic/exps/templates/diffusion_policy_vision_force_ur5e.json \
    --dataset /path/to/synced_data_10hz.hdf5 \
    --name ur5e_vision_force_experiment
```

**Train a vision-only policy on Franka R3 data:**

```bash
python robomimic/scripts/train_real_robot.py \
    --config robomimic/exps/templates/diffusion_policy_vision_only_franka.json \
    --dataset /path/to/franka_demos.hdf5 \
    --name franka_vision_only
```

**Train with fusion strategies:**

```bash
# Gating-based fusion (learned gate between vision and force)
python robomimic/scripts/train_real_robot.py \
    --config robomimic/exps/templates/diffusion_policy_vision_force_gating.json \
    --dataset /path/to/dataset.hdf5

# Composition-based fusion
python robomimic/scripts/train_real_robot.py \
    --config robomimic/exps/templates/diffusion_policy_vision_force_composition.json \
    --dataset /path/to/dataset.hdf5
```

**Quick debug run (small number of epochs/steps):**

```bash
python robomimic/scripts/train_real_robot.py \
    --config robomimic/exps/templates/diffusion_policy_vision_force_ur5e.json \
    --dataset /path/to/dataset.hdf5 \
    --debug
```

Training outputs are saved to:

```
<experiment_name>/<timestamp>/
├── config.json           # Full config snapshot
├── logs/
│   ├── log.txt           # Terminal output
│   └── tb/               # TensorBoard logs
├── models/
│   ├── last.pth          # Latest checkpoint
│   └── best_rollout_success_rate.pth
└── videos/               # Rollout videos (if enabled)
```

Monitor training with TensorBoard:

```bash
tensorboard --logdir <experiment_name>/<timestamp>/logs/tb
```

### Real Robot Inference

**Deploy a trained policy on a UR5e robot:**

```bash
python robomimic/scripts/run_policy_real_robot.py \
    --agent /path/to/best_rollout_success_rate.pth \
    --robot_ip 10.125.145.89 \
    --horizon 400
```

**Deploy with observation saving (for debugging and data collection):**

```bash
python robomimic/scripts/run_policy_real_robot.py \
    --agent /path/to/model.pth \
    --robot_ip 10.125.145.89 \
    --horizon 400 \
    --save_obs \
    --obs_save_dir ./rollout_logs
```

### Data Processing

**Convert synchronized sensor data to robomimic HDF5 format:**

```bash
python robomimic/scripts/data_processing/synced_data_to_robotmimic.py \
    --input_dir /path/to/raw_synced_data \
    --output /path/to/dataset.hdf5
```

**Visualize a dataset:**

```bash
python robomimic/scripts/data_processing/visualize_robomimic_dataset.py \
    --dataset /path/to/dataset.hdf5
```

## Config Templates

Pre-configured experiment templates are provided in `robomimic/exps/templates/`:

| Template | Description |
|----------|-------------|
| `diffusion_policy_vision_force_ur5e.json` | Vision + force/torque for UR5e |
| `diffusion_policy_vision_force_fanka.json` | Vision + force/torque for Franka R3 |
| `diffusion_policy_vision_only_franka.json` | Vision-only for Franka R3 |
| `diffusion_policy_vision_force_gating.json` | Gating-based fusion |
| `diffusion_policy_vision_force_composition.json` | Composition-based fusion |
| `diffusion_policy_vision_force_aux.json` | Auxiliary task fusion |
| `diffusion_policy_pointcloud.json` | Point cloud input |

Task-specific configs are available under `egg_boiler/`, `water_bottle/`, `usb_insertion/`, and `twisty_connector/` subdirectories.

## Supported Algorithms

| Algorithm | Description |
|-----------|-------------|
| BC | Behavioral Cloning |
| Diffusion Policy | Denoising diffusion for action generation |
| BCQ | Batch-Constrained Q-Learning |
| CQL | Conservative Q-Learning |
| IQL | Implicit Q-Learning |
| TD3-BC | TD3 + Behavioral Cloning |
| GL | Goal Learning |
| HBC | Hierarchical BC |
| IRIS | Implicit Reinforcement without Interaction at Scale |

## Acknowledgements

This project is built on top of [robomimic](https://github.com/ARISE-Initiative/robomimic), a modular framework for robot learning from demonstration developed by the [ARISE Initiative](https://arise-initiative.org/) at Stanford University. We thank the robomimic team for providing a well-structured and extensible codebase.
