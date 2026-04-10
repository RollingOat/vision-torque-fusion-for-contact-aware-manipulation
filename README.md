# Adaptive Vision-Torque Fusion for Contact-Aware Manipulation

learning contact-aware robot manipulation policies from demonstration for adaptive vision-torque fusion. This project is implemented based on [robomimic](https://github.com/ARISE-Initiative/robomimic) with force/torque sensing integration, multi-modal fusion strategies.


## Installation 

```bash
git clone https://github.com/RollingOat/vision-torque-fusion-for-contact-aware-manipulation.git
cd vision-torque-fusion-for-contact-aware-manipulation
# Create conda environment
conda create -n adaptive_vision_torque_fusion python=3.10
conda activate adaptive_vision_torque_fusion

# Install the package
pip install -e .
```

## Quick Start

### Training

**Train a diffusion policy:**

```bash
python robomimic/scripts/train_real_robot.py \
    --config path/to/config.json
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


## Acknowledgements

This project is built on top of [robomimic](https://github.com/ARISE-Initiative/robomimic), a modular framework for robot learning from demonstration developed by the [ARISE Initiative](https://arise-initiative.org/) at Stanford University. We thank the robomimic team for providing a well-structured and extensible codebase.
