#!/bin/bash
# Train simplevec diffusion policy algorithms in sequence for three tasks:
# twisty_connector, usb_insertion, egg_boiler
set +e  # continue to next command even if a command exits with an error

TRAIN_SCRIPT=~/robomimic_suite/robomimic/robomimic/scripts/train_real_robot.py
CONFIG_ROOT=~/robomimic_suite/robomimic/robomimic/exps/templates

# ============================================================
# Task: twisty_connector
# ============================================================
# echo "========== [twisty_connector] vision_only_simplevec =========="
# python $TRAIN_SCRIPT --config $CONFIG_ROOT/twisty_connector/diffusion_policy_vision_only_simplevec.json
echo "========== [twisty_connector] vision_torque_gating_simplevec =========="
python $TRAIN_SCRIPT --config $CONFIG_ROOT/twisty_connector/diffusion_policy_vision_torque_gating_simplevec.json


# ============================================================
# Task: usb_insertion
# ============================================================
echo "========== [usb_insertion] vision_torque_gating_simplevec =========="
python $TRAIN_SCRIPT --config $CONFIG_ROOT/usb_insertion/diffusion_policy_vision_torque_gating_simplevec.json


# ============================================================
# Task: egg_boiler
# ============================================================
echo "========== [egg_boiler] vision_torque_gating_simplevec =========="
python $TRAIN_SCRIPT --config $CONFIG_ROOT/egg_boiler/diffusion_policy_vision_torque_gating_simplevec.json

# ============================================================
# Task: water bottle
# ============================================================
echo "========== [water_bottle] vision_torque_gating_simplevec =========="
python $TRAIN_SCRIPT --config $CONFIG_ROOT/water_bottle/diffusion_policy_vision_torque_gating_simplevec.json


echo "========== All training runs complete =========="
