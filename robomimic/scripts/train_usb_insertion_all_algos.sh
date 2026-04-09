#!/bin/bash
# Train all algorithms for the usb_insertion task.
set +e  # continue to next command even if a command exits with an error

TRAIN_SCRIPT=~/robomimic_suite/robomimic/robomimic/scripts/train_real_robot.py
CONFIG_ROOT=~/robomimic_suite/robomimic/robomimic/exps/templates/usb_insertion

echo "========== [usb_insertion] vision_only_simplevec =========="
python $TRAIN_SCRIPT --config $CONFIG_ROOT/diffusion_policy_vision_only_simplevec.json --resume

echo "========== [usb_insertion] vision_torque_simplevec =========="
python $TRAIN_SCRIPT --config $CONFIG_ROOT/diffusion_policy_vision_torque_simplevec.json --resume

echo "========== [usb_insertion] vision_torque_aux_simplevec =========="
python $TRAIN_SCRIPT --config $CONFIG_ROOT/diffusion_policy_vision_torque_aux_simplevec.json --resume

echo "========== [usb_insertion] vision_torque_composition_simplevec =========="
python $TRAIN_SCRIPT --config $CONFIG_ROOT/diffusion_policy_vision_torque_composition_simplevec.json --resume

echo "========== [usb_insertion] vision_torque_gated_composition_learned_scale_simplevec =========="
python $TRAIN_SCRIPT --config $CONFIG_ROOT/diffusion_policy_vision_torque_gated_composition_learned_scale_simplevec.json --resume

echo "========== [usb_insertion] vision_torque_gated_weighted_composition_simplevec =========="
python $TRAIN_SCRIPT --config $CONFIG_ROOT/diffusion_policy_vision_torque_gated_weighted_composition_simplevec.json --resume

echo "========== [usb_insertion] vision_torque_gating_simplevec =========="
python $TRAIN_SCRIPT --config $CONFIG_ROOT/diffusion_policy_vision_torque_gating_simplevec.json --resume

echo "========== All training runs complete =========="
