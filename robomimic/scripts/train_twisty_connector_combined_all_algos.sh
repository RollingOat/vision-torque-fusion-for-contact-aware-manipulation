#!/bin/bash
# Train all simvector algorithms for the twisty_connector task
# using the combined dataset.
set +e  # continue to next command even if a command exits with an error

TRAIN_SCRIPT=~/robomimic_suite/robomimic/robomimic/scripts/train_real_robot.py
CONFIG_ROOT=~/robomimic_suite/robomimic/robomimic/exps/templates/twisty_connector

echo "========== [twisty_connector] vision_torque_composition_simplevec =========="
python $TRAIN_SCRIPT --config $CONFIG_ROOT/diffusion_policy_vision_torque_composition_simplevec.json --resume

echo "========== [twisty_connector] vision_torque_gating_simplevec =========="
python $TRAIN_SCRIPT --config $CONFIG_ROOT/diffusion_policy_vision_torque_gating_simplevec.json --resume

echo "========== [twisty_connector] vision_torque_gated_composition_learned_scale_simplevec =========="
python $TRAIN_SCRIPT --config $CONFIG_ROOT/diffusion_policy_vision_torque_gated_composition_learned_scale_simplevec.json --resume

echo "========== [twisty_connector] vision_torque_composition_simplevec =========="
python $TRAIN_SCRIPT --config $CONFIG_ROOT/diffusion_policy_vision_torque_composition.json

echo "========== [twisty_connector] vision_only_simplevec =========="
python $TRAIN_SCRIPT --config $CONFIG_ROOT/diffusion_policy_vision_only_simplevec.json --resume 

echo "========== [twisty_connector] vision_torque_simplevec =========="
python $TRAIN_SCRIPT --config $CONFIG_ROOT/diffusion_policy_vision_torque_simplevec.json --resume

echo "========== [twisty_connector] vision_torque_aux_simplevec =========="
python $TRAIN_SCRIPT --config $CONFIG_ROOT/diffusion_policy_vision_torque_aux_simplevec.json --resume



echo "========== All training runs complete =========="
