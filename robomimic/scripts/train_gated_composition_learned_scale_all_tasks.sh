#!/bin/bash
# Train diffusion_policy_gated_composition with learned guidance scale for all four tasks.
# Uses scale_predictor MLP (softplus output) to replace the fixed max_guidance_scale:
#     w_force = phi * sigmoid(weight_logit) * softplus(scale_logit)
set +e  # continue to next command even if a command exits with an error

TRAIN_SCRIPT=~/robomimic_suite/robomimic/robomimic/scripts/train_real_robot.py
CONFIG_ROOT=~/robomimic_suite/robomimic/robomimic/exps/templates
CONFIG=diffusion_policy_vision_torque_gated_composition_learned_scale_simplevec.json

# ============================================================
# Task: egg_boiler
# ============================================================
echo "========== [egg_boiler] gated_composition_learned_scale =========="
python $TRAIN_SCRIPT --config $CONFIG_ROOT/egg_boiler/$CONFIG

# ============================================================
# Task: water_bottle
# ============================================================
echo "========== [water_bottle] gated_composition_learned_scale =========="
python $TRAIN_SCRIPT --config $CONFIG_ROOT/water_bottle/$CONFIG

# ============================================================
# Task: twisty_connector
# ============================================================
echo "========== [twisty_connector] gated_composition_learned_scale =========="
python $TRAIN_SCRIPT --config $CONFIG_ROOT/twisty_connector/$CONFIG

# ============================================================
# Task: usb_insertion
# ============================================================
echo "========== [usb_insertion] gated_composition_learned_scale =========="
python $TRAIN_SCRIPT --config $CONFIG_ROOT/usb_insertion/$CONFIG

echo "========== All training runs complete =========="
