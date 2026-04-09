#!/usr/bin/env bash

# fail_data_process.sh
#
# Configure these variables below (no CLI args required):

set -euo pipefail

# ------------------------
# USER CONFIGURATION
# ------------------------
# Input folder that contains:
#   synced_image_data.hdf5
#   low_dim_data.hdf5
INPUT_FOLDER="/home/jiuzl/Data/10_tasks"

# Output folder where the generated file will be saved
OUTPUT_FOLDER="/home/jiuzl/robomimic_suite/datasets"

# Image / low-dim keys (space-separated)
IMAGE_KEYS="agentview_image"
LOWDIM_KEYS="robot0_eef_pos robot0_eef_quat robot0_gripper_qpos"

# Resize parameters (set both to non-empty to enable resizing)
RESIZE_H="180"
RESIZE_W="320"

# Skip N frames after saving one (integer >= 0)
SKIP_N=2

# If true, allow demos with missing keys to be skipped
ALLOW_MISSING="--allow-missing"

# ------------------------
# End config
# ------------------------

if [ ! -d "$INPUT_FOLDER" ]; then
    echo "Input folder does not exist: $INPUT_FOLDER" >&2
    exit 1
fi

if [ ! -d "$OUTPUT_FOLDER" ]; then
    echo "Output folder does not exist, creating: $OUTPUT_FOLDER"
    mkdir -p "$OUTPUT_FOLDER"
fi

IMAGE_H5="$INPUT_FOLDER/realsense_data.hdf5"
LOWDIM_H5="$INPUT_FOLDER/low_dim_data.hdf5"

if [ ! -f "$IMAGE_H5" ]; then
    echo "Image HDF5 not found: $IMAGE_H5" >&2
    exit 1
fi
if [ ! -f "$LOWDIM_H5" ]; then
    echo "Low-dim HDF5 not found: $LOWDIM_H5" >&2
    exit 1
fi

TASK_NAME="$(basename "$INPUT_FOLDER")"
OUTPUT_H5="$OUTPUT_FOLDER/${TASK_NAME}_synced_failure.hdf5"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PY_SCRIPT="$SCRIPT_DIR/sync_data_for_failure_learning.py"

echo "Task: $TASK_NAME"
echo "Image dataset: $IMAGE_H5"
echo "Low-dim dataset: $LOWDIM_H5"
echo "Output: $OUTPUT_H5"
echo "Image keys: $IMAGE_KEYS"
echo "Low-dim keys: $LOWDIM_KEYS"
if [ -n "$RESIZE_H" ] && [ -n "$RESIZE_W" ]; then
    echo "Resize: ${RESIZE_H}x${RESIZE_W}"
else
    echo "Resize: (none)"
fi
echo "Skip N: $SKIP_N"

CMD=(python "$PY_SCRIPT" \
    --image-h5 "$IMAGE_H5" \
    --lowdim-h5 "$LOWDIM_H5" \
    --output-h5 "$OUTPUT_H5" \
    --image-keys $IMAGE_KEYS \
    --lowdim-keys $LOWDIM_KEYS \
    --skip_n "$SKIP_N" \
    $ALLOW_MISSING)

if [ -n "$RESIZE_H" ] && [ -n "$RESIZE_W" ]; then
    CMD+=(--resize_h "$RESIZE_H" --resize_w "$RESIZE_W")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"

if [ $? -eq 0 ]; then
    echo "Synchronization completed successfully: $OUTPUT_H5"
else
    echo "Synchronization failed" >&2
    exit 1
fi
