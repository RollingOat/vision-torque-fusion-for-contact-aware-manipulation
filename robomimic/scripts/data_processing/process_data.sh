#!/bin/bash

# ==========================================
# USER CONFIGURATION SECTION
# ==========================================

# Base workspace directory (used to derive script and default config paths)
BASE_DIR="/home/jiuzl/robomimic_suite"

# 1. Input folder — the task name is taken from the folder's basename.
#    Expected files inside the folder:
#      synced_image_data.hdf5
#      low_dim_data.hdf5
INPUT_DIR="/media/jiuzl/PortableSSD/twisty_connnector/Mar-1"  # Example: "/home/jiuzl/Data/pick_the_banana"

# 2. Output directory for generated files
OUTPUT_DIR="${BASE_DIR}/datasets"

# 3. Optional Parameters
# Space-separated list of zero-based demo indices to exclude (e.g., "0 5 10")
# Leave empty string "" if no demos should be excluded
EXCLUDE_DEMO_INDEX="83 109 110"
IMAGE_TIME_REFERENCE="timestamp"  # Key in the image dataset to use as the reference timestamp for synchronization
# Force/torque sync mode: "single_raw_ft" (nearest single sample) or "mean_raw_ft" (mean of past N samples)
FT_MODE="single_raw_ft"
# Number of raw F/T samples to average when FT_MODE=mean_raw_ft (30 samples @ 1 kHz ≈ 30 ms)
FT_MEAN_WINDOW=30
# Skip N frames between every kept frame (0 = keep all, 2 = keep 1 skip 2 → step of 3)
SKIP_N=2

# 4. Point-cloud processing (only used when GENERATE_POINTCLOUD=true)
# Set to "true" to compute and store agentview_pointcloud in the output HDF5
GENERATE_POINTCLOUD=false
# Path to the RealSense calibration JSON (required when GENERATE_POINTCLOUD=true)
CALIB_JSON="${BASE_DIR}/robomimic/robomimic/scripts/data_processing/realsense_calib_405622072966.json"
# Path to pointcloud processing config JSON; controls depth_trunc, voxel_size,
# n_points, use_fps, center, and radius.  Leave empty "" to use hardcoded defaults.
PC_CONFIG="${BASE_DIR}/robomimic/robomimic/scripts/data_processing/pointcloud_config_water_bottle.json"
# ==========================================
# END OF CONFIGURATION
# ==========================================

# Derive task name and fixed-pattern dataset paths from the input folder
TASK="$(basename "$INPUT_DIR")"
IMAGE_DATASET="${INPUT_DIR}/synced_image_data.hdf5"
LOWDIM_DATASET="${INPUT_DIR}/low_dim_data.hdf5"

# Output filenames
SYNC_OUTPUT="${OUTPUT_DIR}/synced_data_${TASK}.hdf5"
ROBOMIMIC_OUTPUT="${OUTPUT_DIR}/robomimic_data_${TASK}.hdf5"

# Script Paths (based on workspace structure)
SYNC_SCRIPT="${BASE_DIR}/robomimic/robomimic/scripts/data_processing/sync_image_low_dim.py"
CONVERT_SCRIPT="${BASE_DIR}/robomimic/robomimic/scripts/data_processing/synced_data_to_robotmimic.py"
VISUALIZE_SCRIPT="${BASE_DIR}/robomimic/robomimic/scripts/data_processing/visualize_synced_data.py"

# Default visualization output directory
VIS_OUT_DIR="${BASE_DIR}/temp"

# Ensure output directory exists
mkdir -p "${OUTPUT_DIR}"

# Detect python executable (prefer python3)
if command -v python3 >/dev/null 2>&1; then
    PY_CMD=python3
elif command -v python >/dev/null 2>&1; then
    PY_CMD=python
else
    echo "Error: Python is not installed or not in PATH. Install python3 or add python to PATH."
    exit 1
fi

# Function to check if file exists
check_file_exists() {
    if [ ! -f "$1" ]; then
        echo "Error: File '$1' not found. Please check the path in the configuration section."
        exit 1
    fi
}

echo "Starting data processing pipeline..."

# Check inputs
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input folder '$INPUT_DIR' not found. Please set INPUT_DIR in the configuration section."
    exit 1
fi

check_file_exists "$IMAGE_DATASET"
check_file_exists "$LOWDIM_DATASET"

# 1. Run sync_image_low_dim
echo "------------------------------------------------"
echo "Step 1: Running sync_image_low_dim..."
echo "------------------------------------------------"

# Build arguments for sync_image_low_dim
SYNC_ARGS="--image-h5 \"$IMAGE_DATASET\" --lowdim-h5 \"$LOWDIM_DATASET\" --output-h5 \"$SYNC_OUTPUT\""

if [ -n "$EXCLUDE_DEMO_INDEX" ]; then
    SYNC_ARGS="$SYNC_ARGS --exclude-demo-index $EXCLUDE_DEMO_INDEX --allow-missing"
fi

SYNC_ARGS="$SYNC_ARGS --ft-mode $FT_MODE --ft-mean-window $FT_MEAN_WINDOW"

if [ "$SKIP_N" -gt 0 ] 2>/dev/null; then
    SYNC_ARGS="$SYNC_ARGS --skip-n $SKIP_N"
fi

if [ "$GENERATE_POINTCLOUD" = "true" ]; then
    SYNC_ARGS="$SYNC_ARGS --pointcloud --calib \"$CALIB_JSON\""
    if [ -n "$PC_CONFIG" ]; then
        SYNC_ARGS="$SYNC_ARGS --pc-config \"$PC_CONFIG\""
    fi
fi

# We use eval to handle the variable expansion properly especially if arguments have quotes or spaces
eval "$PY_CMD \"$SYNC_SCRIPT\" $SYNC_ARGS"

# Check step 1 success
if [ $? -ne 0 ]; then
    echo "Error: sync_image_low_dim failed."
    exit 1
fi
echo "Step 1 complete. Output saved to: $SYNC_OUTPUT"


# 2. Run synced_data_to_robomimic
echo "------------------------------------------------"
echo "Step 2: Converting to Robomimic format..."
echo "------------------------------------------------"
"$PY_CMD" "$CONVERT_SCRIPT" --input "$SYNC_OUTPUT" --out "$ROBOMIMIC_OUTPUT"

# Check step 2 success
if [ $? -ne 0 ]; then
    echo "Error: synced_data_to_robomimic failed."
    exit 1
fi
echo "Step 2 complete. Output saved to: $ROBOMIMIC_OUTPUT"

rm "$SYNC_OUTPUT"