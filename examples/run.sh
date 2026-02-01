#!/bin/bash
#SBATCH --job-name=realdrivesim-sample
#SBATCH --output=realdrivesim-sample_%j.out
#SBATCH --error=realdrivesim-sample_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00

# -------------------------------------------------------------------- #
# Dataset paths  (edit these to match your setup)                       #
# -------------------------------------------------------------------- #
RGB_PATH="/cluster/scratch/${USER}/realdrivesim/rgb.zip"
DEPTH_PATH="/cluster/scratch/${USER}/realdrivesim/depth.zip"
SEGMENTATION_PATH="/cluster/scratch/${USER}/realdrivesim/segmentation.zip"
CALIBRATION_PATH="/cluster/scratch/${USER}/realdrivesim/calibration.zip"

# -------------------------------------------------------------------- #
# Environment setup                                                     #
# -------------------------------------------------------------------- #
module load stack/2024-06 python/3.11.6
source "${HOME}/venvs/ds-crawler/bin/activate"   # adjust to your venv

# -------------------------------------------------------------------- #
# Run                                                                   #
# -------------------------------------------------------------------- #
python "$(dirname "$0")/sample_realdrivesim.py" \
    --rgb_path           "${RGB_PATH}" \
    --depth_path         "${DEPTH_PATH}" \
    --segmentation_path  "${SEGMENTATION_PATH}" \
    --calibration_path   "${CALIBRATION_PATH}"
