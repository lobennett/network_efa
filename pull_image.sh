#!/bin/bash

#SBATCH --job-name=pull_fmri_env
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=russpold,hns,normal
#SBATCH --output=log/pull_image-%j.out
#SBATCH --error=log/pull_image-%j.err

# Pull apptainer image for tedana processing
# Can be run standalone or as a SLURM job

set -e  # Exit on any error

# Configuration
USER=lobennett
IMG=fmri_env
TAG=latest
APPTAINER_DIR="$(pwd)/apptainer"
IMAGE_FILE="$APPTAINER_DIR/${IMG}_${TAG}.sif"

# Create apptainer directory if it doesn't exist
mkdir -p "$APPTAINER_DIR"

# Check if image already exists
if [ -f "$IMAGE_FILE" ]; then
    echo "Image already exists: $IMAGE_FILE"
    echo "Remove it if you want to re-download:"
    echo "  rm $IMAGE_FILE"
    exit 0
fi

echo "Pulling apptainer image..."
echo "Source: docker://ghcr.io/$USER/$IMG:$TAG"
echo "Destination: $IMAGE_FILE"

# Pull the image
apptainer pull "$IMAGE_FILE" docker://ghcr.io/$USER/$IMG:$TAG

echo "Successfully pulled image to: $IMAGE_FILE"