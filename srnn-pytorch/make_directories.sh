#!/bin/bash

# Create directories for training logs and model saves
# Usage: bash make_directories.sh [leaveDataset]

LEAVE_DATASET=${1:-3}  # Default to 3 if not provided

# Create directories for distributed training
mkdir -p "log/${LEAVE_DATASET}/log_attention"
mkdir -p "save/${LEAVE_DATASET}/save_attention"

# Create directories for single GPU training
mkdir -p "log/${LEAVE_DATASET}/log_attention_single"
mkdir -p "save/${LEAVE_DATASET}/save_attention_single"

echo "Directories created successfully!"
echo "  - log/${LEAVE_DATASET}/log_attention"
echo "  - save/${LEAVE_DATASET}/save_attention"
echo "  - log/${LEAVE_DATASET}/log_attention_single"
echo "  - save/${LEAVE_DATASET}/save_attention_single"
