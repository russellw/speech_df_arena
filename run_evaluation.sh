#!/bin/bash

# Set environment variables for correct paths
export DF_ARENA_PROTOCOL_FILES_DIR="/mnt/c/speech_df_arena/protocol_files"
export DF_ARENA_CHECKPOINTS_DIR="/mnt/c/speech_df_arena/df_arena_checkpoints"

# Run the evaluation script
python evaluate.py "$@"