#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"


METADATA_FILE="${METADATA_FILE:-datasets/fashion_vace/metadata_pose_batch_16fps.json}"
SAVE_DIR="samples/1.3b"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-/data/shared/models/Wan2.1-VACE-1.3B}"
SAMPLE_H="${SAMPLE_H:-576}"
SAMPLE_W="${SAMPLE_W:-448}"
VIDEO_LENGTH="${VIDEO_LENGTH:-81}"
EXPORT_FPS="${EXPORT_FPS:-16}"
SEED="${SEED:-42}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-40}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-5.0}"

python scripts/wan2.1_vace/batch_predict_first_frames_pose_1_3b.py \
  --metadata "${METADATA_FILE}" \
  --save_dir "${SAVE_DIR}" \
  --pretrained_model_name_or_path "${BASE_MODEL_DIR}" \
  --sample_height "${SAMPLE_H}" \
  --sample_width "${SAMPLE_W}" \
  --video_length "${VIDEO_LENGTH}" \
  --fps "${EXPORT_FPS}" \
  --seed "${SEED}" \
  --num_inference_steps "${NUM_INFERENCE_STEPS}" \
  --guidance_scale "${GUIDANCE_SCALE}" \
  --skip_existing
