#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
BASE_MODEL_DIR="../models/Wan2.1-VACE-1.3B"
CHECKPOINT_DIR="checkpoints/reference-control-disentangled-2"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-latest}"
METADATA_FILE="datasets/fashion_vace/metadata_test_16fps.json"
SAMPLE_ID=""
GT_VIDEO_DIR="${ROOT}/datasets/fashion_vace/videos_16fps/test/gt"
RESULT_DIR="output_dir_fashion_vace/test_results/reference-control-disentangled-2/16fps-720x944"
SAMPLE_H="944"
SAMPLE_W="720"
VIDEO_LENGTH="81"
EXPORT_FPS="16"
REFERENCE_RESIDUAL_SCALE="${REFERENCE_RESIDUAL_SCALE:-1.5}"
CONTROL_RESIDUAL_SCALE="${CONTROL_RESIDUAL_SCALE:-0.8}"

export CUDA_VISIBLE_DEVICES


if [[ ! -f "${METADATA_FILE}" ]]; then
  echo "Missing ${METADATA_FILE}. Run:"
  echo "  python scripts/wan2.1_vace/convert_fashion_videos_16fps.py --workers 8"
  exit 1
fi

RUN_METADATA_FILE="${METADATA_FILE}"
if [[ -n "${SAMPLE_ID}" ]]; then
  RUN_METADATA_FILE="${ROOT}/datasets/fashion_vace/.single_${SAMPLE_ID}.json"
  python3 - "${METADATA_FILE}" "${RUN_METADATA_FILE}" "${SAMPLE_ID}" <<'PY'
import json
import sys

src, dst, sample_id = sys.argv[1:4]
with open(src, "r", encoding="utf-8") as f:
    samples = json.load(f)

selected = [sample for sample in samples if sample.get("id") == sample_id]
if not selected:
    raise SystemExit(f"SAMPLE_ID not found in {src}: {sample_id}")

with open(dst, "w", encoding="utf-8") as f:
    json.dump(selected, f, ensure_ascii=False, indent=2)
PY
fi


python scripts/wan2.1_vace/batch_predict_fashion_test.py \
  --output_dir "${CHECKPOINT_DIR}" \
  --checkpoints "${CHECKPOINT_STEPS}" \
  --pretrained_model_name_or_path "${BASE_MODEL_DIR}" \
  --save_dir "${RESULT_DIR}" \
  --sample_height "${SAMPLE_H}" \
  --sample_width "${SAMPLE_W}" \
  --video_length "${VIDEO_LENGTH}" \
  --metadata "${RUN_METADATA_FILE}" \
  --vace_reference_context_scale "${REFERENCE_RESIDUAL_SCALE}" \
  --vace_control_context_scale "${CONTROL_RESIDUAL_SCALE}" \
  --fps "${EXPORT_FPS}" \
  --gt_dir "${GT_VIDEO_DIR}"
