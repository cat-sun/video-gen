#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
BASE_MODEL_DIR="/data/shared/models/Wan2.1-VACE-1.3B"
CHECKPOINT_DIR="/data/miaomiao/checkpoints/multi-control"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-latest}"
METADATA_FILE="datasets/fashion_vace/metadata_test_16fps.json"
SAMPLE_ID=""
GT_VIDEO_DIR="${ROOT}/datasets/fashion_vace/videos_16fps/test/gt"
RESULT_DIR="output_dir_fashion_vace/test_results/multi-control-2/16fps-720x944"
SAMPLE_H="944"
SAMPLE_W="720"
VIDEO_LENGTH="81"
EXPORT_FPS="16"

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


cat <<EOF
========== Fashion VACE inference ==========
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
BASE_MODEL_DIR=${BASE_MODEL_DIR}
CHECKPOINT_DIR="/data/miaomiao/checkpoints/multi-control"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-latest}"
METADATA_FILE=${METADATA_FILE}
SAMPLE_ID=${SAMPLE_ID:-<all>}
RUN_METADATA_FILE=${RUN_METADATA_FILE}
GT_VIDEO_DIR=${GT_VIDEO_DIR}
RESULT_DIR=${RESULT_DIR}
SAMPLE_H/W=${SAMPLE_H}/${SAMPLE_W}
VIDEO_LENGTH=${VIDEO_LENGTH}
EXPORT_FPS=${EXPORT_FPS}
============================================
EOF

python scripts/wan2.1_vace/batch_predict_fashion_test.py \
  --output_dir "${CHECKPOINT_DIR}" \
  --checkpoints "${CHECKPOINT_STEPS}" \
  --pretrained_model_name_or_path "${BASE_MODEL_DIR}" \
  --save_dir "${RESULT_DIR}" \
  --sample_height "${SAMPLE_H}" \
  --sample_width "${SAMPLE_W}" \
  --video_length "${VIDEO_LENGTH}" \
  --metadata "${RUN_METADATA_FILE}" \
  --fps "${EXPORT_FPS}" \
  --gt_dir "${GT_VIDEO_DIR}"
