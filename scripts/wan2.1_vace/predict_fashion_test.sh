#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

# Edit these values directly when you want to change inference.
BASE_MODEL_DIR="/data/shared/models/Wan2.1-VACE-1.3B"
CHECKPOINT_DIR="/data/miaomiao/checkpoints/multi-control-residual-1000-steps"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-latest}"
METADATA_FILE="datasets/fashion_vace/metadata_test_16fps.json"
SAMPLE_ID=""
GT_VIDEO_DIR="${ROOT}/datasets/fashion_vace/videos_16fps/test/gt"
RESULT_DIR="output_dir_fashion_vace/test_results/multi-control-30fps-448x576-1000-steps/16fps-720x944"
SAMPLE_H=944
SAMPLE_W=720
VIDEO_LENGTH=81
EXPORT_FPS=16


if [[ ! -f "${METADATA_FILE}" ]]; then
  echo "Metadata file not found: ${METADATA_FILE}" >&2
  exit 1
fi

RUN_METADATA_FILE="${METADATA_FILE}"
if [[ -n "${SAMPLE_ID}" ]]; then
  RUN_METADATA_FILE="/tmp/fashion_vace_predict_${SAMPLE_ID}.json"
  python - <<PY_INNER
import json
from pathlib import Path
metadata = Path("${METADATA_FILE}")
sample_id = "${SAMPLE_ID}"
out = Path("${RUN_METADATA_FILE}")
items = json.loads(metadata.read_text())
if not isinstance(items, list):
    raise SystemExit(f"Expected metadata list in {metadata}")
matched = [item for item in items if str(item.get("sample_id", "")) == sample_id]
if not matched:
    raise SystemExit(f"sample_id={sample_id!r} not found in {metadata}")
out.write_text(json.dumps(matched, ensure_ascii=False, indent=2))
print(f"Wrote single-sample metadata: {out}")
PY_INNER
fi

cat <<EOF
Running Wan2.1 VACE fashion inference
  BASE_MODEL_DIR=${BASE_MODEL_DIR}
  CHECKPOINT_DIR=${CHECKPOINT_DIR}
  CHECKPOINT_STEPS=${CHECKPOINT_STEPS}
  METADATA=${RUN_METADATA_FILE}
  RESULT_DIR=${RESULT_DIR}
  SAMPLE=${SAMPLE_W}x${SAMPLE_H}, frames=${VIDEO_LENGTH}, fps=${EXPORT_FPS}
  GPU=${CUDA_VISIBLE_DEVICES}
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
  --config_path "config/wan2.1/wan_civitai.yaml" \
  --enable_multi_control_adapter \
  --fps "${EXPORT_FPS}" \
  --gt_dir "${GT_VIDEO_DIR}"
