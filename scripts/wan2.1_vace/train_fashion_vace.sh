#!/usr/bin/env bash
# Wan2.1-VACE-1.3B: fashion GT + normal control (16 fps copies under datasets/)
#
# Default data: datasets/fashion_vace/videos_16fps/train/{gt,normal}/
#   metadata: datasets/fashion_vace/metadata_train_16fps.json
#
# Build 16 fps train/test copies (originals untouched):
#   python scripts/wan2.1_vace/convert_fashion_videos_16fps.py --workers 8
#
# Legacy 30 fps metadata (original paths):
#   DATASET_META_NAME=datasets/fashion_vace/metadata_train.json bash ...
#
# Regenerate split lists from originals:
#   python scripts/wan2.1_vace/prepare_fashion_metadata.py
#
# Logs (stdout+stderr, one file per run):
#   ${OUTPUT_DIR}/train_logs/train_YYYYMMDD_HHMMSS.log
#   ${OUTPUT_DIR}/train_logs/latest.log  -> symlink to latest run
# Custom log path: TRAIN_LOG_FILE=/path/to/foo.log bash ...
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

export MODEL_NAME="${MODEL_NAME:-/data/shared/models/Wan2.1-VACE-1.3B}"
export DATASET_NAME="${DATASET_NAME:-}"
export DATASET_META_NAME="${DATASET_META_NAME:-datasets/fashion_vace/metadata_train.json}"

# 81 帧双视频解码极占内存：默认 0 worker（主进程加载），避免 worker 被 OOM Kill
# 若内存充足可设 DATALOADER_NUM_WORKERS=1
export NUM_PROCESSES="${NUM_PROCESSES:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
# 竖屏素材（约 720×940）：sample_size = [H, W] = [576, 320]
# 32GB 卡 + 81 帧；显存不够可改为 480×272 等（须为 16 的倍数）
export FIX_SAMPLE_H="${FIX_SAMPLE_H:-576}"
export FIX_SAMPLE_W="${FIX_SAMPLE_W:-448}"
export VIDEO_SAMPLE_N_FRAMES="${VIDEO_SAMPLE_N_FRAMES:-81}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

export OUTPUT_DIR="${OUTPUT_DIR:-/data/miaomiao/checkpoints/multi-control}"
LOG_DIR="${OUTPUT_DIR}/train_logs"
mkdir -p "${LOG_DIR}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
TRAIN_LOG_FILE="${TRAIN_LOG_FILE:-${LOG_DIR}/train_${RUN_TS}.log}"
ln -sfn "$(basename "${TRAIN_LOG_FILE}")" "${LOG_DIR}/latest.log"

if [[ ! -f "${DATASET_META_NAME}" ]]; then
  echo "Missing ${DATASET_META_NAME}."
  echo "  python scripts/wan2.1_vace/convert_fashion_videos_16fps.py --workers 8"
  echo "  # or: python scripts/wan2.1_vace/prepare_fashion_metadata.py  (30 fps originals)"
  exit 1
fi

TRAIN_DATA_DIR_ARGS=()
if [[ -n "${DATASET_NAME}" ]]; then
  TRAIN_DATA_DIR_ARGS=(--train_data_dir="$DATASET_NAME")
fi

RESUME_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  RESUME_ARGS=(--resume_from_checkpoint="${RESUME_FROM_CHECKPOINT}")
fi

{
  echo "========== train run ${RUN_TS} =========="
  echo "log_file=${TRAIN_LOG_FILE}"
  echo "OUTPUT_DIR=${OUTPUT_DIR}"
  echo "DATASET_META_NAME=${DATASET_META_NAME}"
  echo "NUM_PROCESSES=${NUM_PROCESSES} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "FIX_SAMPLE_H/W=${FIX_SAMPLE_H}/${FIX_SAMPLE_W} VIDEO_SAMPLE_N_FRAMES=${VIDEO_SAMPLE_N_FRAMES}"
  echo "RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-}"
  echo "=========================================="
} | tee "${TRAIN_LOG_FILE}"

set -o pipefail
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" accelerate launch --num_processes="${NUM_PROCESSES}" --mixed_precision="bf16" scripts/wan2.1_vace/train.py \
  --config_path="config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  "${TRAIN_DATA_DIR_ARGS[@]}" \
  --train_data_meta="$DATASET_META_NAME" \
  --image_sample_size="${FIX_SAMPLE_H}" \
  --video_sample_size="${FIX_SAMPLE_W}" \
  --token_sample_size="${FIX_SAMPLE_W}" \
  --video_sample_stride=1 \
  --video_sample_n_frames="${VIDEO_SAMPLE_N_FRAMES}" \
  --train_batch_size=1 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers="${DATALOADER_NUM_WORKERS}" \
  --max_train_steps=1000 \
  --checkpointing_steps=50 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="${OUTPUT_DIR}" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --enable_bucket \
  --uniform_sampling \
  --low_vram \
  --control_ref_image="first_frame" \
  --control_context_ratio=0.85 \
  --photo_ref_ratio=1.0 \
  --force_subject_ref \
  --align_gt_frames_to_control \
  --enable_multi_control_adapter \
  --trainable_modules "vace" \
  "${RESUME_ARGS[@]}" \
  2>&1 | tee -a "${TRAIN_LOG_FILE}"
exit "${PIPESTATUS[0]}"
