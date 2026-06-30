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
#   /data/miaomiao/checkpoints/multi-control-residual-1000-steps/train_logs/train_YYYYMMDD_HHMMSS.log
#   /data/miaomiao/checkpoints/multi-control-residual-1000-steps/train_logs/latest.log  -> symlink to latest run
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
TRAIN_LOG_FILE="/data/miaomiao/checkpoints/multi-control-residual-1000-steps/train_logs/train_${RUN_TS}.log"
mkdir -p "/data/miaomiao/checkpoints/multi-control-residual-1000-steps/train_logs"
ln -sfn "$(basename "${TRAIN_LOG_FILE}")" "/data/miaomiao/checkpoints/multi-control-residual-1000-steps/train_logs/latest.log"

if [[ ! -f "datasets/fashion_vace/metadata_train_16fps.json" ]]; then
  echo "Missing datasets/fashion_vace/metadata_train_16fps.json."
  echo "  python scripts/wan2.1_vace/convert_fashion_videos_16fps.py --workers 8"
  echo "  # or: python scripts/wan2.1_vace/prepare_fashion_metadata.py  (30 fps originals)"
  exit 1
fi

{
  echo "========== train run ${RUN_TS} =========="
  echo "log_file=${TRAIN_LOG_FILE}"
  echo "OUTPUT_DIR=/data/miaomiao/checkpoints/multi-control-residual-1000-steps"
  echo "DATASET_META_NAME=datasets/fashion_vace/metadata_train.json"
  echo "NUM_PROCESSES=4 CUDA_VISIBLE_DEVICES=0,1,2,3"
  echo "FIX_SAMPLE_H/W=576/448 VIDEO_SAMPLE_N_FRAMES=81"
  echo "ENABLE_MULTI_CONTROL_ADAPTER=1"
  echo "SYNTHETIC_MODALITY_DROPOUT_PROB=0.5"
  echo "SYNTHETIC_FULL_MODALITY_PROB=0.25"
  echo "=========================================="
} | tee "${TRAIN_LOG_FILE}"

set -o pipefail
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
NO_ALBUMENTATIONS_UPDATE="1" \
NCCL_ASYNC_ERROR_HANDLING="1" \
PYTHONUNBUFFERED="1" \
CUDA_VISIBLE_DEVICES="1,2,3,4" \
accelerate launch --num_processes="4" --mixed_precision="bf16" scripts/wan2.1_vace/train.py \
  --config_path="config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path="/data/shared/models/Wan2.1-VACE-1.3B" \
  --train_data_meta="datasets/fashion_vace/metadata_train_16fps.json" \
  --image_sample_size="576" \
  --video_sample_size="448" \
  --token_sample_size="448" \
  --video_sample_stride=1 \
  --video_sample_n_frames="81" \
  --train_batch_size=1 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers="0" \
  --max_train_steps=1000 \
  --checkpointing_steps=200 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="/data/miaomiao/checkpoints/multi-control-residual-1000-steps" \
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
  --trainable_modules "vace" \
  --enable_multi_control_adapter \
  --synthetic_modality_dropout_prob="0.5" \
  --synthetic_full_modality_prob="0.25" \
  2>&1 | tee -a "${TRAIN_LOG_FILE}"
exit "${PIPESTATUS[0]}"
