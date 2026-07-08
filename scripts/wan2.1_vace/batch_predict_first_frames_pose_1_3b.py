#!/usr/bin/env python3
"""Batch Wan2.1-VACE inference following examples/wan2.1_vace/predict_v2v_control.py.

Input metadata rows must contain:
  id, ref_image_path, control_file_path
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
from pathlib import Path

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from videox_fun.dist import set_multi_gpus_devices
from videox_fun.models import AutoencoderKLWan, VaceWanTransformer3DModel, WanT5EncoderModel
from videox_fun.models.cache_utils import get_teacache_coefficients
from videox_fun.pipeline import WanVacePipeline
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from videox_fun.utils.fp8_optimization import replace_parameters_by_name
from videox_fun.utils.utils import (
    filter_kwargs,
    get_image_latent,
    get_image_to_video_latent,
    get_video_to_video_latent,
    save_videos_grid,
)

DEFAULT_PROMPT = (
    "photorealistic fashion video, natural lighting, consistent clothing texture "
    "and skin tone, stable motion."
)
DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
    "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Batch predict VACE videos from metadata.")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=ROOT / "datasets/fashion_vace/metadata_pose_batch_16fps.json",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="/data/shared/models/Wan2.1-VACE-1.3B",
    )
    parser.add_argument("--config_path", type=str, default="config/wan2.1/wan_civitai.yaml")
    parser.add_argument("--save_dir", type=Path, default=ROOT / "samples/1.3b")
    parser.add_argument("--sample_height", type=int, default=576)
    parser.add_argument("--sample_width", type=int, default=448)
    parser.add_argument("--video_length", type=int, default=81)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--vace_context_scale", type=float, default=1.0)
    parser.add_argument("--shift", type=int, default=16)
    parser.add_argument(
        "--sampler_name",
        type=str,
        default="Flow_Unipc",
        choices=["Flow", "Flow_Unipc", "Flow_DPM++"],
    )
    parser.add_argument(
        "--gpu_memory_mode",
        type=str,
        default="sequential_cpu_offload",
        choices=["model_full_load", "model_cpu_offload", "sequential_cpu_offload"],
    )
    parser.add_argument("--disable_teacache", action="store_true")
    parser.add_argument("--teacache_threshold", type=float, default=0.05)
    parser.add_argument("--num_skip_start_steps", type=int, default=5)
    parser.add_argument("--teacache_offload", action="store_true")
    parser.add_argument("--cfg_skip_ratio", type=float, default=0.0)
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def safe_name(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]+', "_", str(name))


def load_samples(metadata: Path) -> list[dict]:
    metadata = resolve_path(metadata)
    with open(metadata, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"metadata must be a non-empty list: {metadata}")

    samples = []
    for index, row in enumerate(rows):
        sample_id = row.get("id") or Path(row.get("ref_image_path", "")).stem
        ref_path = row.get("ref_image_path")
        control_path = row.get("control_file_path")
        if not sample_id or not ref_path or not control_path:
            print(f"skip row {index}: missing id/ref_image_path/control_file_path")
            continue

        ref_path = resolve_path(ref_path)
        control_path = resolve_path(control_path)
        if not ref_path.is_file():
            print(f"skip {sample_id}: missing ref image {ref_path}")
            continue
        if not control_path.is_file():
            print(f"skip {sample_id}: missing control video {control_path}")
            continue

        item = dict(row)
        item["id"] = str(sample_id)
        item["ref_image_path"] = str(ref_path)
        item["control_file_path"] = str(control_path)
        samples.append(item)

    if not samples:
        raise ValueError(f"No valid samples in metadata: {metadata}")
    print(f"metadata: {metadata}")
    print(f"samples: {len(samples)}")
    return samples


def build_pipeline(args, config, device: torch.device, weight_dtype: torch.dtype) -> WanVacePipeline:
    model_name = args.pretrained_model_name_or_path
    transformer = VaceWanTransformer3DModel.from_pretrained(
        os.path.join(
            model_name,
            config["transformer_additional_kwargs"].get("transformer_subpath", "transformer"),
        ),
        transformer_additional_kwargs=OmegaConf.to_container(
            config["transformer_additional_kwargs"]
        ),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )

    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(model_name, config["vae_kwargs"].get("vae_subpath", "vae")),
        additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
    ).to(weight_dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(
            model_name,
            config["text_encoder_kwargs"].get("tokenizer_subpath", "tokenizer"),
        )
    )

    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(
            model_name,
            config["text_encoder_kwargs"].get("text_encoder_subpath", "text_encoder"),
        ),
        additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    ).eval()

    scheduler_cls = {
        "Flow": FlowMatchEulerDiscreteScheduler,
        "Flow_Unipc": FlowUniPCMultistepScheduler,
        "Flow_DPM++": FlowDPMSolverMultistepScheduler,
    }[args.sampler_name]
    if args.sampler_name in ("Flow_Unipc", "Flow_DPM++"):
        config["scheduler_kwargs"]["shift"] = 1
    scheduler = scheduler_cls(
        **filter_kwargs(scheduler_cls, OmegaConf.to_container(config["scheduler_kwargs"]))
    )

    pipeline = WanVacePipeline(
        transformer=transformer,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
    )

    if args.gpu_memory_mode == "sequential_cpu_offload":
        replace_parameters_by_name(transformer, ["modulation"], device=device)
        transformer.freqs = transformer.freqs.to(device=device)
        pipeline.enable_sequential_cpu_offload(device=device)
    elif args.gpu_memory_mode == "model_cpu_offload":
        pipeline.enable_model_cpu_offload(device=device)
    else:
        pipeline.to(device=device)

    if not args.disable_teacache:
        coefficients = get_teacache_coefficients(model_name)
        if coefficients is not None:
            print(
                f"Enable TeaCache threshold={args.teacache_threshold}, "
                f"skip_start={args.num_skip_start_steps}"
            )
            pipeline.transformer.enable_teacache(
                coefficients,
                args.num_inference_steps,
                args.teacache_threshold,
                num_skip_start_steps=args.num_skip_start_steps,
                offload=args.teacache_offload,
            )

    if args.cfg_skip_ratio is not None:
        print(f"Enable cfg_skip_ratio {args.cfg_skip_ratio}.")
        pipeline.transformer.enable_cfg_skip(args.cfg_skip_ratio, args.num_inference_steps)

    return pipeline


@torch.no_grad()
def predict_one(
    pipeline: WanVacePipeline,
    sample: dict,
    args,
    sample_size: list[int],
    video_length: int,
    device: torch.device,
) -> torch.Tensor:
    subject_ref_images = [
        get_image_latent(sample["ref_image_path"], sample_size=sample_size, padding=True)
    ]
    subject_ref_images = torch.cat(subject_ref_images, dim=2)

    inpaint_video, inpaint_video_mask, _ = get_image_to_video_latent(
        None,
        None,
        video_length=video_length,
        sample_size=sample_size,
    )
    control_video, _, _, _ = get_video_to_video_latent(
        sample["control_file_path"],
        video_length=video_length,
        sample_size=sample_size,
        fps=args.fps,
        ref_image=None,
    )

    prompt = sample.get("text") or args.prompt
    generator = torch.Generator(device=device).manual_seed(args.seed)
    return pipeline(
        prompt,
        num_frames=video_length,
        negative_prompt=args.negative_prompt,
        height=sample_size[0],
        width=sample_size[1],
        generator=generator,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        video=inpaint_video,
        mask_video=inpaint_video_mask,
        control_video=control_video,
        subject_ref_images=subject_ref_images,
        shift=args.shift,
        vace_context_scale=args.vace_context_scale,
    ).videos


def main():
    args = parse_args()
    os.chdir(ROOT)

    samples = load_samples(args.metadata)
    args.save_dir.mkdir(parents=True, exist_ok=True)

    device = set_multi_gpus_devices(ulysses_degree=1, ring_degree=1)
    config = OmegaConf.load(args.config_path)
    weight_dtype = torch.bfloat16
    sample_size = [args.sample_height, args.sample_width]
    video_length = (
        int((args.video_length - 1) // 4 * 4) + 1 if args.video_length != 1 else 1
    )

    pipeline = build_pipeline(args, config, device, weight_dtype)
    temporal_ratio = int(pipeline.vae.config.temporal_compression_ratio)
    video_length = (
        int((args.video_length - 1) // temporal_ratio * temporal_ratio) + 1
        if args.video_length != 1
        else 1
    )
    print(f"sample_size={sample_size}, video_length={video_length}, save_dir={args.save_dir}")

    try:
        for sample in samples:
            out_path = args.save_dir / f"{safe_name(sample['id'])}.mp4"
            if args.skip_existing and out_path.is_file():
                print(f"skip existing {out_path}")
                continue

            print(f"infer {sample['id']}")
            videos = predict_one(
                pipeline,
                sample,
                args,
                sample_size,
                video_length,
                device,
            )
            save_videos_grid(videos, str(out_path), fps=args.fps)
            print(f"  -> {out_path}")

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        del pipeline
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
