#!/usr/bin/env python3
"""Batch inference on fashion_vace test set (default: metadata_test_16fps.json).

Data layout (16 fps, originals untouched):
  datasets/fashion_vace/videos_16fps/test/{gt,normal}/
  datasets/fashion_vace/metadata_test_16fps.json

Aligned with examples/wan2.1_vace/predict_v2v_control.py:
  - ref_image_path  -> subject_ref_images (saved GT first-frame PNG)
  - control_file_path -> control_video (normal map)
  - empty inpaint + full mask (generate all frames)

Run prepare_fashion_test_first_frames.py to populate ref_image_path.

Usage:
  python scripts/wan2.1_vace/batch_predict_fashion_test.py
  bash scripts/wan2.1_vace/predict_fashion_test.sh

Checkpoint layout:
  {output_dir}/checkpoint-{step}/transformer/
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import traceback
from pathlib import Path

import torch
from decord import VideoReader
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from videox_fun.dist import set_multi_gpus_devices
from videox_fun.models import (AutoencoderKLWan, VaceWanTransformer3DModel,
                               WanT5EncoderModel)
from videox_fun.models.cache_utils import get_teacache_coefficients
from videox_fun.pipeline import WanVacePipeline
from videox_fun.utils.fp8_optimization import replace_parameters_by_name
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from videox_fun.utils.utils import (filter_kwargs, get_image_latent,
                                    get_image_to_video_latent,
                                    get_video_to_video_latent, save_videos_grid)

DEFAULT_META = ROOT / "datasets/fashion_vace/metadata_test_16fps.json"
DEFAULT_TEXT = (
    "photorealistic fashion video, natural lighting, consistent clothing texture "
    "and skin tone, stable motion."
)
DEFAULT_NEGATIVE = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，畸形，毁容"
)

# Fallback when metadata file_path is missing (docker / legacy 30 fps paths)
GT_PATH_CANDIDATES = [
    str(ROOT / "datasets/fashion_vace/videos_16fps/test/gt"),
    "/workspace/VideoX-Fun/datasets/fashion_vace/videos_16fps/test/gt",
    "/workspace/NormalCrafter/fashion_train_videos",
    "/home/miaomiao/NormalCrafter/fashion_train_videos",
    str(ROOT.parent / "NormalCrafter" / "fashion_train_videos"),
]


def parse_args():
    p = argparse.ArgumentParser(description="Batch test fashion VACE checkpoints.")
    p.add_argument("--metadata", type=Path, default=DEFAULT_META)
    p.add_argument(
        "--output_dir",
        type=Path,
        default=ROOT / "output_dir_fashion_vace",
        help="Training output_dir containing checkpoint-* folders.",
    )
    p.add_argument(
        "--checkpoints",
        type=str,
        default="all",
        help='Checkpoint steps: "all", "latest", or comma-separated e.g. "500,3000".',
    )
    p.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="/data/shared/models/Wan2.1-VACE-1.3B",
    )
    p.add_argument("--config_path", type=str, default="config/wan2.1/wan_civitai.yaml")
    p.add_argument(
        "--save_dir",
        type=Path,
        default=ROOT / "output_dir_fashion_vace/test_results",
    )
    p.add_argument("--sample_height", type=int, default=576, help="Portrait H, match FIX_SAMPLE_H in training.")
    p.add_argument("--sample_width", type=int, default=320, help="Portrait W, match FIX_SAMPLE_W in training.")
    p.add_argument("--video_length", type=int, default=81, help="Max frames (4n+1), match VIDEO_SAMPLE_N_FRAMES in training.")
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--guidance_scale", type=float, default=5.0)
    p.add_argument("--num_inference_steps", type=int, default=40)
    p.add_argument("--vace_context_scale", type=float, default=1.0)
    p.add_argument("--shift", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--sampler_name",
        type=str,
        default="Flow_Unipc",
        choices=["Flow", "Flow_Unipc", "Flow_DPM++"],
    )
    p.add_argument(
        "--gpu_memory_mode",
        type=str,
        default="sequential_cpu_offload",
        choices=[
            "model_full_load",
            "model_cpu_offload",
            "sequential_cpu_offload",
        ],
    )
    p.add_argument(
        "--disable_teacache",
        action="store_true",
        help="Disable TeaCache (enabled by default, same as predict_v2v_control.py).",
    )
    p.add_argument("--teacache_threshold", type=float, default=0.05)
    p.add_argument("--num_skip_start_steps", type=int, default=5)
    p.add_argument("--teacache_offload", action="store_true")
    p.add_argument("--cfg_skip_ratio", type=float, default=0.0)
    p.add_argument("--gt_dir", type=str, default="", help="Override GT video directory.")
    p.add_argument("--save_gt", action="store_true", help="Also save aligned GT clip.")
    p.add_argument("--skip_existing", action="store_true")
    return p.parse_args()


def resolve_project_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return str((ROOT / path).resolve())


def resolve_media_path(path: str, gt_dir: str | None = None) -> str:
    """Return first existing path; remap GT host paths to container paths when needed."""
    if os.path.isfile(path):
        return path

    name = Path(path).name
    if gt_dir:
        candidate = os.path.join(gt_dir, name)
        if os.path.isfile(candidate):
            return candidate

    for base in GT_PATH_CANDIDATES:
        candidate = os.path.join(base, name)
        if os.path.isfile(candidate):
            return candidate

    return path


def resolve_sample(sample: dict, gt_dir: str | None) -> dict | None:
    """Require ref_image_path + control_file_path (same inputs as predict_v2v_control.py)."""
    sample_id = sample.get("id") or Path(sample.get("file_path", "")).stem
    ref_raw = sample.get("ref_image_path")
    if ref_raw is None and sample_id:
        ref_raw = f"datasets/fashion_vace/first_frames/{sample_id}.png"
    if ref_raw is None:
        return None

    ref_path = resolve_project_path(ref_raw)
    ctrl_path = sample["control_file_path"]
    if not os.path.isfile(ref_path):
        return None
    if not os.path.isfile(ctrl_path):
        return None

    out = dict(sample)
    out["ref_image_path"] = ref_path
    out["control_file_path"] = ctrl_path
    if "file_path" in sample:
        out["file_path"] = resolve_media_path(sample["file_path"], gt_dir)
    return out


def _list_checkpoints(output_dir: Path) -> list[tuple[int, Path]]:
    found: list[tuple[int, Path]] = []
    for name in os.listdir(output_dir):
        m = re.fullmatch(r"checkpoint-(\d+)", name)
        if not m:
            continue
        step = int(m.group(1))
        ckpt = output_dir / name
        if (ckpt / "transformer").is_dir():
            found.append((step, ckpt))
    found.sort(key=lambda x: x[0])
    return found


def resolve_checkpoint_output_dir(output_dir: Path) -> Path:
    """Use output_dir if it has checkpoints; else try output_dir/81-frames."""
    if not output_dir.is_dir():
        raise FileNotFoundError(f"output_dir not found: {output_dir}")
    if _list_checkpoints(output_dir):
        return output_dir
    nested = output_dir / "81-frames"
    if nested.is_dir() and _list_checkpoints(nested):
        print(f"==> checkpoints under {nested} (not in {output_dir})")
        return nested
    raise FileNotFoundError(
        f"No checkpoint-*/transformer under {output_dir} or {nested}"
    )


def discover_checkpoints(output_dir: Path, spec: str) -> list[tuple[int, Path]]:
    output_dir = resolve_checkpoint_output_dir(output_dir)
    found = _list_checkpoints(output_dir)
    if not found:
        raise FileNotFoundError(f"No checkpoint-*/transformer under {output_dir}")

    if spec.strip().lower() == "all":
        return found
    if spec.strip().lower() == "latest":
        return [found[-1]]

    wanted = {int(s.strip()) for s in spec.split(",") if s.strip()}
    selected = [(s, p) for s, p in found if s in wanted]
    missing = wanted - {s for s, _ in selected}
    if missing:
        raise FileNotFoundError(f"Checkpoints not found for steps: {sorted(missing)}")
    return selected


def aligned_video_length(control_path: str, max_frames: int, temporal_ratio: int) -> int:
    ctrl_n = len(VideoReader(control_path))
    if ctrl_n == 0:
        raise ValueError(f"Empty control video: {control_path}")
    n = min(ctrl_n, max_frames)
    if temporal_ratio > 1 and n != 1:
        n = (n - 1) // temporal_ratio * temporal_ratio + 1
    return max(int(n), 1)


def build_pipeline(
    model_name: str,
    config,
    checkpoint_dir: Path | None,
    weight_dtype: torch.dtype,
    device: torch.device,
    sampler_name: str,
    gpu_memory_mode: str,
    enable_teacache: bool,
    teacache_threshold: float,
    num_skip_start_steps: int,
    teacache_offload: bool,
    cfg_skip_ratio: float,
    num_inference_steps: int,
) -> WanVacePipeline:
    transformer_kwargs = OmegaConf.to_container(config["transformer_additional_kwargs"])
    transformer_subpath = config["transformer_additional_kwargs"].get(
        "transformer_subpath", "transformer"
    )

    if checkpoint_dir is not None:
        transformer = VaceWanTransformer3DModel.from_pretrained(
            str(checkpoint_dir / "transformer"),
            transformer_additional_kwargs=transformer_kwargs,
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        )
        print(f"Loaded transformer from {checkpoint_dir / 'transformer'}")
    else:
        transformer = VaceWanTransformer3DModel.from_pretrained(
            os.path.join(model_name, transformer_subpath),
            transformer_additional_kwargs=transformer_kwargs,
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
        ),
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

    scheduler_dict = {
        "Flow": FlowMatchEulerDiscreteScheduler,
        "Flow_Unipc": FlowUniPCMultistepScheduler,
        "Flow_DPM++": FlowDPMSolverMultistepScheduler,
    }
    chosen = scheduler_dict[sampler_name]
    if sampler_name in ("Flow_Unipc", "Flow_DPM++"):
        config["scheduler_kwargs"]["shift"] = 1
    scheduler = chosen(
        **filter_kwargs(chosen, OmegaConf.to_container(config["scheduler_kwargs"]))
    )

    pipeline = WanVacePipeline(
        transformer=transformer,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
    )

    if gpu_memory_mode == "sequential_cpu_offload":
        replace_parameters_by_name(pipeline.transformer, ["modulation"], device=device)
        pipeline.transformer.freqs = pipeline.transformer.freqs.to(device=device)
        pipeline.enable_sequential_cpu_offload(device=device)
    elif gpu_memory_mode == "model_cpu_offload":
        pipeline.enable_model_cpu_offload(device=device)
    else:
        pipeline.to(device=device)

    if enable_teacache:
        coefficients = get_teacache_coefficients(model_name)
        if coefficients is not None:
            print(
                f"TeaCache: threshold={teacache_threshold}, "
                f"skip_start={num_skip_start_steps}"
            )
            pipeline.transformer.enable_teacache(
                coefficients,
                num_inference_steps,
                teacache_threshold,
                num_skip_start_steps=num_skip_start_steps,
                offload=teacache_offload,
            )

    if cfg_skip_ratio:
        print(f"cfg_skip_ratio={cfg_skip_ratio}")
        pipeline.transformer.enable_cfg_skip(cfg_skip_ratio, num_inference_steps)

    return pipeline


@torch.no_grad()
def run_one_sample(
    pipeline: WanVacePipeline,
    vae: AutoencoderKLWan,
    sample: dict,
    sample_size: list[int],
    max_video_length: int,
    fps: int,
    prompt: str,
    negative_prompt: str,
    guidance_scale: float,
    num_inference_steps: int,
    shift: int,
    vace_context_scale: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Same data flow as predict_v2v_control.py: ref_image_path + control_file_path."""
    ref_path = sample["ref_image_path"]
    control_path = sample["control_file_path"]
    temporal_ratio = int(vae.config.temporal_compression_ratio)

    n_raw = aligned_video_length(control_path, max_video_length, temporal_ratio)
    video_length = (
        int((n_raw - 1) // temporal_ratio * temporal_ratio) + 1 if n_raw != 1 else 1
    )

    subject_ref_images = [
        get_image_latent(ref_path, sample_size=sample_size, padding=True)
    ]
    subject_ref_images = torch.cat(subject_ref_images, dim=2)

    inpaint_video, inpaint_video_mask, _ = get_image_to_video_latent(
        None, None, video_length=video_length, sample_size=sample_size
    )
    control_video, _, _, _ = get_video_to_video_latent(
        control_path,
        video_length=video_length,
        sample_size=sample_size,
        fps=fps,
        ref_image=None,
    )

    return pipeline(
        prompt,
        num_frames=video_length,
        negative_prompt=negative_prompt,
        height=sample_size[0],
        width=sample_size[1],
        generator=generator,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        video=inpaint_video,
        mask_video=inpaint_video_mask,
        control_video=control_video,
        subject_ref_images=subject_ref_images,
        shift=shift,
        vace_context_scale=vace_context_scale,
    ).videos


def save_gt_clip(
    gt_path: str, video_length: int, sample_size: list[int], fps: int, out_path: Path
):
    import numpy as np

    vr = VideoReader(gt_path)
    n = min(len(vr), video_length)
    frames = []
    for i in range(n):
        img = Image.fromarray(vr[i].asnumpy()).convert("RGB")
        frames.append(np.array(img.resize((sample_size[1], sample_size[0]), Image.BILINEAR)))
    if not frames:
        return
    arr = np.stack(frames, axis=0)[np.newaxis, ...]
    tensor = torch.from_numpy(arr).permute(0, 4, 1, 2, 3).float() / 255.0
    save_videos_grid(tensor, str(out_path), fps=fps)


def main():
    args = parse_args()
    os.chdir(ROOT)
    enable_teacache = not args.disable_teacache
    gt_dir = args.gt_dir or None

    with open(args.metadata, "r", encoding="utf-8") as f:
        raw_samples = json.load(f)
    if not raw_samples:
        raise ValueError(f"Empty metadata: {args.metadata}")

    test_samples = []
    for s in raw_samples:
        resolved = resolve_sample(s, gt_dir)
        if resolved is None:
            print(
                f"  skip missing ref/control: {s.get('id')} "
                f"(ref={s.get('ref_image_path')}, ctrl={s.get('control_file_path')})"
            )
            continue
        test_samples.append(resolved)
    if not test_samples:
        raise ValueError("No valid test samples after path resolution.")

    checkpoints = discover_checkpoints(args.output_dir, args.checkpoints)
    config = OmegaConf.load(args.config_path)
    weight_dtype = torch.bfloat16
    device = set_multi_gpus_devices(ulysses_degree=1, ring_degree=1)
    sample_size = [args.sample_height, args.sample_width]

    args.save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Test samples: {len(test_samples)}, checkpoints: {[s for s, _ in checkpoints]}")

    vae_for_length = AutoencoderKLWan.from_pretrained(
        os.path.join(
            args.pretrained_model_name_or_path,
            config["vae_kwargs"].get("vae_subpath", "vae"),
        ),
        additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
    )

    for step, ckpt_dir in checkpoints:
        ckpt_name = f"checkpoint-{step}"
        out_ckpt_dir = args.save_dir / ckpt_name
        out_ckpt_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n========== {ckpt_name} ==========")

        pipeline = build_pipeline(
            args.pretrained_model_name_or_path,
            config,
            ckpt_dir,
            weight_dtype,
            device,
            args.sampler_name,
            args.gpu_memory_mode,
            enable_teacache,
            args.teacache_threshold,
            args.num_skip_start_steps,
            args.teacache_offload,
            args.cfg_skip_ratio,
            args.num_inference_steps,
        )

        for sample in test_samples:
            sample_id = sample.get("id") or Path(sample["ref_image_path"]).stem
            safe_id = re.sub(r'[\\/:*?"<>|]+', "_", str(sample_id))
            out_path = out_ckpt_dir / f"{safe_id}_pred.mp4"

            if args.skip_existing and out_path.is_file():
                print(f"  skip existing {out_path.name}")
                continue

            prompt = sample.get("text") or DEFAULT_TEXT
            generator = torch.Generator(device=device).manual_seed(args.seed)

            print(f"  infer {safe_id} ...")
            try:
                videos = run_one_sample(
                    pipeline,
                    vae_for_length,
                    sample,
                    sample_size,
                    args.video_length,
                    args.fps,
                    prompt,
                    DEFAULT_NEGATIVE,
                    args.guidance_scale,
                    args.num_inference_steps,
                    args.shift,
                    args.vace_context_scale,
                    generator,
                )
                save_videos_grid(videos, str(out_path), fps=args.fps)
                print(f"    -> {out_path}")

                if args.save_gt:
                    gt_path = sample.get("file_path")
                    if not gt_path or not os.path.isfile(gt_path):
                        print(f"    skip GT export (missing file_path): {safe_id}")
                    else:
                        gt_out = out_ckpt_dir / f"{safe_id}_gt.mp4"
                        if not (args.skip_existing and gt_out.is_file()):
                            tr = int(vae_for_length.config.temporal_compression_ratio)
                            n = aligned_video_length(
                                sample["control_file_path"], args.video_length, tr
                            )
                            vl = int((n - 1) // tr * tr) + 1 if n != 1 else 1
                            save_gt_clip(gt_path, vl, sample_size, args.fps, gt_out)
            except Exception as e:
                print(f"    FAILED {safe_id}: {e}")
                traceback.print_exc()

        del pipeline
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del vae_for_length
    print(f"\nDone. Results under {args.save_dir}")


if __name__ == "__main__":
    main()
