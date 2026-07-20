#!/usr/bin/env python3
"""Convert Blender render-data PNG sequences into VideoX-Fun training videos.

Expected input layout:

    render-data/004gbuffer/{alpha,depth,motion,normal,preview,uv}/*.png

Default output layout:

    render-data-training/{gt,depth,normal,uv,motion,mask}/*.mp4
    render-data-training/metadata.json

The Blender G-buffer PNGs are 16-bit sRGB-encoded data images. This tool
linearizes them before quantization. Normals are converted from Blender world
coordinates (X, Y, Z) to the camera convention used by the current training
normals (X, Z, -Y). Alpha becomes the training ``mask`` modality and preview
becomes ``gt``.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


INPUT_CHANNELS = ("alpha", "depth", "motion", "normal", "preview", "uv")
OUTPUT_CHANNELS = ("gt", "depth", "normal", "uv", "motion", "mask")
FRAME_NUMBER_RE = re.compile(r"(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert render-data G-buffer PNGs to training MP4 videos."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("render-data"),
        help="Input render-data directory (default: render-data).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("render-data-training"),
        help="New output dataset directory (default: render-data-training).",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Keep only the first N seconds of each clip (default: 10).",
    )
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="medium")
    parser.add_argument(
        "--size",
        metavar="WIDTHxHEIGHT",
        help="Optional output size. By default the source resolution is kept.",
    )
    parser.add_argument(
        "--resize-mode",
        choices=("stretch", "contain", "cover"),
        default="contain",
        help="Resize policy used with --size (default: contain).",
    )
    parser.add_argument(
        "--clips",
        nargs="*",
        help="Optional clip IDs, e.g. --clips 004 023. Default: all *gbuffer dirs.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Limit frames per clip for smoke tests.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output videos instead of failing.",
    )
    parser.add_argument(
        "--text",
        default=(
            "photorealistic fashion video, natural lighting, consistent clothing "
            "texture and skin tone, stable motion."
        ),
        help="Text prompt stored in generated metadata.",
    )
    return parser.parse_args()


def parse_size(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None
    match = re.fullmatch(r"(\d+)[xX](\d+)", value)
    if not match:
        raise ValueError(f"Invalid --size {value!r}; expected WIDTHxHEIGHT")
    width, height = map(int, match.groups())
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise ValueError("Output width and height must be positive even numbers")
    return width, height


def frame_number(path: Path) -> int:
    match = FRAME_NUMBER_RE.search(path.stem)
    if not match:
        raise ValueError(f"No numeric frame index in {path.name}")
    return int(match.group(1))


def indexed_frames(directory: Path) -> dict[int, Path]:
    frames = {}
    for path in directory.glob("*.png"):
        index = frame_number(path)
        if index in frames:
            raise ValueError(f"Duplicate frame {index} under {directory}")
        frames[index] = path
    return frames


def validate_clip(clip_dir: Path, max_frames: int | None) -> list[int]:
    channel_frames = {
        channel: indexed_frames(clip_dir / channel) for channel in INPUT_CHANNELS
    }
    missing = [channel for channel, frames in channel_frames.items() if not frames]
    if missing:
        raise ValueError(f"{clip_dir}: empty/missing channels: {', '.join(missing)}")

    reference = set(channel_frames["preview"])
    for channel, frames in channel_frames.items():
        indices = set(frames)
        if indices != reference:
            absent = sorted(reference - indices)[:10]
            extra = sorted(indices - reference)[:10]
            raise ValueError(
                f"{clip_dir}/{channel}: frame mismatch; missing={absent}, extra={extra}"
            )

    indices = sorted(reference)
    if indices != list(range(indices[0], indices[-1] + 1)):
        raise ValueError(f"{clip_dir}: frame indices are not contiguous")
    if max_frames is not None:
        indices = indices[:max_frames]
    return indices


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Cannot read {path}")
    return image


def rgb01(image: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    if image.dtype == np.uint16:
        scale = 65535.0
    elif image.dtype == np.uint8:
        scale = 255.0
    else:
        raise TypeError(f"Unsupported image dtype {image.dtype}")

    if image.ndim == 2:
        rgb = np.repeat(image[..., None], 3, axis=-1)
        alpha = None
    elif image.shape[2] == 4:
        rgb = image[..., :3][..., ::-1]
        alpha = image[..., 3].astype(np.float32) / scale
    elif image.shape[2] == 3:
        rgb = image[..., ::-1]
        alpha = None
    else:
        raise ValueError(f"Unsupported image shape {image.shape}")
    return rgb.astype(np.float32) / scale, alpha


def srgb_to_linear(value: np.ndarray) -> np.ndarray:
    return np.where(
        value <= 0.04045,
        value / 12.92,
        ((value + 0.055) / 1.055) ** 2.4,
    )


def load_alpha(path: Path) -> np.ndarray:
    rgb, embedded_alpha = rgb01(read_image(path))
    if embedded_alpha is not None:
        alpha = embedded_alpha
    else:
        # Blender saved this pass through the display transform as RGB.
        alpha = srgb_to_linear(rgb).mean(axis=-1)
    return np.clip(alpha, 0.0, 1.0)


def convert_frame(channel: str, path: Path, alpha: np.ndarray) -> np.ndarray:
    rgb, embedded_alpha = rgb01(read_image(path))

    if channel == "preview":
        # Preview is a display image; do not linearize it again.
        coverage = embedded_alpha if embedded_alpha is not None else alpha
        output = rgb * coverage[..., None]
    elif channel == "alpha":
        output = np.repeat(alpha[..., None], 3, axis=-1)
    elif channel == "normal":
        world = srgb_to_linear(rgb) * 2.0 - 1.0
        camera = np.stack(
            (world[..., 0], world[..., 2], -world[..., 1]), axis=-1
        )
        length = np.linalg.norm(camera, axis=-1, keepdims=True)
        camera = camera / np.maximum(length, 1e-8)
        output = camera * 0.5 + 0.5
        output[alpha <= 0.5] = 0.0
    elif channel == "depth":
        # Blender depth is near=0/far=1; current training depth is near=bright.
        depth = srgb_to_linear(rgb).mean(axis=-1)
        inverse_depth = 1.0 - depth
        output = np.repeat(inverse_depth[..., None], 3, axis=-1)
        output[alpha <= 0.5] = 0.0
    elif channel in ("uv", "motion"):
        output = srgb_to_linear(rgb)
        output[alpha <= 0.5] = 0.0
    else:
        raise ValueError(f"Unknown channel {channel}")

    return np.clip(np.rint(output * 255.0), 0, 255).astype(np.uint8)


def resize_frame(
    frame: np.ndarray,
    target_size: tuple[int, int] | None,
    mode: str,
) -> np.ndarray:
    if target_size is None:
        return frame
    target_w, target_h = target_size
    source_h, source_w = frame.shape[:2]
    if (source_w, source_h) == target_size:
        return frame
    if mode == "stretch":
        return cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)

    scale = (
        min(target_w / source_w, target_h / source_h)
        if mode == "contain"
        else max(target_w / source_w, target_h / source_h)
    )
    resized_w = max(1, int(round(source_w * scale)))
    resized_h = max(1, int(round(source_h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=interpolation)

    if mode == "contain":
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        x = (target_w - resized_w) // 2
        y = (target_h - resized_h) // 2
        canvas[y : y + resized_h, x : x + resized_w] = resized
        return canvas

    x = (resized_w - target_w) // 2
    y = (resized_h - target_h) // 2
    return resized[y : y + target_h, x : x + target_w]


class VideoWriter:
    def __init__(
        self,
        path: Path,
        width: int,
        height: int,
        fps: float,
        crf: int,
        preset: str,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ]
        self.path = path
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        assert self.process.stdin is not None
        self.process.stdin.close()
        return_code = self.process.wait()
        if return_code:
            raise RuntimeError(f"ffmpeg failed for {self.path} ({return_code})")


def relative_or_absolute(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def convert_clip(
    clip_dir: Path,
    output_root: Path,
    fps: float,
    crf: int,
    preset: str,
    target_size: tuple[int, int] | None,
    resize_mode: str,
    max_frames: int | None,
    overwrite: bool,
) -> dict[str, Path]:
    clip_id = clip_dir.name.removesuffix("gbuffer")
    indices = validate_clip(clip_dir, max_frames)
    paths = {
        "gt": output_root / "gt" / f"{clip_id}.mp4",
        "depth": output_root / "depth" / f"{clip_id}_depth.mp4",
        "normal": output_root / "normal" / f"{clip_id}_normal.mp4",
        "uv": output_root / "uv" / f"{clip_id}_uv.mp4",
        "motion": output_root / "motion" / f"{clip_id}_motion.mp4",
        "mask": output_root / "mask" / f"{clip_id}_mask.mp4",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Outputs already exist for {clip_id}; pass --overwrite: {existing[0]}"
        )

    first_preview = read_image(clip_dir / "preview" / f"preview_{indices[0]:04d}.png")
    source_h, source_w = first_preview.shape[:2]
    width, height = target_size or (source_w, source_h)
    if width % 2 or height % 2:
        raise ValueError(f"{clip_id}: output dimensions must be even, got {width}x{height}")

    writers = {
        name: VideoWriter(path, width, height, fps, crf, preset)
        for name, path in paths.items()
    }
    input_for_output = {
        "gt": "preview",
        "depth": "depth",
        "normal": "normal",
        "uv": "uv",
        "motion": "motion",
        "mask": "alpha",
    }
    try:
        for position, index in enumerate(indices, start=1):
            alpha = load_alpha(clip_dir / "alpha" / f"alpha_{index:04d}.png")
            for output_name, input_name in input_for_output.items():
                input_path = clip_dir / input_name / f"{input_name}_{index:04d}.png"
                frame = convert_frame(input_name, input_path, alpha)
                frame = resize_frame(frame, target_size, resize_mode)
                writers[output_name].write(frame)
            if position == 1 or position % 50 == 0 or position == len(indices):
                print(f"[{clip_id}] {position}/{len(indices)}", flush=True)
    finally:
        errors = []
        for writer in writers.values():
            try:
                writer.close()
            except Exception as error:  # Preserve all ffmpeg close failures.
                errors.append(error)
        if errors:
            raise errors[0]
    return paths


def main() -> int:
    args = parse_args()
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required but was not found in PATH")
    if not args.input_root.is_dir():
        raise FileNotFoundError(args.input_root)
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    if args.duration <= 0:
        raise ValueError("--duration must be positive")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    target_size = parse_size(args.size)
    duration_frames = max(1, int(round(args.duration * args.fps)))
    frame_limit = (
        duration_frames
        if args.max_frames is None
        else min(duration_frames, args.max_frames)
    )

    clip_dirs = sorted(path for path in args.input_root.glob("*gbuffer") if path.is_dir())
    if args.clips:
        requested = set(args.clips)
        clip_dirs = [
            path for path in clip_dirs if path.name.removesuffix("gbuffer") in requested
        ]
        found = {path.name.removesuffix("gbuffer") for path in clip_dirs}
        missing = sorted(requested - found)
        if missing:
            raise FileNotFoundError(f"Missing clips: {', '.join(missing)}")
    if not clip_dirs:
        raise RuntimeError(f"No *gbuffer directories found under {args.input_root}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    for channel in OUTPUT_CHANNELS:
        (args.output_root / channel).mkdir(parents=True, exist_ok=True)

    metadata = []
    metadata_base = args.output_root.parent.resolve()
    for clip_dir in clip_dirs:
        paths = convert_clip(
            clip_dir=clip_dir,
            output_root=args.output_root,
            fps=args.fps,
            crf=args.crf,
            preset=args.preset,
            target_size=target_size,
            resize_mode=args.resize_mode,
            max_frames=frame_limit,
            overwrite=args.overwrite,
        )
        clip_id = clip_dir.name.removesuffix("gbuffer")
        rel = {name: relative_or_absolute(path, metadata_base) for name, path in paths.items()}
        metadata.append(
            {
                "file_path": rel["gt"],
                "control_file_path": rel["normal"],
                "depth_file_path": rel["depth"],
                "normal_file_path": rel["normal"],
                "uv_file_path": rel["uv"],
                "motion_file_path": rel["motion"],
                "mask_file_path": rel["mask"],
                "gbuffer_paths": {
                    "depth": rel["depth"],
                    "normal": rel["normal"],
                    "uv": rel["uv"],
                    "motion": rel["motion"],
                    "mask": rel["mask"],
                },
                "text": args.text,
                "type": "video",
                "id": clip_id,
            }
        )

    metadata_path = args.output_root / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {metadata_path} ({len(metadata)} clips)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        print("ffmpeg pipe closed unexpectedly", file=sys.stderr)
        raise
