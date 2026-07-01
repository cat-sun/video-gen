#!/usr/bin/env python3
"""Build Wan VACE metadata for fashion GT + normal control videos.

Only writes JSON (no video copy/symlink). Training reads originals directly.

GT:      NormalCrafter/fashion_train_videos/*.mp4
Control: fashion_train/{id}_normal.mp4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_GT_DIR = Path(__file__).resolve().parents[2] / ".." / "NormalCrafter" / "fashion_train_videos"
DEFAULT_NORMAL_DIR = Path("/data/shared/miaomiao/fashion_train")
DEFAULT_OUT_DIR = Path(__file__).resolve().parents[2] / "datasets" / "fashion_vace"

DEFAULT_TEXT = (
    "photorealistic fashion video, natural lighting, consistent clothing texture "
    "and skin tone, stable motion."
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gt_dir", type=Path, default=DEFAULT_GT_DIR.resolve())
    p.add_argument(
        "--normal_dir",
        type=Path,
        default=DEFAULT_NORMAL_DIR,
        help="Directory with {id}_normal.mp4 control videos.",
    )
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--test_count", type=int, default=8, help="First N clips (sorted) for test.")
    p.add_argument("--text", type=str, default=DEFAULT_TEXT)
    p.add_argument(
        "--min_frames",
        type=int,
        default=81,
        help="Skip pairs whose control video has fewer frames (0 disables).",
    )
    return p.parse_args()


def list_gt_videos(gt_dir: Path) -> list[str]:
    names = sorted(f.name for f in gt_dir.iterdir() if f.suffix.lower() == ".mp4")
    if not names:
        raise FileNotFoundError(f"No .mp4 in {gt_dir}")
    return names


def count_video_frames(path: Path) -> int:
    try:
        from decord import VideoReader
    except ImportError as e:
        raise ImportError("decord is required for --min_frames validation") from e
    if path.stat().st_size == 0:
        return 0
    return len(VideoReader(str(path)))


def make_entry(gt_name: str, gt_dir: Path, normal_dir: Path, text: str) -> dict:
    stem = gt_name[:-4]
    gt_src = (gt_dir / gt_name).resolve()
    norm_src = (normal_dir / f"{stem}_normal.mp4").resolve()
    if not norm_src.is_file():
        raise FileNotFoundError(f"Missing control video: {norm_src}")
    return {
        "file_path": str(gt_src),
        "control_file_path": str(norm_src),
        "text": text,
        "type": "video",
        "id": stem,
    }


def main():
    args = parse_args()
    gt_dir = args.gt_dir.resolve()
    normal_dir = args.normal_dir.resolve()
    out_dir = args.out_dir.resolve()

    if not gt_dir.is_dir():
        raise FileNotFoundError(f"GT dir not found: {gt_dir}")
    if not normal_dir.is_dir():
        raise FileNotFoundError(f"Normal dir not found: {normal_dir}")

    names = list_gt_videos(gt_dir)
    test_names = names[: args.test_count]
    train_names = names[args.test_count :]

    out_dir.mkdir(parents=True, exist_ok=True)

    def build_meta(split_names: list[str]) -> list[dict]:
        entries = []
        skipped = []
        for n in split_names:
            stem = n[:-4]
            norm_src = normal_dir / f"{stem}_normal.mp4"
            if args.min_frames > 0:
                n_ctrl = count_video_frames(norm_src)
                if n_ctrl < args.min_frames:
                    skipped.append((n, n_ctrl))
                    continue
            entries.append(make_entry(n, gt_dir, normal_dir, args.text))
        if skipped:
            print(f"Skipped {len(skipped)} (control < {args.min_frames} frames), e.g. {skipped[:3]}")
        return entries

    train_meta = build_meta(train_names)
    test_meta = build_meta(test_names)

    train_path = out_dir / "metadata_train.json"
    test_path = out_dir / "metadata_test.json"
    train_path.write_text(json.dumps(train_meta, indent=2, ensure_ascii=False) + "\n")
    test_path.write_text(json.dumps(test_meta, indent=2, ensure_ascii=False) + "\n")

    (out_dir / "test_clip_names.txt").write_text("\n".join(test_names) + "\n")
    (out_dir / "train_clip_names.txt").write_text(
        f"# train={len(train_names)} test={len(test_names)} total={len(names)}\n"
        + "\n".join(train_names[:20])
        + ("\n...\n" if len(train_names) > 20 else "")
    )

    print(f"GT dir:     {gt_dir} ({len(names)} videos)")
    print(f"Normal dir: {normal_dir}")
    print(f"Train:      {len(train_meta)} -> {train_path}")
    print(f"Test:       {len(test_meta)} -> {test_path}")
    print("Paths: absolute only (no local video copy/symlink).")
    print(f"Test clips (sorted 1-{args.test_count}):")
    for n in test_names:
        print(f"  - {n}")


if __name__ == "__main__":
    main()
