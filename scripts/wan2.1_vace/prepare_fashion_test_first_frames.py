#!/usr/bin/env python3
"""Extract GT first frames for fashion_vace metadata_test.json.

Writes PNGs to datasets/fashion_vace/first_frames/{id}.png and adds
ref_image_path to each entry in metadata_test.json.

Usage:
  python scripts/wan2.1_vace/prepare_fashion_test_first_frames.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_META = ROOT / "datasets/fashion_vace/metadata_test.json"
DEFAULT_OUT_DIR = ROOT / "datasets/fashion_vace/first_frames"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", type=Path, default=DEFAULT_META)
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--update_metadata", action="store_true", default=True)
    return p.parse_args()


def resolve_gt_path(file_path: str) -> Path:
    p = Path(file_path)
    if p.is_file():
        return p.resolve()
    candidate = (ROOT / p).resolve()
    if candidate.is_file():
        return candidate
    # metadata relative path ../NormalCrafter/...
    candidate = (ROOT / ".." / p).resolve()
    if candidate.is_file():
        return candidate
    return p.resolve()


def extract_first_frame(video_path: Path) -> Image.Image:
    try:
        from decord import VideoReader

        vr = VideoReader(str(video_path))
        if len(vr) == 0:
            raise ValueError(f"Empty video: {video_path}")
        return Image.fromarray(vr[0].asnumpy()).convert("RGB")
    except ImportError:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise ValueError(f"Cannot read first frame: {video_path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame).convert("RGB")


def main():
    args = parse_args()
    meta_path = args.metadata.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(meta_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    updated = []
    for sample in samples:
        sample_id = sample["id"]
        gt_path = resolve_gt_path(sample["file_path"])
        if not gt_path.is_file():
            print(f"skip missing GT: {gt_path}")
            updated.append(sample)
            continue

        out_png = out_dir / f"{sample_id}.png"
        img = extract_first_frame(gt_path)
        img.save(out_png)
        print(f"saved {out_png}")

        ref_rel = str(out_png.relative_to(ROOT))
        entry = dict(sample)
        entry["ref_image_path"] = ref_rel
        updated.append(entry)

    if args.update_metadata:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(updated, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"updated {meta_path}")


if __name__ == "__main__":
    main()
