#!/usr/bin/env python3
"""Resample fashion GT + normal/depth videos to 16 fps under datasets/fashion_vace/.

Originals are never modified. Layout:
  datasets/fashion_vace/videos_16fps/train/gt/{id}.mp4
  datasets/fashion_vace/videos_16fps/train/normal/{id}_normal.mp4
  datasets/fashion_vace/videos_16fps/train/depth/{id}_depth.mp4
  datasets/fashion_vace/videos_16fps/test/gt/{id}.mp4
  datasets/fashion_vace/videos_16fps/test/normal/{id}_normal.mp4
  datasets/fashion_vace/videos_16fps/test/depth/{id}_depth.mp4

Writes metadata_train_16fps.json and metadata_test_16fps.json.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_META_DIR = ROOT / "datasets" / "fashion_vace"
DEFAULT_OUT_SUBDIR = "videos_16fps"
DEFAULT_FPS = 16


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--meta_dir", type=Path, default=DEFAULT_META_DIR)
    p.add_argument("--fps", type=int, default=DEFAULT_FPS)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--crf", type=int, default=18)
    p.add_argument("--force", action="store_true", help="Re-encode even if output exists.")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument(
        "--reorganize_only",
        action="store_true",
        help="Move videos_16fps/gt|normal -> train|test (no ffmpeg).",
    )
    return p.parse_args()


def resolve_path(raw: str, project_root: Path) -> Path:
    p = Path(raw)
    if not p.is_absolute():
        p = (project_root / p).resolve()
    return p.resolve()


def split_dirs(out_root: Path, split: str) -> tuple[Path, Path, Path]:
    return out_root / split / "gt", out_root / split / "normal", out_root / split / "depth"


def convert_one(src: Path, dst: Path, fps: int, crf: int, force: bool) -> tuple[str, str]:
    if not src.is_file():
        return "missing", str(src)
    if dst.is_file() and not force and dst.stat().st_mtime >= src.stat().st_mtime:
        return "skip", str(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-vf",
        f"fps={fps}",
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-preset",
        "medium",
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(dst),
    ]
    subprocess.run(cmd, check=True)
    return "ok", str(dst)


def _worker(args: tuple) -> tuple[str, str]:
    return convert_one(*args)


def collect_jobs(
    entries: list[dict],
    project_root: Path,
    split: str,
    out_root: Path,
    fps: int,
    crf: int,
    force: bool,
) -> list[tuple]:
    out_gt, out_normal, out_depth = split_dirs(out_root, split)
    jobs = []
    for e in entries:
        vid = e["id"]
        gt_src = resolve_path(e["file_path"], project_root)
        norm_src = resolve_path(e.get("normal_file_path", e["control_file_path"]), project_root)
        depth_src = resolve_path(e["depth_file_path"], project_root)
        jobs.append((gt_src, out_gt / f"{vid}.mp4", fps, crf, force))
        jobs.append((norm_src, out_normal / f"{vid}_normal.mp4", fps, crf, force))
        jobs.append((depth_src, out_depth / f"{vid}_depth.mp4", fps, crf, force))
    return jobs


def rewrite_meta(entries: list[dict], out_root: Path, split: str) -> list[dict]:
    out_gt, out_normal, out_depth = split_dirs(out_root, split)
    out = []
    for e in entries:
        vid = e["id"]
        ne = dict(e)
        ne["file_path"] = str((out_gt / f"{vid}.mp4").resolve())
        ne["control_file_path"] = str((out_normal / f"{vid}_normal.mp4").resolve())
        ne["normal_file_path"] = str((out_normal / f"{vid}_normal.mp4").resolve())
        ne["depth_file_path"] = str((out_depth / f"{vid}_depth.mp4").resolve())
        out.append(ne)
    return out


def reorganize_legacy(out_root: Path, train_meta: list[dict], test_meta: list[dict]) -> None:
    """Move flat videos_16fps/gt|normal into train/ and test/."""
    legacy_gt = out_root / "gt"
    legacy_normal = out_root / "normal"
    if not legacy_gt.is_dir() and not legacy_normal.is_dir():
        return

    def move_clip(split: str, vid: str) -> None:
        out_gt, out_normal, _ = split_dirs(out_root, split)
        out_gt.mkdir(parents=True, exist_ok=True)
        out_normal.mkdir(parents=True, exist_ok=True)
        for legacy_dir, name, dst_dir in (
            (legacy_gt, f"{vid}.mp4", out_gt),
            (legacy_normal, f"{vid}_normal.mp4", out_normal),
        ):
            src = legacy_dir / name
            dst = dst_dir / name
            if not src.is_file():
                if dst.is_file():
                    return
                raise FileNotFoundError(src)
            if dst.exists():
                src.unlink()
            else:
                src.rename(dst)

    for e in train_meta:
        move_clip("train", e["id"])
    for e in test_meta:
        move_clip("test", e["id"])

    for d in (legacy_gt, legacy_normal):
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()
    print("Reorganized legacy gt/normal -> train/ and test/")


def main():
    args = parse_args()
    meta_dir = args.meta_dir.resolve()
    out_root = meta_dir / DEFAULT_OUT_SUBDIR

    train_path = meta_dir / "metadata_train.json"
    test_path = meta_dir / "metadata_test.json"
    if not train_path.is_file():
        sys.exit(f"Missing {train_path}")

    train_meta = json.loads(train_path.read_text())
    test_meta = json.loads(test_path.read_text()) if test_path.is_file() else []

    if args.reorganize_only:
        reorganize_legacy(out_root, train_meta, test_meta)
    else:
        jobs = []
        jobs.extend(collect_jobs(train_meta, ROOT, "train", out_root, args.fps, args.crf, args.force))
        jobs.extend(collect_jobs(test_meta, ROOT, "test", out_root, args.fps, args.crf, args.force))
        print(f"Videos to process: {len(jobs)} (fps={args.fps})")
        print(f"  train: {len(train_meta)} clips x3")
        print(f"  test:  {len(test_meta)} clips x3")
        print(f"Output: {out_root}/{{train,test}}/{{gt,normal,depth}}")

        if args.dry_run:
            for j in jobs[:4]:
                print(f"  {j[0]} -> {j[1]}")
            return

        stats = {"ok": 0, "skip": 0, "missing": 0, "fail": 0}
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_worker, j): j for j in jobs}
            for i, fut in enumerate(as_completed(futs), 1):
                j = futs[fut]
                try:
                    status, _ = fut.result()
                    stats[status] = stats.get(status, 0) + 1
                except Exception as exc:
                    stats["fail"] += 1
                    print(f"FAIL {j[0]}: {exc}", file=sys.stderr)
                if i % 50 == 0 or i == len(jobs):
                    print(f"Progress {i}/{len(jobs)} {stats}")
        print(f"Encode done. {stats}")
        reorganize_legacy(out_root, [], [])

    train_out = meta_dir / "metadata_train_16fps.json"
    test_out = meta_dir / "metadata_test_16fps.json"
    train_out.write_text(
        json.dumps(rewrite_meta(train_meta, out_root, "train"), indent=2, ensure_ascii=False) + "\n"
    )
    if test_meta:
        test_out.write_text(
            json.dumps(rewrite_meta(test_meta, out_root, "test"), indent=2, ensure_ascii=False) + "\n"
        )

    print(f"Wrote {train_out} ({len(train_meta)} entries)")
    if test_meta:
        print(f"Wrote {test_out} ({len(test_meta)} entries)")


if __name__ == "__main__":
    main()
