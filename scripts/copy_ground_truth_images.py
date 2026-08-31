#!/usr/bin/env python3
"""Copy ảnh được tham chiếu trong ground_truth.jsonl sang data/ground_truth_images/.

Mục đích: chỉ đưa vào git những ảnh thực sự có nhãn ground truth,
thay vì toàn bộ 69 K ảnh thô trong data/input/Images/.

Usage:
    python scripts/facebook/copy_ground_truth_images.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GT     = ROOT / "data" / "output" / "DeepSeek_ground_truth" / "ground_truth.jsonl"
DEFAULT_SRC    = ROOT / "data" / "input" / "Images"
DEFAULT_DST    = ROOT / "data" / "ground_truth_images"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ground-truth", type=Path, default=DEFAULT_GT,
                   help="File ground_truth.jsonl (mặc định: %(default)s)")
    p.add_argument("--src", type=Path, default=DEFAULT_SRC,
                   help="Thư mục ảnh nguồn (mặc định: %(default)s)")
    p.add_argument("--dst", type=Path, default=DEFAULT_DST,
                   help="Thư mục ảnh đích để commit (mặc định: %(default)s)")
    p.add_argument("--dry-run", action="store_true",
                   help="In kế hoạch mà không copy")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.ground_truth.is_file():
        print(f"LỖI: không tìm thấy {args.ground_truth}", file=sys.stderr)
        return 2
    if not args.src.is_dir():
        print(f"LỖI: thư mục nguồn không tồn tại: {args.src}", file=sys.stderr)
        return 2

    # Thu thập tên file từ ground_truth.jsonl
    filenames: set[str] = set()
    with args.ground_truth.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            img = row.get("image", "")
            if img:
                filenames.add(Path(img).name)

    print(f"Ground truth references {len(filenames)} unique images")

    # Kiểm tra ảnh nào tồn tại trong src
    found: list[tuple[Path, Path]] = []
    missing: list[str] = []
    for name in sorted(filenames):
        src_file = args.src / name
        if src_file.is_file():
            found.append((src_file, args.dst / name))
        else:
            missing.append(name)

    print(f"Found in {args.src.name}/: {len(found)}")
    print(f"Missing (skipped)       : {len(missing)}")

    if args.dry_run:
        print("\n[dry-run] Would copy to:", args.dst)
        for src_f, dst_f in found[:5]:
            print(f"  {src_f.name} → {dst_f}")
        if len(found) > 5:
            print(f"  ... and {len(found)-5} more")
        return 0

    # Copy
    args.dst.mkdir(parents=True, exist_ok=True)
    copied = skipped = 0
    for src_f, dst_f in found:
        if dst_f.exists():
            skipped += 1
            continue
        shutil.copy2(src_f, dst_f)
        copied += 1

    print(f"\nCopied  : {copied}")
    print(f"Skipped (already exist): {skipped}")
    print(f"Dest    : {args.dst}")
    if missing:
        print(f"\nWarning: {len(missing)} images missing from source — not in ground_truth_images/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
