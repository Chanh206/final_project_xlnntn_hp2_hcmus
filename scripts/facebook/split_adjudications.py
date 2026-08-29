#!/usr/bin/env python3
"""Tách adjudications.jsonl thành valid và invalid.

Điều kiện INVALID (thỏa một trong các điều kiện sau):
  1. confidence < --min-confidence (mặc định 0.75)
  2. ground_truth_text chỉ chứa "Reels" (thumbnail video Reels)
  3. selected_source == "blank"    → ảnh không có chữ
  4. selected_source == "uncertain" → ảnh mờ, không đọc được
  5. "image_unclear" trong reason_codes → ảnh kém chất lượng

Mọi record không thỏa điều kiện trên → VALID.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT  = ROOT / "data" / "output" / "DeepSeek_ground_truth" / "adjudications.jsonl"
DEFAULT_VALID  = ROOT / "data" / "output" / "DeepSeek_ground_truth" / "adjudications_valid.jsonl"
DEFAULT_INVALID = ROOT / "data" / "output" / "DeepSeek_ground_truth" / "adjudications_invalid.jsonl"

# Các text coi là "Reels thumbnail" — so sánh sau khi strip + lower
REELS_TEXTS = {"reels"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",   type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--valid",   type=Path, default=DEFAULT_VALID)
    parser.add_argument("--invalid", type=Path, default=DEFAULT_INVALID)
    parser.add_argument(
        "--min-confidence", type=float, default=0.75,
        help="Ngưỡng confidence tối thiểu để vào valid (mặc định: 0.75)",
    )
    return parser.parse_args()


def classify(row: dict, min_confidence: float) -> tuple[bool, list[str]]:
    """Trả về (is_invalid, reasons)."""
    decision = row.get("decision", {})
    confidence     = float(decision.get("confidence", 0))
    selected_source = str(decision.get("selected_source", ""))
    ground_truth   = str(decision.get("ground_truth_text", "")).strip()
    reason_codes   = decision.get("reason_codes", [])

    reasons: list[str] = []

    if confidence < min_confidence:
        reasons.append(f"low_confidence({confidence:.3f}<{min_confidence})")

    if ground_truth.lower() in REELS_TEXTS:
        reasons.append("reels_thumbnail")

    if selected_source == "blank":
        reasons.append("no_visible_text(blank)")

    if selected_source == "uncertain":
        reasons.append("uncertain_source")

    if "image_unclear" in reason_codes:
        reasons.append("image_unclear")

    return bool(reasons), reasons


def main() -> int:
    args = parse_args()

    if not args.input.is_file():
        print(f"LỖI: không tìm thấy {args.input}", file=sys.stderr)
        return 2

    args.valid.parent.mkdir(parents=True, exist_ok=True)
    args.invalid.parent.mkdir(parents=True, exist_ok=True)

    total = valid_count = invalid_count = 0
    reason_summary: dict[str, int] = {}

    with args.input.open(encoding="utf-8") as fin, \
         args.valid.open("w", encoding="utf-8") as fv, \
         args.invalid.open("w", encoding="utf-8") as fi:

        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            row = json.loads(line)
            is_invalid, reasons = classify(row, args.min_confidence)

            if is_invalid:
                invalid_count += 1
                row["_invalid_reasons"] = reasons
                fi.write(json.dumps(row, ensure_ascii=False) + "\n")
                for r in reasons:
                    # Gộp các reason có cùng tiền tố
                    key = r.split("(")[0]
                    reason_summary[key] = reason_summary.get(key, 0) + 1
            else:
                valid_count += 1
                fv.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Tổng input : {total}")
    print(f"Valid      : {valid_count} ({valid_count/total*100:.1f}%) → {args.valid}")
    print(f"Invalid    : {invalid_count} ({invalid_count/total*100:.1f}%) → {args.invalid}")
    print()
    print("Lý do invalid (có thể chồng nhau):")
    for reason, count in sorted(reason_summary.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
