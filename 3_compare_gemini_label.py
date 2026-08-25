#!/usr/bin/env python3
"""Chia OCR Gemini thành nhóm gần giống/không giống caption gốc.

Không sao chép ảnh. Mỗi record giữ ``local_path`` để DeepSeek và PaddleOCR
có thể đọc lại ảnh gốc ở các bước sau.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "mrDuc_data"
DEFAULT_LABELS = DATA_DIR / "valid.jsonl"
DEFAULT_GEMINI = DATA_DIR / "facebook_posts_ocr.jsonl"
DEFAULT_IMAGES = DATA_DIR / "Images"
DEFAULT_SAME = ROOT / "data" / "output" / "Gemini_same_Label"
DEFAULT_DIFF = ROOT / "data" / "output" / "Gemini_diff_Label"

URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.I)
MENTION_RE = re.compile(r"(?<!\w)[@#][^\s#@]+")
BOILERPLATE_PATTERNS = (
    re.compile(r"今天的練習\s*\d*", re.I),
    re.compile(r"今日練習\s*\d*", re.I),
    re.compile(r"感謝書友欣賞|感谢书友欣赏|謝謝欣賞|谢谢欣赏"),
    re.compile(r"歡迎關注|欢迎关注"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--gemini", type=Path, default=DEFAULT_GEMINI)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--same-dir", type=Path, default=DEFAULT_SAME)
    parser.add_argument("--diff-dir", type=Path, default=DEFAULT_DIFF)
    parser.add_argument("--threshold", type=float, default=0.58)
    parser.add_argument("--min-common-han", type=int, default=4)
    parser.add_argument("--limit", type=int, help="Chỉ phân tích N output Gemini đầu tiên")
    parser.add_argument("--dry-run", action="store_true", help="Tính thống kê, không ghi output")
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold phải nằm trong 0..1")
    if args.min_common_han < 1:
        parser.error("--min-common-han phải lớn hơn 0")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit phải lớn hơn 0")
    return args


def read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON lỗi tại {path}:{line_number}: {exc}") from exc
            if isinstance(item, dict):
                yield line_number, item


def is_han(char: str) -> bool:
    code = ord(char)
    return (
        0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
        or 0x20000 <= code <= 0x3134F
    )


def normalize(text: str, han_only: bool = False) -> str:
    value = unicodedata.normalize("NFKC", text or "").casefold()
    if han_only:
        return "".join(char for char in value if is_han(char))
    return "".join(char for char in value if char.isalnum() or is_han(char))


def caption_core(text: str) -> str:
    value = unicodedata.normalize("NFKC", text or "")
    value = URL_RE.sub(" ", value)
    value = MENTION_RE.sub(" ", value)
    for pattern in BOILERPLATE_PATTERNS:
        value = pattern.sub(" ", value)
    return re.sub(r"\s+", " ", value).strip()


def counter_f1(left: str, right: str) -> tuple[float, int]:
    if not left or not right:
        return 0.0, 0
    a, b = Counter(left), Counter(right)
    common = sum((a & b).values())
    precision = common / len(right)
    recall = common / len(left)
    return (2 * precision * recall / (precision + recall) if precision + recall else 0.0), common


def ngram_dice(left: str, right: str, n: int = 2) -> float:
    if len(left) < n or len(right) < n:
        return 1.0 if left == right and left else 0.0
    a = Counter(left[i : i + n] for i in range(len(left) - n + 1))
    b = Counter(right[i : i + n] for i in range(len(right) - n + 1))
    common = sum((a & b).values())
    return 2 * common / (sum(a.values()) + sum(b.values()))


def compare_texts(label: str, gemini_text: str, threshold: float, min_common_han: int) -> dict[str, Any]:
    core = caption_core(label)
    label_han = normalize(core, han_only=True) or normalize(label, han_only=True)
    gemini_han = normalize(gemini_text, han_only=True)
    # Nếu không có chữ Hán, vẫn cho phép so sánh chữ/số đã chuẩn hóa.
    left = label_han if label_han and gemini_han else normalize(core or label)
    right = gemini_han if label_han and gemini_han else normalize(gemini_text)

    if not right:
        return {
            "similarity_score": 0.0, "sequence_ratio": 0.0, "character_f1": 0.0,
            "bigram_dice": 0.0, "ocr_coverage": 0.0, "label_coverage": 0.0,
            "common_characters": 0, "common_han_characters": 0,
            "classification": "diff", "reason": "gemini_blank",
            "label_compare_text": left, "gemini_compare_text": right,
        }

    matcher = SequenceMatcher(None, left, right, autojunk=False)
    ordered_common = sum(block.size for block in matcher.get_matching_blocks())
    sequence_ratio = matcher.ratio()
    char_f1, common = counter_f1(left, right)
    bigram = ngram_dice(left, right)
    ocr_coverage = ordered_common / len(right) if right else 0.0
    label_coverage = ordered_common / len(left) if left else 0.0
    short_coverage = ordered_common / min(len(left), len(right)) if left and right else 0.0
    containment = bool(left and right and (left in right or right in left))
    score = 0.30 * sequence_ratio + 0.25 * char_f1 + 0.25 * short_coverage + 0.20 * bigram
    common_han = sum((Counter(label_han) & Counter(gemini_han)).values())

    shorter = min(len(left), len(right))
    if shorter <= 3:
        same = left == right
        reason = "short_exact_match" if same else "short_text_not_exact"
    elif containment and short_coverage >= 0.90 and common_han >= min(min_common_han, shorter):
        same, reason = True, "one_text_contains_the_other"
    elif common_han >= min_common_han and max(ocr_coverage, label_coverage) >= 0.72 and char_f1 >= 0.55:
        same, reason = True, "high_directional_coverage"
    elif score >= threshold and common_han >= min_common_han:
        same, reason = True, "combined_score_above_threshold"
    else:
        same, reason = False, "insufficient_textual_agreement"

    return {
        "similarity_score": round(score, 6),
        "sequence_ratio": round(sequence_ratio, 6),
        "character_f1": round(char_f1, 6),
        "bigram_dice": round(bigram, 6),
        "ocr_coverage": round(ocr_coverage, 6),
        "label_coverage": round(label_coverage, 6),
        "common_characters": common,
        "common_han_characters": common_han,
        "classification": "same" if same else "diff",
        "reason": reason,
        "label_compare_text": left,
        "gemini_compare_text": right,
    }


def gemini_text(regions: Any) -> str:
    if not isinstance(regions, list):
        return ""
    return "\n".join(
        str(region.get("text", "")).strip()
        for region in regions
        if isinstance(region, dict) and str(region.get("text", "")).strip()
    )


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower, upper = math.floor(index), math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def main() -> int:
    args = parse_args()
    if not args.labels.is_file() or not args.gemini.is_file():
        print("LỖI: thiếu valid.jsonl hoặc output Gemini", file=sys.stderr)
        return 2

    labels = {
        str(item.get("post_id", item.get("id", ""))): str(item.get("label", ""))
        for _, item in read_jsonl(args.labels)
    }
    rows = list(read_jsonl(args.gemini))
    if args.limit is not None:
        rows = rows[: args.limit]

    outputs: dict[str, list[dict[str, Any]]] = {"same": [], "diff": []}
    missing_label = duplicates = 0
    seen: set[str] = set()
    scores: list[float] = []
    reason_counts: Counter[str] = Counter()

    for _, source in rows:
        image_key = str(source.get("image", ""))
        if not image_key or image_key in seen:
            duplicates += bool(image_key in seen)
            continue
        seen.add(image_key)
        post_id = str(source.get("post_id", source.get("id", "")))
        if post_id not in labels:
            missing_label += 1
            continue
        label = labels[post_id]
        regions = source.get("gemini", [])
        ocr_text = gemini_text(regions)
        metrics = compare_texts(label, ocr_text, args.threshold, args.min_common_han)
        group = metrics["classification"]
        local_path = args.images_dir / Path(image_key).name
        record = {
            "post_id": post_id,
            "image": image_key,
            "local_path": str(local_path.resolve()),
            "image_exists": local_path.is_file(),
            "label": label,
            "label_core": caption_core(label),
            "gemini_text": ocr_text,
            "gemini_regions": regions,
            "gemini_model": source.get("ocr_model", ""),
            "comparison": metrics,
        }
        outputs[group].append(record)
        scores.append(float(metrics["similarity_score"]))
        reason_counts[str(metrics["reason"])] += 1

    summary = {
        "gemini_records": len(rows),
        "classified": len(outputs["same"]) + len(outputs["diff"]),
        "same": len(outputs["same"]),
        "diff": len(outputs["diff"]),
        "same_percent": round(100 * len(outputs["same"]) / max(1, len(scores)), 3),
        "diff_percent": round(100 * len(outputs["diff"]) / max(1, len(scores)), 3),
        "missing_label": missing_label,
        "duplicate_images_skipped": duplicates,
        "threshold": args.threshold,
        "min_common_han": args.min_common_han,
        "score_percentiles": {
            "p10": round(percentile(scores, 0.10), 6),
            "p25": round(percentile(scores, 0.25), 6),
            "p50": round(percentile(scores, 0.50), 6),
            "p75": round(percentile(scores, 0.75), 6),
            "p90": round(percentile(scores, 0.90), 6),
        },
        "reasons": dict(reason_counts),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0

    for group, directory in (("same", args.same_dir), ("diff", args.diff_dir)):
        directory.mkdir(parents=True, exist_ok=True)
        jsonl = directory / "records.jsonl"
        temp = jsonl.with_suffix(".jsonl.tmp")
        with temp.open("w", encoding="utf-8") as handle:
            for record in outputs[group]:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(temp, jsonl)
        with (directory / "comparison.tsv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["image", "score", "reason", "label", "gemini_text", "local_path"])
            for record in outputs[group]:
                writer.writerow([
                    record["image"], record["comparison"]["similarity_score"],
                    record["comparison"]["reason"], record["label"],
                    record["gemini_text"], record["local_path"],
                ])
        write_json_atomic(directory / "summary.json", {**summary, "this_group": group, "records": len(outputs[group])})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
