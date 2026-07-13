#!/usr/bin/env python3
"""So sánh OCR.

Hai chế độ:
1. Legacy: ``--run-name`` so sánh scan gốc với trang processed.
2. Layout: ``--full-page-run`` và ``--columns-run`` so sánh OCR nguyên trang
   với OCR từng cột trên cùng ảnh processed.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean
from typing import Any


PART_RE = re.compile(r"^(.*)_p(\d{2})$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", required=True, help="Work ID, ví dụ HVH_001")
    parser.add_argument("--chapter", default="01")
    parser.add_argument("--run-name", help="Run original/processed kiểu cũ")
    parser.add_argument("--full-page-run", help="Run bước 4 dùng --ocr-layout full-page")
    parser.add_argument("--columns-run", help="Run bước 4 dùng --ocr-layout columns")
    parser.add_argument("--intermediate-root", default="data/intermediate")
    args = parser.parse_args()
    if bool(args.full_page_run) != bool(args.columns_run):
        parser.error("Phải truyền đồng thời --full-page-run và --columns-run")
    if not args.run_name and not args.full_page_run:
        parser.error("Cần --run-name hoặc cặp --full-page-run/--columns-run")
    return args


def load_success(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if data.get("status") == "success" else None


def normalized(text: str) -> str:
    return re.sub(r"\s+", "", text)


def cjk_count(text: str) -> int:
    return sum(
        1
        for char in text
        if "\u3400" <= char <= "\u4dbf"
        or "\u4e00" <= char <= "\u9fff"
        or "\uf900" <= char <= "\ufaff"
    )


def unknown_count(text: str) -> int:
    return text.count("〓") + text.count("\ufffd") + text.count("[UNK]")


def heuristic_indicator(original: dict[str, Any], processed: list[dict[str, Any]]) -> str:
    original_text = str(original.get("text", ""))
    processed_text = "\n".join(str(item.get("text", "")) for item in processed)
    original_cjk = cjk_count(original_text)
    processed_cjk = cjk_count(processed_text)
    original_conf = float(original.get("confidence", 0))
    processed_conf = mean(float(item.get("confidence", 0)) for item in processed)
    original_unknown = unknown_count(original_text)
    processed_unknown = unknown_count(processed_text)
    if processed_cjk > original_cjk * 1.05 and processed_conf >= original_conf - 5 and processed_unknown <= original_unknown:
        return "processed_tends_better"
    if original_cjk > processed_cjk * 1.05 and original_conf >= processed_conf - 5 and original_unknown <= processed_unknown:
        return "original_tends_better"
    return "inconclusive"


def compare_layout_runs(args: argparse.Namespace, chapter_id: str) -> int:
    base = Path(args.intermediate_root) / args.id / chapter_id / "ocr_runs"
    full_dir = base / args.full_page_run / "processed"
    columns_dir = base / args.columns_run / "processed"
    if not full_dir.is_dir() or not columns_dir.is_dir():
        print(f"LỖI: Thiếu run processed: {full_dir} / {columns_dir}", file=sys.stderr)
        return 1
    # Run columns thường là pilot nhỏ hơn run full-page. Dùng chính tập trang
    # của run columns làm phạm vi, tránh báo các trang ngoài pilot là missing.
    stems = sorted(path.stem for path in columns_dir.glob("*.json"))
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for stem in stems:
        full = load_success(full_dir / f"{stem}.json")
        columns = load_success(columns_dir / f"{stem}.json")
        if not full or not columns:
            missing.append(stem)
            continue
        full_text = str(full.get("text", ""))
        columns_text = str(columns.get("text", ""))
        full_conf = float(full.get("confidence", 0))
        columns_conf = float(columns.get("confidence", 0))
        full_cjk = cjk_count(full_text)
        columns_cjk = cjk_count(columns_text)
        full_unknown = unknown_count(full_text)
        columns_unknown = unknown_count(columns_text)
        if columns_cjk > full_cjk * 1.03 and columns_conf >= full_conf - 2 and columns_unknown <= full_unknown:
            indicator = "columns_tends_better"
        elif full_cjk > columns_cjk * 1.03 and full_conf >= columns_conf - 2 and full_unknown <= columns_unknown:
            indicator = "full_page_tends_better"
        else:
            indicator = "inconclusive"
        rows.append({
            "page_id": stem,
            "full_page_confidence": round(full_conf, 2),
            "columns_confidence": round(columns_conf, 2),
            "detected_columns": len(columns.get("columns", [])),
            "full_page_cjk_chars": full_cjk,
            "columns_cjk_chars": columns_cjk,
            "full_page_unknown": full_unknown,
            "columns_unknown": columns_unknown,
            "text_similarity": round(SequenceMatcher(None, normalized(full_text), normalized(columns_text)).ratio(), 4),
            "heuristic_indicator": indicator,
            "full_page_text": full_text,
            "columns_text": columns_text,
            "manual_choice": "",
            "manual_note": "",
        })
    if not rows:
        print("LỖI: Không có cặp page OCR thành công", file=sys.stderr)
        return 1
    comparison_name = f"{args.full_page_run}_vs_{args.columns_run}"
    report_dir = Path(args.intermediate_root) / args.id / chapter_id / "ocr_comparison" / comparison_name
    report_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = report_dir / "comparison.tsv"
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    indicators: dict[str, int] = {}
    for row in rows:
        key = str(row["heuristic_indicator"])
        indicators[key] = indicators.get(key, 0) + 1
    full_mean = mean(float(row["full_page_confidence"]) for row in rows)
    columns_mean = mean(float(row["columns_confidence"]) for row in rows)
    full_cjk_total = sum(int(row["full_page_cjk_chars"]) for row in rows)
    columns_cjk_total = sum(int(row["columns_cjk_chars"]) for row in rows)
    columns_wins = indicators.get("columns_tends_better", 0)
    full_wins = indicators.get("full_page_tends_better", 0)
    recommendation = "manual_review"
    if columns_wins > full_wins and columns_mean >= full_mean - 1 and columns_cjk_total >= full_cjk_total:
        recommendation = "columns"
    elif full_wins > columns_wins and full_mean >= columns_mean - 1 and full_cjk_total >= columns_cjk_total:
        recommendation = "full-page"
    summary = {
        "chapter_id": chapter_id,
        "comparison": comparison_name,
        "full_page_run": args.full_page_run,
        "columns_run": args.columns_run,
        "compared_pages": len(rows),
        "missing_or_failed_pages": missing,
        "heuristic_indicators": indicators,
        "aggregate_metrics": {
            "full_page_mean_confidence": round(full_mean, 3),
            "columns_mean_confidence": round(columns_mean, 3),
            "full_page_total_cjk_chars": full_cjk_total,
            "columns_total_cjk_chars": columns_cjk_total,
            "mean_text_similarity": round(mean(float(row["text_similarity"]) for row in rows), 4),
        },
        "recommended_layout": recommendation,
        "important_note": "Heuristic, không phải accuracy; muốn kết luận cần ground truth/CER.",
    }
    summary_path = report_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Đã so sánh {len(rows)} trang: full-page vs columns")
    print(f"Khuyến nghị heuristic: {recommendation}")
    print(f"TSV: {tsv_path}")
    print(f"Tổng kết: {summary_path}")
    return 0


def main() -> int:
    args = parse_args()
    chapter_id = f"{args.id}_{args.chapter}"
    if args.full_page_run:
        return compare_layout_runs(args, chapter_id)
    run_root = Path(args.intermediate_root) / args.id / chapter_id / "ocr_runs" / args.run_name
    original_dir = run_root / "original"
    processed_dir = run_root / "processed"
    if not original_dir.is_dir() or not processed_dir.is_dir():
        print(f"LỖI: Run chưa có đủ original/processed: {run_root}", file=sys.stderr)
        return 1

    originals = {path.stem: load_success(path) for path in sorted(original_dir.glob("*.json"))}
    processed_groups: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for path in sorted(processed_dir.glob("*.json")):
        match = PART_RE.fullmatch(path.stem)
        data = load_success(path)
        if match and data:
            processed_groups.setdefault(match.group(1), []).append((int(match.group(2)), data))

    rows = []
    missing = []
    for scan_key, original in originals.items():
        parts = sorted(processed_groups.get(scan_key, []), key=lambda item: item[0])
        if not original or not parts:
            missing.append(scan_key)
            continue
        processed = [item[1] for item in parts]
        original_text = str(original.get("text", ""))
        processed_text = "\n".join(str(item.get("text", "")) for item in processed)
        rows.append(
            {
                "scan_id": scan_key,
                "processed_parts": len(processed),
                "original_confidence": original.get("confidence", 0),
                "processed_mean_confidence": round(mean(float(item.get("confidence", 0)) for item in processed), 2),
                "original_cjk_chars": cjk_count(original_text),
                "processed_cjk_chars": cjk_count(processed_text),
                "original_unknown": unknown_count(original_text),
                "processed_unknown": unknown_count(processed_text),
                "text_similarity": round(SequenceMatcher(None, normalized(original_text), normalized(processed_text)).ratio(), 4),
                "heuristic_indicator": heuristic_indicator(original, processed),
                "original_text": original_text,
                "processed_text": processed_text,
                "manual_choice": "",
                "manual_note": "",
            }
        )

    if not rows:
        print("LỖI: Không có cặp OCR thành công để so sánh", file=sys.stderr)
        return 1

    report_dir = Path(args.intermediate_root) / args.id / chapter_id / "ocr_comparison" / args.run_name
    report_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = report_dir / "comparison.tsv"
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    indicators: dict[str, int] = {}
    for row in rows:
        key = str(row["heuristic_indicator"])
        indicators[key] = indicators.get(key, 0) + 1
    original_mean_confidence = round(mean(float(row["original_confidence"]) for row in rows), 3)
    processed_mean_confidence = round(mean(float(row["processed_mean_confidence"]) for row in rows), 3)
    original_total_cjk = sum(int(row["original_cjk_chars"]) for row in rows)
    processed_total_cjk = sum(int(row["processed_cjk_chars"]) for row in rows)
    processed_cjk_wins = sum(
        int(row["processed_cjk_chars"]) > int(row["original_cjk_chars"]) for row in rows
    )
    processed_confidence_wins = sum(
        float(row["processed_mean_confidence"]) > float(row["original_confidence"]) for row in rows
    )
    processed_has_more_text = processed_total_cjk >= original_total_cjk * 1.01
    processed_confidence_not_worse = processed_mean_confidence >= original_mean_confidence - 1.0
    recommended_source = (
        "processed" if processed_has_more_text and processed_confidence_not_worse else "manual_review"
    )
    summary = {
        "chapter_id": chapter_id,
        "run_name": args.run_name,
        "compared_scans": len(rows),
        "missing_or_failed_scans": missing,
        "heuristic_indicators": indicators,
        "aggregate_metrics": {
            "original_mean_confidence": original_mean_confidence,
            "processed_mean_confidence": processed_mean_confidence,
            "original_total_cjk_chars": original_total_cjk,
            "processed_total_cjk_chars": processed_total_cjk,
            "processed_cjk_wins": processed_cjk_wins,
            "processed_confidence_wins": processed_confidence_wins,
            "mean_text_similarity": round(mean(float(row["text_similarity"]) for row in rows), 4),
        },
        "recommended_source": recommended_source,
        "recommendation_basis": (
            "Processed có >=1% ký tự Hán hơn và confidence trung bình không thấp hơn quá 1 điểm."
            if recommended_source == "processed"
            else "Chỉ số tự động chưa đủ để chọn nguồn."
        ),
        "important_note": "Chỉ báo tham khảo; cần đối chiếu ảnh/ground truth và điền manual_choice.",
    }
    summary_path = report_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Đã so sánh {len(rows)} scan")
    print(f"TSV: {tsv_path}")
    print(f"Tổng kết: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
