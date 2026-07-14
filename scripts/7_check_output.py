#!/usr/bin/env python3
"""Kiểm tra hai output bắt buộc: ``_raw.txt`` và ``_seg.tsv``."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

CHAPTER_ID_RE = re.compile(r"^(HVH_\d{3})_(\d{2})$")


def read_utf8(path: Path) -> tuple[str | None, list[str]]:
    try:
        return path.read_text(encoding="utf-8"), []
    except UnicodeDecodeError as exc:
        return None, [f"Không phải UTF-8 hợp lệ: {exc}"]


def check_raw(path: Path) -> list[str]:
    if not path.is_file():
        return ["File không tồn tại"]
    text, issues = read_utf8(path)
    if text is not None:
        if not text.strip():
            issues.append("File rỗng")
        if "\ufffd" in text:
            issues.append("Có ký tự thay thế Unicode U+FFFD")
    return issues


def check_seg(path: Path, chapter_id: str) -> tuple[int, str, list[str]]:
    if not path.is_file():
        return 0, "", ["File không tồn tại"]
    text, issues = read_utf8(path)
    if text is None:
        return 0, "", issues

    seen_ids: set[str] = set()
    sentences: list[str] = []
    sentence_id_re = re.compile(rf"^{re.escape(chapter_id)}_(\d{{6}})$")
    total = 0
    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            issues.append(f"Dòng {lineno}: dòng rỗng")
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            issues.append(f"Dòng {lineno}: phải có đúng 2 cột ngăn bằng tab, hiện có {len(parts)}")
            continue

        sentence_id, sentence = parts
        match = sentence_id_re.fullmatch(sentence_id)
        if not match:
            issues.append(f"Dòng {lineno}: sentence_id sai: {sentence_id!r}")
        elif int(match.group(1)) != total + 1:
            issues.append(
                f"Dòng {lineno}: thứ tự ID là {match.group(1)}, "
                f"mong đợi {total + 1:06d}"
            )
        if sentence_id in seen_ids:
            issues.append(f"Dòng {lineno}: sentence_id trùng: {sentence_id}")
        seen_ids.add(sentence_id)
        if not sentence.strip():
            issues.append(f"Dòng {lineno}: câu rỗng")
        if "\ufffd" in sentence:
            issues.append(f"Dòng {lineno}: có ký tự thay thế Unicode U+FFFD")
        sentences.append(sentence)
        total += 1

    if total == 0:
        issues.append("Không có câu hợp lệ")
    return total, "".join(sentences), issues


def normalize_coverage(text: str) -> str:
    """Bỏ whitespace để đối chiếu nội dung raw với toàn bộ các câu seg."""
    return "".join(text.split())


def print_result(name: str, issues: list[str], detail: str = "") -> None:
    print(f"\n{name}{detail}")
    if not issues:
        print("  ✅ Hợp lệ")
        return
    for issue in issues:
        print(f"  ⚠️  {issue}")


def chapter_in_catalog(path: Path, chapter_id: str) -> bool:
    if not path.is_file():
        return False
    with path.open(encoding="utf-8", newline="") as handle:
        return any(row.get("chapter_id") == chapter_id for row in csv.DictReader(handle))


def run_check(
    chapter_id: str,
    output_root: Path,
    catalog_path: Path,
    allow_corrected: bool = False,
) -> int:
    match = CHAPTER_ID_RE.fullmatch(chapter_id)
    if not match:
        print("ID phải có dạng HVH_NNN_CC, ví dụ HVH_001_01")
        return 2
    if not chapter_in_catalog(catalog_path, chapter_id):
        print(f"Chapter ID không có trong catalog: {chapter_id}")
        return 2

    work_id = match.group(1)
    output_dir = output_root / work_id / chapter_id
    raw_path = output_dir / f"{chapter_id}_raw.txt"
    seg_path = output_dir / f"{chapter_id}_seg.tsv"
    print(f"=== Kiểm tra output: {chapter_id} ===")
    print(f"Thư mục: {output_dir}")

    raw_issues = check_raw(raw_path)
    sentence_count, seg_content, seg_issues = check_seg(seg_path, chapter_id)
    raw_text, _ = read_utf8(raw_path) if raw_path.is_file() else (None, [])
    coverage_changed = (
        raw_text is not None
        and normalize_coverage(raw_text) != normalize_coverage(seg_content)
    )
    if coverage_changed and not allow_corrected:
        seg_issues.append(
            "Nội dung các câu không khớp raw; nếu đây là bản LLM đã hiệu chỉnh, "
            "chạy lại với --allow-corrected"
        )
    print_result(raw_path.name, raw_issues)
    print_result(seg_path.name, seg_issues, f" — {sentence_count} câu")
    if coverage_changed and allow_corrected:
        print("  ℹ️  Seg khác raw theo chế độ cho phép hiệu chỉnh; cần giữ báo cáo bước 8.")

    all_issues = raw_issues + seg_issues
    print("=" * 50)
    if all_issues:
        print(f"Không đạt: {len(all_issues)} lỗi/cảnh báo")
        return 1
    print("Output đạt yêu cầu.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", required=True, help="Chapter ID, ví dụ HVH_001_01")
    parser.add_argument("--output-root", default="data/output")
    parser.add_argument("--catalog", default="configs/corpus_catalog.csv")
    parser.add_argument(
        "--allow-corrected",
        action="store_true",
        help="Cho phép seg khác raw khi đã publish bằng bước hiệu chỉnh LLM",
    )
    args = parser.parse_args()
    raise SystemExit(
        run_check(
            args.id,
            Path(args.output_root),
            Path(args.catalog),
            allow_corrected=args.allow_corrected,
        )
    )
