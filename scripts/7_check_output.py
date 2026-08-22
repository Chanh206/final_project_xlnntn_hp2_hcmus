#!/usr/bin/env python3
"""Kiểm tra output theo layout từng ảnh hoặc một cặp file cho mỗi folder."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

SUBMISSION_ITEM_RE = re.compile(r"^(HVH_\d{3})_(\d{4})$")


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


def catalog_groups(path: Path) -> dict[str, int]:
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy catalog: {path}")
    groups: dict[str, int] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            group_id = str(row.get("legacy_folder", ""))
            if not re.fullmatch(r"HVH_\d{3}", group_id):
                raise ValueError(f"legacy_folder không hợp lệ: {group_id!r}")
            if group_id in groups:
                raise ValueError(f"legacy_folder bị trùng: {group_id}")
            try:
                groups[group_id] = int(row["image_count"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"image_count không hợp lệ cho {group_id}") from exc
    expected = {f"HVH_{number:03d}" for number in range(1, 33)}
    if set(groups) != expected:
        raise ValueError("Catalog phải có đúng legacy_folder HVH_001..HVH_032")
    return groups


def expected_item_ids(group_id: str, image_count: int) -> list[str]:
    return [f"{group_id}_{number:04d}" for number in range(1, image_count + 1)]


def run_folder(
    group_id: str,
    output_root: Path,
    catalog_path: Path,
    allow_corrected: bool = False,
) -> int:
    try:
        groups = catalog_groups(catalog_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"LỖI: {exc}")
        return 2
    if group_id not in groups:
        print(f"Folder không có trong catalog: {group_id}")
        return 2
    expected_ids = expected_item_ids(group_id, groups[group_id])
    group_dir = output_root / group_id
    actual_ids = {path.name for path in group_dir.iterdir() if path.is_dir()} if group_dir.is_dir() else set()
    missing = sorted(set(expected_ids) - actual_ids)
    extra = sorted(actual_ids - set(expected_ids))
    failures = sum(
        run_check(item_id, output_root, catalog_path, allow_corrected, quiet=True) != 0
        for item_id in expected_ids
        if item_id in actual_ids
    )
    if missing:
        print(f"{group_id}: THIẾU {len(missing)} folder: {', '.join(missing[:10])}")
    if extra:
        print(f"{group_id}: THỪA {len(extra)} folder: {', '.join(extra[:10])}")
    if missing or extra or failures:
        print(f"{group_id}: KHÔNG ĐẠT missing={len(missing)}, extra={len(extra)}, invalid={failures}")
        return 1
    print(f"{group_id}: ĐẠT {len(expected_ids)}/{groups[group_id]} ảnh")
    return 0


def run_folder_aggregate(group_id: str, output_root: Path) -> int:
    """Kiểm tra layout HVH_NNN/HVH_NNN_raw.txt + HVH_NNN_seg.tsv."""
    if not re.fullmatch(r"HVH_\d{3}", group_id):
        print("Folder phải có dạng HVH_NNN, ví dụ HVH_021")
        return 2
    output_dir = output_root / group_id
    raw_path = output_dir / f"{group_id}_raw.txt"
    seg_path = output_dir / f"{group_id}_seg.tsv"
    raw_issues = check_raw(raw_path)
    sentence_count, seg_content, seg_issues = check_seg(seg_path, group_id)
    raw_text, _ = read_utf8(raw_path) if raw_path.is_file() else (None, [])
    if raw_text is not None and normalize_coverage(raw_text) != normalize_coverage(seg_content):
        seg_issues.append("Nội dung các câu không khớp toàn bộ nội dung raw")
    print(f"=== Kiểm tra output tổng hợp: {group_id} ===")
    print(f"Thư mục: {output_dir}")
    print_result(raw_path.name, raw_issues)
    print_result(seg_path.name, seg_issues, f" — {sentence_count} câu")
    issues = raw_issues + seg_issues
    if issues:
        print(f"Không đạt: {len(issues)} lỗi/cảnh báo")
        return 1
    print("Output tổng hợp đạt yêu cầu.")
    return 0


def run_all(
    output_root: Path,
    catalog_path: Path,
    allow_corrected: bool = False,
) -> int:
    try:
        groups = catalog_groups(catalog_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"LỖI: {exc}")
        return 2
    actual_groups = {path.name for path in output_root.iterdir() if path.is_dir()} if output_root.is_dir() else set()
    missing_groups = sorted(set(groups) - actual_groups)
    extra_groups = sorted(actual_groups - set(groups))
    failures = sum(
        run_folder(group_id, output_root, catalog_path, allow_corrected) != 0
        for group_id in sorted(groups)
    )
    if missing_groups or extra_groups or failures:
        print(f"KHÔNG ĐẠT TOÀN BỘ: missing_groups={len(missing_groups)}, extra_groups={len(extra_groups)}, invalid_groups={failures}")
        return 1
    print(f"ĐẠT TOÀN BỘ: 32 folder, {sum(groups.values())} ảnh trong {output_root}")
    return 0


def run_check(
    item_id: str,
    output_root: Path,
    catalog_path: Path,
    allow_corrected: bool = False,
    quiet: bool = False,
) -> int:
    match = SUBMISSION_ITEM_RE.fullmatch(item_id)
    if not match:
        print("ID phải có dạng HVH_GGG_NNNN, ví dụ HVH_001_0001")
        return 2
    group_id = match.group(1)
    item_number = int(match.group(2))
    try:
        groups = catalog_groups(catalog_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"LỖI: {exc}")
        return 2
    if group_id not in groups or not 1 <= item_number <= groups[group_id]:
        print(f"Ảnh không có trong catalog: {item_id}")
        return 2

    output_dir = output_root / group_id / item_id
    raw_path = output_dir / f"{item_id}_raw.txt"
    seg_path = output_dir / f"{item_id}_seg.tsv"

    raw_issues = check_raw(raw_path)
    sentence_count, seg_content, seg_issues = check_seg(seg_path, item_id)
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
    all_issues = raw_issues + seg_issues
    if quiet and not all_issues:
        return 0
    print(f"=== Kiểm tra output: {item_id} ===")
    print(f"Thư mục: {output_dir}")
    print_result(raw_path.name, raw_issues)
    print_result(seg_path.name, seg_issues, f" — {sentence_count} câu")
    if coverage_changed and allow_corrected:
        print("  ℹ️  Seg khác raw theo chế độ cho phép hiệu chỉnh; cần giữ báo cáo bước 8.")
    print("=" * 50)
    if all_issues:
        print(f"Không đạt: {len(all_issues)} lỗi/cảnh báo")
        return 1
    print("Output đạt yêu cầu.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--id", help="Submission item ID, ví dụ HVH_001_0001")
    selector.add_argument("--folder", help="Kiểm tra một folder nguồn, ví dụ HVH_001")
    selector.add_argument("--all", action="store_true", help="Kiểm tra đủ 32 folder và toàn bộ ảnh")
    parser.add_argument(
        "--output-root",
        default="final_output",
        help="Thư mục nộp chung (mặc định: final_output)",
    )
    parser.add_argument(
        "--output-layout",
        choices=("per-image", "folder"),
        default="per-image",
        help="Layout cần kiểm tra; folder dùng cho final_output_revised",
    )
    parser.add_argument("--catalog", default="configs/corpus_catalog.csv")
    parser.add_argument(
        "--allow-corrected",
        action="store_true",
        help="Cho phép seg khác raw khi đã publish bằng bước hiệu chỉnh LLM",
    )
    args = parser.parse_args()
    if args.output_layout == "folder":
        if args.id or args.all:
            parser.error("Layout folder hiện kiểm tra từng --folder; không dùng --id/--all")
        raise SystemExit(run_folder_aggregate(args.folder, Path(args.output_root)))
    if args.all:
        raise SystemExit(
            run_all(
                Path(args.output_root),
                Path(args.catalog),
                allow_corrected=args.allow_corrected,
            )
        )
    if args.folder:
        raise SystemExit(
            run_folder(
                args.folder,
                Path(args.output_root),
                Path(args.catalog),
                allow_corrected=args.allow_corrected,
            )
        )
    raise SystemExit(
        run_check(
            args.id,
            Path(args.output_root),
            Path(args.catalog),
            allow_corrected=args.allow_corrected,
        )
    )
