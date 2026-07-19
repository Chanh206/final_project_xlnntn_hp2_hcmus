#!/usr/bin/env python3
"""Ghép OCR processed thành ``_raw.txt`` và tách câu thành ``_seg.tsv``.

Thứ tự trang lấy từ manifest của bước 3, không dựa vào thứ tự
filesystem. Với bản Hán Nôm có dấu ngắt màu đỏ, script phát hiện dấu
trên ảnh, ánh xạ về vị trí ký tự bằng polygon OCR rồi nối văn bản qua
ranh giới cột. Dấu câu Unicode OCR nhận được vẫn luôn được ưu tiên.

File ``segmentation_review.tsv`` trong ``build_reports`` ghi nguồn và
phương pháp tạo từng câu để hỗ trợ hiệu đính; đây không phải output nộp.

Ví dụ:
    python scripts/6_build_outputs.py \
      --id HVH_001 --chapter 01 --run-name paddle_full_v1 --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import cv2
import numpy as np


WORK_ID_RE = re.compile(r"^HVH_\d{3}$")
CHAPTER_RE = re.compile(r"^\d{2}$")
SENTENCE_END_RE = re.compile(r"[。！？；!?;]+$")
SENTENCE_PART_RE = re.compile(r".+?[。！？；!?;]+|.+$")
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


@dataclass(frozen=True)
class RedMark:
    x: int
    y: int
    width: int
    height: int
    area: int
    center_x: float
    center_y: float
    local_density: float


@dataclass(frozen=True)
class Boundary:
    char_index: int
    method: str
    mark: RedMark | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--id", required=True, help="Work ID, ví dụ HVH_001")
    parser.add_argument("--chapter", default="01")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--catalog", default="configs/corpus_catalog.csv")
    parser.add_argument("--processed-root", default="data/processed")
    parser.add_argument("--intermediate-root", default="data/intermediate")
    parser.add_argument("--output-root", default="data/output")
    parser.add_argument(
        "--segmentation-mode",
        choices=("red-marks", "ocr-punctuation", "columns"),
        default="red-marks",
        help="red-marks: dùng dấu đỏ; ocr-punctuation: chỉ dấu OCR; columns: fallback cũ",
    )
    parser.add_argument(
        "--red-saturation",
        type=int,
        default=125,
        help="Ngưỡng saturation HSV để nhận dấu đỏ (mặc định: 125)",
    )
    parser.add_argument(
        "--max-blank-ratio",
        type=float,
        default=0.10,
        help=(
            "Tỷ lệ trang BLANK tối đa được phép trước khi tạo output "
            "(0-1, mặc định: 0.10 tương ứng 10%%)"
        ),
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not WORK_ID_RE.fullmatch(args.id):
        parser.error("--id phải có dạng HVH_NNN")
    if not CHAPTER_RE.fullmatch(args.chapter):
        parser.error("--chapter phải có hai chữ số")
    if not 0 <= args.max_blank_ratio <= 1:
        parser.error("--max-blank-ratio phải nằm trong khoảng 0 đến 1")
    return args


def catalog_entry(path: Path, work_id: str, chapter_id: str) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as handle:
        matches = [
            row
            for row in csv.DictReader(handle)
            if row.get("work_id") == work_id and row.get("chapter_id") == chapter_id
        ]
    if len(matches) != 1:
        raise ValueError(
            f"Catalog không có duy nhất một dòng cho {work_id}/{chapter_id}"
        )
    return matches[0]


def load_manifest_order(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy manifest: {path}")
    stems: list[str] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            processed_path = row.get("processed_image", "").strip()
            if not processed_path:
                raise ValueError("manifest có đường dẫn processed rỗng")
            stems.append(Path(processed_path).stem)
    if len(stems) != len(set(stems)):
        raise ValueError("manifest có processed_image trùng")
    return stems


def load_ocr_results(run_dir: Path, expected_stems: list[str], allow_incomplete: bool) -> tuple[list[dict[str, Any]], list[str]]:
    results = []
    problems = []
    for stem in expected_stems:
        path = run_dir / f"{stem}.json"
        if not path.is_file():
            problems.append(f"MISSING:{stem}")
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            problems.append(f"INVALID_JSON:{stem}:{exc}")
            continue
        status = data.get("status")
        if status not in {"success", "blank"}:
            problems.append(f"OCR_ERROR:{stem}:{data.get('error', '')}")
            continue
        if status == "success" and not str(data.get("text", "")).strip():
            problems.append(f"EMPTY_TEXT:{stem}")
            continue
        results.append({"stem": stem, **data})
    if problems and not allow_incomplete:
        preview = "; ".join(problems[:10])
        raise ValueError(f"OCR chưa đầy đủ ({len(problems)} lỗi): {preview}")
    return results, problems


def clean_unit(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\t", " ")).strip()


def polygon_bounds(line: dict[str, Any]) -> tuple[float, float, float, float] | None:
    polygon = line.get("polygon")
    if not isinstance(polygon, list) or len(polygon) < 4:
        return None
    try:
        xs = [float(point[0]) for point in polygon]
        ys = [float(point[1]) for point in polygon]
    except (TypeError, ValueError, IndexError):
        return None
    return min(xs), min(ys), max(xs), max(ys)


def red_mask(image: np.ndarray, saturation: int) -> np.ndarray:
    """Tách mực đỏ/cam, tránh nhầm giấy ngả vàng và nét chữ đen."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blue, green, red = cv2.split(image)
    blue_i = blue.astype(np.int16)
    green_i = green.astype(np.int16)
    red_i = red.astype(np.int16)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    mask = (
        ((hue <= 25) | (hue >= 170))
        & (sat >= saturation)
        & (val >= 55)
        & ((red_i - green_i) >= 18)
        & ((red_i - blue_i) >= 18)
    ).astype(np.uint8) * 255
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))


def detect_red_marks(image: np.ndarray, saturation: int) -> list[RedMark]:
    mask = red_mask(image, saturation)
    # Nối các nét gần nhau để nhận diện vùng con dấu/khung/gạch đỏ lớn.
    # Dấu ngắt riêng lẻ cách nhau xa nên không tạo component lớn ở bước này.
    clustered = cv2.dilate(
        mask, cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    )
    cluster_count, _, cluster_stats, _ = cv2.connectedComponentsWithStats(clustered, 8)
    exclusion_boxes: list[tuple[int, int, int, int]] = []
    for cluster_index in range(1, cluster_count):
        cx, cy, cw, ch, cluster_area = (
            int(value) for value in cluster_stats[cluster_index]
        )
        if ch >= 80 and cluster_area >= 4000:
            exclusion_boxes.append((cx - 5, cy - 5, cx + cw + 5, cy + ch + 5))

    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    marks: list[RedMark] = []
    image_height, image_width = mask.shape
    for index in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[index])
        if not (15 <= area <= 500 and 3 <= width <= 38 and 5 <= height <= 60):
            continue
        center_x = x + width / 2
        center_y = y + height / 2
        if any(
            x0 <= center_x <= x1 and y0 <= center_y <= y1
            for x0, y0, x1, y1 in exclusion_boxes
        ):
            continue
        radius = 40
        x0, x1 = max(0, int(center_x) - radius), min(image_width, int(center_x) + radius)
        y0, y1 = max(0, int(center_y) - radius), min(image_height, int(center_y) + radius)
        neighborhood = mask[y0:y1, x0:x1]
        density = float(np.count_nonzero(neighborhood)) / max(1, neighborhood.size)
        # Con dấu/khung đỏ tạo cụm mực dày; dấu ngắt thường đứng riêng lẻ.
        if density > 0.095:
            continue
        marks.append(
            RedMark(x, y, width, height, area, center_x, center_y, round(density, 5))
        )
    return marks


def is_vertical_text(line: dict[str, Any], text: str) -> bool:
    bounds = polygon_bounds(line)
    if bounds is None or len(text) < 2:
        return False
    x0, y0, x1, y1 = bounds
    return (y1 - y0) >= 1.8 * max(1.0, x1 - x0)


def line_red_boundaries(
    line: dict[str, Any], text: str, marks: list[RedMark]
) -> list[Boundary]:
    bounds = polygon_bounds(line)
    if bounds is None or not is_vertical_text(line, text):
        return []
    x0, y0, x1, y1 = bounds
    width = max(1.0, x1 - x0)
    height = max(1.0, y1 - y0)
    center_x = (x0 + x1) / 2
    char_step = height / max(1, len(text))
    candidates: dict[int, tuple[float, RedMark]] = {}
    for mark in marks:
        offset_x = mark.center_x - center_x
        if not (0.06 * width <= offset_x <= 0.72 * width + 8):
            continue
        if not (y0 - 8 <= mark.center_y <= y1 + 8):
            continue
        char_index = int(round((mark.center_y - y0) / char_step))
        if not (1 <= char_index <= len(text)):
            continue
        predicted_y = y0 + char_index * char_step
        residual = abs(mark.center_y - predicted_y)
        if residual > max(12.0, 0.42 * char_step):
            continue
        score = residual + 35.0 * mark.local_density + abs(offset_x - 0.32 * width) * 0.08
        current = candidates.get(char_index)
        if current is None or score < current[0]:
            candidates[char_index] = (score, mark)
    return [
        Boundary(char_index, "red_mark", candidates[char_index][1])
        for char_index in sorted(candidates)
    ]


def explicit_boundaries(text: str) -> list[Boundary]:
    return [
        Boundary(index, "ocr_punctuation")
        for index, char in enumerate(text, 1)
        if SENTENCE_END_RE.fullmatch(char)
    ]


def merge_boundaries(*groups: list[Boundary]) -> list[Boundary]:
    merged: dict[int, Boundary] = {}
    for group in groups:
        for boundary in group:
            existing = merged.get(boundary.char_index)
            if existing is None or boundary.method == "ocr_punctuation":
                merged[boundary.char_index] = boundary
    return [merged[index] for index in sorted(merged)]


def cjk_ratio(text: str) -> float:
    visible = [char for char in text if not char.isspace()]
    return sum(bool(CJK_RE.fullmatch(char)) for char in visible) / max(1, len(visible))


def is_layout_unit(
    line: dict[str, Any], text: str, image_height: int, image_width: int
) -> bool:
    """Tách riêng số trang/chữ Latin; không ghép chúng vào câu Hán văn."""
    bounds = polygon_bounds(line)
    if bounds is None:
        return True
    x0, y0, _, y1 = bounds
    short_right_title = (
        (y1 - y0) < 0.30 * image_height
        and len(text) <= 10
        and x0 > 0.70 * image_width
    )
    return cjk_ratio(text) < 0.55 or short_right_title


def segment_with_geometry(
    results: list[dict[str, Any]], mode: str, saturation: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sentences: list[dict[str, Any]] = []
    detections: list[dict[str, Any]] = []
    pending_text = ""
    pending_sources: list[str] = []

    def flush(method: str) -> None:
        nonlocal pending_text, pending_sources
        sentence = clean_unit(pending_text)
        if sentence:
            sentences.append(
                {
                    "text": sentence,
                    "method": method,
                    "sources": list(dict.fromkeys(pending_sources)),
                }
            )
        pending_text = ""
        pending_sources = []

    for result in results:
        if result.get("status") == "blank":
            continue
        stem = str(result["stem"])
        image = cv2.imread(str(result.get("image_path", "")))
        if image is None:
            raise FileNotFoundError(f"Không đọc được ảnh processed: {result.get('image_path')}")
        marks = detect_red_marks(image, saturation) if mode == "red-marks" else []
        lines = result.get("lines")
        if not isinstance(lines, list) or not lines:
            lines = [{"text": line} for line in str(result.get("text", "")).splitlines()]

        for line_index, line in enumerate(lines, 1):
            text = clean_unit(str(line.get("text", "")))
            if not text:
                continue
            if mode == "columns":
                flush("unmarked_tail")
                pending_text = text
                pending_sources = [stem]
                flush("column_fallback")
                continue
            if is_layout_unit(line, text, image.shape[0], image.shape[1]):
                flush("unmarked_tail")
                pending_text = text
                pending_sources = [stem]
                flush("layout_unit")
                continue

            red_boundaries = line_red_boundaries(line, text, marks) if mode == "red-marks" else []
            boundaries = merge_boundaries(red_boundaries, explicit_boundaries(text))
            start = 0
            pending_sources.append(stem)
            for boundary in boundaries:
                pending_text += text[start : boundary.char_index]
                flush(boundary.method)
                start = boundary.char_index
                if boundary.mark is not None:
                    mark = boundary.mark
                    detections.append(
                        {
                            "stem": stem,
                            "line_index": line_index,
                            "char_index": boundary.char_index,
                            "line_text": text,
                            "x": mark.x,
                            "y": mark.y,
                            "width": mark.width,
                            "height": mark.height,
                            "area": mark.area,
                            "local_density": mark.local_density,
                        }
                    )
                pending_sources.append(stem)
            pending_text += text[start:]
        # Không nối phần không có dấu qua ranh giới trang: tránh ghép tiêu đề/chú thích.
        flush("page_tail")
    flush("corpus_tail")
    return sentences, detections


def segment_ocr_text(text: str) -> list[tuple[str, str]]:
    segments: list[tuple[str, str]] = []
    for raw_line in text.splitlines():
        line = clean_unit(raw_line)
        if not line:
            continue
        parts = [clean_unit(part) for part in SENTENCE_PART_RE.findall(line)]
        for part in parts:
            if not part:
                continue
            method = "punctuation" if SENTENCE_END_RE.search(part) else "column_fallback"
            segments.append((part, method))
    return segments


def cjk_count(text: str) -> int:
    return sum(
        1
        for char in text
        if "\u3400" <= char <= "\u4dbf"
        or "\u4e00" <= char <= "\u9fff"
        or "\uf900" <= char <= "\ufaff"
    )


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(text, encoding="utf-8", newline="\n")
    temp.replace(path)


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    write_text_atomic(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def main() -> int:
    args = parse_args()
    chapter_id = f"{args.id}_{args.chapter}"
    try:
        catalog = catalog_entry(Path(args.catalog), args.id, chapter_id)
        manifest_path = Path(args.processed_root) / args.id / chapter_id / "manifest.csv"
        expected_stems = load_manifest_order(manifest_path)
        run_dir = Path(args.intermediate_root) / args.id / chapter_id / "ocr_runs" / args.run_name / "processed"
        results, problems = load_ocr_results(run_dir, expected_stems, args.allow_incomplete)
    except (FileNotFoundError, ValueError) as exc:
        print(f"LỖI: {exc}", file=sys.stderr)
        return 1

    blank_count = sum(result.get("status") == "blank" for result in results)
    blank_ratio = blank_count / len(expected_stems) if expected_stems else 0.0
    if blank_ratio > args.max_blank_ratio:
        print(
            "LỖI: Không tạo output vì tỷ lệ BLANK "
            f"{blank_count}/{len(expected_stems)} = {blank_ratio:.2%}, "
            f"vượt ngưỡng {args.max_blank_ratio:.2%}.",
            file=sys.stderr,
        )
        print(
            "Hãy kiểm tra cấu hình OCR, chạy lại các trang BLANK rồi mới chạy bước 6.",
            file=sys.stderr,
        )
        return 1

    output_dir = Path(args.output_root) / args.id / chapter_id
    raw_path = output_dir / f"{chapter_id}_raw.txt"
    seg_path = output_dir / f"{chapter_id}_seg.tsv"
    if not args.overwrite and (raw_path.exists() or seg_path.exists()):
        print("LỖI: Output đã tồn tại; dùng --overwrite để ghi lại", file=sys.stderr)
        return 1

    raw_pages = [str(result["text"]).strip() for result in results]
    raw_text = "\n\n".join(raw_pages).rstrip() + "\n"
    try:
        all_segments, detections = segment_with_geometry(
            results, args.segmentation_mode, args.red_saturation
        )
    except FileNotFoundError as exc:
        print(f"LỖI: {exc}", file=sys.stderr)
        return 1
    if not all_segments:
        print("LỖI: Không tạo được câu nào", file=sys.stderr)
        return 1

    seg_lines = [
        f"{chapter_id}_{index:06d}\t{segment['text']}"
        for index, segment in enumerate(all_segments, 1)
    ]
    write_text_atomic(raw_path, raw_text)
    write_text_atomic(seg_path, "\n".join(seg_lines) + "\n")

    confidences = [
        float(result.get("confidence", 0))
        for result in results
        if result.get("status") == "success"
    ]
    method_counts = dict(Counter(str(segment["method"]) for segment in all_segments))
    report_dir = Path(args.intermediate_root) / args.id / chapter_id / "build_reports"
    review_path = report_dir / f"{args.run_name}.segmentation_review.tsv"
    detection_path = report_dir / f"{args.run_name}.red_mark_detections.tsv"
    review_lines = ["sentence_id\tmethod\tsource_pages\tlength\tsentence"]
    for index, segment in enumerate(all_segments, 1):
        review_lines.append(
            "\t".join(
                [
                    f"{chapter_id}_{index:06d}",
                    str(segment["method"]),
                    ",".join(segment["sources"]),
                    str(len(segment["text"])),
                    str(segment["text"]),
                ]
            )
        )
    write_text_atomic(review_path, "\n".join(review_lines) + "\n")
    detection_fields = [
        "stem", "line_index", "char_index", "line_text", "x", "y",
        "width", "height", "area", "local_density",
    ]
    detection_lines = ["\t".join(detection_fields)]
    detection_lines.extend(
        "\t".join(str(row[field]).replace("\t", " ") for field in detection_fields)
        for row in detections
    )
    write_text_atomic(detection_path, "\n".join(detection_lines) + "\n")
    report = {
        "chapter_id": chapter_id,
        "title_vietnamese": catalog["title_vietnamese"],
        "run_name": args.run_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "expected_processed_pages": len(expected_stems),
        "used_processed_pages": len(results),
        "blank_pages": [result["stem"] for result in results if result.get("status") == "blank"],
        "blank_ratio": round(blank_ratio, 6),
        "max_blank_ratio": args.max_blank_ratio,
        "problems": problems,
        "raw_path": raw_path.as_posix(),
        "seg_path": seg_path.as_posix(),
        "raw_characters": len(raw_text),
        "raw_cjk_characters": cjk_count(raw_text),
        "sentence_count": len(all_segments),
        "segmentation_mode": args.segmentation_mode,
        "red_saturation": args.red_saturation,
        "red_mark_detections": len(detections),
        "segmentation_methods": method_counts,
        "segmentation_review_path": review_path.as_posix(),
        "red_mark_detection_path": detection_path.as_posix(),
        "mean_ocr_confidence": round(mean(confidences), 3) if confidences else 0,
        "low_confidence_pages_below_50": [
            result["stem"]
            for result in results
            if result.get("status") == "success" and float(result.get("confidence", 0)) < 50
        ],
        "warning": (
            "Ranh giới từ mực đỏ được phát hiện tự động; cần duyệt các dòng page_tail/layout_unit và mẫu red_mark."
            if args.segmentation_mode == "red-marks"
            else "Không dùng dấu ngắt màu đỏ; ranh giới câu cần được hiệu đính."
        ),
    }
    report_path = report_dir / f"{args.run_name}.json"
    write_json_atomic(report_path, report)
    print(f"Đã tạo raw: {raw_path}")
    print(f"Đã tạo seg: {seg_path} ({len(all_segments)} câu/đơn vị)")
    print(f"Review tách câu: {review_path}")
    print(f"Báo cáo: {report_path}")
    if problems:
        print(f"CẢNH BÁO: output không đầy đủ, có {len(problems)} trang lỗi/thiếu")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
