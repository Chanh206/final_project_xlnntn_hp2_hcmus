#!/usr/bin/env python3
"""Kiểm tra chất lượng và tiền xử lý ảnh chữ Hán theo từng tác phẩm.

Mặc định script không sửa ``data/input``. Mỗi ảnh được cắt viền nền đen,
tự động tách hai trang nếu phát hiện một spread, rồi ghi sang
``data/processed/<WORK_ID>/<CHAPTER_ID>``. Với spread, trang bên phải được ghi trước trang
bên trái để tên file sắp xếp đúng thứ tự đọc chữ dọc Hán.

Ví dụ:
    python scripts/3_preprocess_images.py --id HVH_001 --limit 10
    python scripts/3_preprocess_images.py --id HVH_001 --split always --overwrite
    python scripts/3_preprocess_images.py --id HVH_001 --variant clahe

Kết quả:
    data/processed/HVH_001/HVH_001_01/HVH_001_01_0001_p01.jpg
    data/processed/HVH_001/HVH_001_01/HVH_001_01_0001_p02.jpg
    data/processed/HVH_001/HVH_001_01/manifest.csv
    data/processed/HVH_001/HVH_001_01/summary.json
    data/intermediate/HVH_001/HVH_001_01/pages/ (dành cho bước OCR sau)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
ID_RE = re.compile(r"^HVH_\d{3}$")
CHAPTER_RE = re.compile(r"^\d{2}$")


@dataclass
class QualityMetrics:
    width: int
    height: int
    blur_score: float
    brightness: float
    contrast: float
    dark_pixel_ratio: float


@dataclass
class ManifestRow:
    source_image: str
    processed_image: str
    source_sha256: str
    source_width: int
    source_height: int
    page_part: int
    reading_order: str
    crop_x1: int
    crop_y1: int
    crop_x2: int
    crop_y2: int
    split_applied: bool
    split_x: int | None
    split_confidence: float
    variant: str
    output_width: int
    output_height: int
    blur_score: float
    brightness: float
    contrast: float
    dark_pixel_ratio: float
    quality_flags: str
    write_status: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--id", help="Mã xử lý nội bộ, ví dụ HVH_018")
    parser.add_argument("--folder", help="Folder nguồn 1-1, ví dụ HVH_019")
    parser.add_argument("--chapter", help="Mã quyển nội bộ; chỉ dùng cùng --id")
    parser.add_argument("--catalog", default="configs/corpus_catalog.csv")
    parser.add_argument("--input-root", default="data/input")
    parser.add_argument("--processed-root", default="data/processed")
    parser.add_argument("--intermediate-root", default="data/intermediate")
    parser.add_argument("--limit", type=int, help="Chỉ xử lý N ảnh đầu tiên để pilot")
    parser.add_argument(
        "--split",
        choices=("auto", "always", "never"),
        default="auto",
        help="Chế độ tách hai trang (mặc định: auto)",
    )
    parser.add_argument(
        "--variant",
        choices=("color", "gray", "clahe", "binary"),
        default="color",
        help="Biến thể ảnh output; color an toàn nhất (mặc định)",
    )
    parser.add_argument("--jpeg-quality", type=int, default=95, choices=range(70, 101), metavar="70..100")
    parser.add_argument("--overwrite", action="store_true", help="Ghi đè ảnh processed đã có")
    parser.add_argument("--fail-on-warning", action="store_true", help="Trả exit code 2 nếu có cảnh báo chất lượng")
    args = parser.parse_args()

    if bool(args.id) == bool(args.folder):
        parser.error("Phải chọn đúng một trong --folder hoặc --id")
    if args.folder:
        if args.chapter:
            parser.error("Không dùng --chapter cùng --folder")
        if not ID_RE.fullmatch(args.folder):
            parser.error("--folder phải có dạng HVH_NNN")
        with Path(args.catalog).open(encoding="utf-8", newline="") as handle:
            matches = [row for row in csv.DictReader(handle) if row.get("legacy_folder") == args.folder]
        if len(matches) != 1:
            parser.error(f"Catalog không có duy nhất một dòng cho --folder {args.folder}")
        args.id = matches[0]["work_id"]
        args.chapter = matches[0]["chapter_id"].rsplit("_", 1)[-1]
    else:
        args.chapter = args.chapter or "01"
        if not ID_RE.fullmatch(args.id):
            parser.error("--id phải có dạng HVH_NNN, ví dụ HVH_001")
        if not CHAPTER_RE.fullmatch(args.chapter):
            parser.error("--chapter phải có hai chữ số, ví dụ 01")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit phải lớn hơn 0")
    return args


def natural_key(path: Path) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def list_images(input_dir: Path, limit: int | None) -> list[Path]:
    images = sorted(
        (path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=natural_key,
    )
    return images[:limit] if limit is not None else images


def catalog_entry(path: Path, work_id: str, chapter_id: str) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy catalog: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        matches = [
            row
            for row in csv.DictReader(handle)
            if row.get("work_id") == work_id and row.get("chapter_id") == chapter_id
        ]
    if len(matches) != 1:
        raise ValueError(f"Catalog không có duy nhất một dòng cho {work_id}/{chapter_id}")
    return matches[0]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quality_metrics(image: np.ndarray) -> QualityMetrics:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    height, width = gray.shape[:2]
    return QualityMetrics(
        width=width,
        height=height,
        blur_score=round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2),
        brightness=round(float(gray.mean()), 2),
        contrast=round(float(gray.std()), 2),
        dark_pixel_ratio=round(float(np.mean(gray < 35)), 4),
    )


def quality_flags(metrics: QualityMetrics) -> list[str]:
    flags = []
    if min(metrics.width, metrics.height) < 800:
        flags.append("LOW_RESOLUTION")
    if metrics.blur_score < 50:
        flags.append("POSSIBLY_BLURRY")
    if metrics.contrast < 25:
        flags.append("LOW_CONTRAST")
    if metrics.brightness < 45:
        flags.append("TOO_DARK")
    elif metrics.brightness > 235:
        flags.append("TOO_BRIGHT")
    return flags


def smooth_1d(values: np.ndarray, window: int) -> np.ndarray:
    window = max(3, window | 1)
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(values, kernel, mode="same")


def content_bbox(image: np.ndarray) -> tuple[int, int, int, int]:
    """Tìm khung trang sách, chỉ cắt viền nền rất tối ở ngoài."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    light = gray > 28
    column_occupancy = light.mean(axis=0)
    row_occupancy = light.mean(axis=1)

    xs = np.flatnonzero(column_occupancy > 0.05)
    ys = np.flatnonzero(row_occupancy > 0.05)
    if len(xs) == 0 or len(ys) == 0:
        return 0, 0, width, height

    padding_x = max(4, int(width * 0.005))
    padding_y = max(4, int(height * 0.005))
    x1 = max(0, int(xs[0]) - padding_x)
    x2 = min(width, int(xs[-1]) + padding_x + 1)
    y1 = max(0, int(ys[0]) - padding_y)
    y2 = min(height, int(ys[-1]) + padding_y + 1)

    if (x2 - x1) < width * 0.4 or (y2 - y1) < height * 0.4:
        return 0, 0, width, height
    return x1, y1, x2, y2


def detect_spread(image: np.ndarray) -> tuple[bool, int | None, float]:
    """Phát hiện khe giữa hai trang bằng profile độ sáng gần tâm ảnh."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    aspect_ratio = width / max(height, 1)
    if aspect_ratio < 0.82:
        return False, None, 0.0

    # Dò đường gáy trên các dải đầu/cuối trang, nơi ít chữ hơn.
    # Cách này tránh nhầm khoảng trắng giữa hai cột chữ là đường gáy.
    band_height = max(1, int(height * 0.18))
    edge_bands = np.vstack((gray[:band_height], gray[-band_height:]))
    occupancy = (edge_bands > 45).mean(axis=0)
    smoothed = smooth_1d(occupancy, max(9, int(width * 0.008)))
    lo, hi = int(width * 0.43), int(width * 0.57)
    if hi <= lo:
        return False, None, 0.0
    split_x = lo + int(np.argmin(smoothed[lo:hi]))
    valley = float(smoothed[split_x])

    left_band = smoothed[int(width * 0.20) : int(width * 0.38)]
    right_band = smoothed[int(width * 0.62) : int(width * 0.80)]
    side_level = float(min(np.median(left_band), np.median(right_band)))
    drop = max(0.0, side_level - valley)
    confidence = max(0.0, min(1.0, drop / max(side_level, 0.01)))

    balanced = width * 0.40 <= split_x <= width * 0.60
    enough_page = split_x >= width * 0.30 and (width - split_x) >= width * 0.30
    # Scan hai trang của bộ dữ liệu có tỉ lệ rộng/cao xấp xỉ 1.1.
    # Với spread còn nguyên, đường gáy có thể không tối hơn mặt giấy nhiều,
    # nên tỉ lệ khung hình là tín hiệu chính và valley là độ tin cậy.
    wide_spread = aspect_ratio >= 0.92
    strong_gutter = valley < 0.75 and drop > 0.10
    detected = balanced and enough_page and (wide_spread or strong_gutter)
    return detected, split_x if detected else None, round(confidence, 3)


def trim_piece(image: np.ndarray, origin_x: int, origin_y: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    x1, y1, x2, y2 = content_bbox(image)
    return image[y1:y2, x1:x2], (origin_x + x1, origin_y + y1, origin_x + x2, origin_y + y2)


def split_pages(
    image: np.ndarray, split_mode: str
) -> tuple[list[tuple[np.ndarray, tuple[int, int, int, int]]], bool, int | None, float]:
    outer_x1, outer_y1, outer_x2, outer_y2 = content_bbox(image)
    cropped = image[outer_y1:outer_y2, outer_x1:outer_x2]
    detected, local_split_x, confidence = detect_spread(cropped)

    should_split = split_mode == "always" or (split_mode == "auto" and detected)
    if not should_split:
        return [(cropped, (outer_x1, outer_y1, outer_x2, outer_y2))], False, None, confidence

    if local_split_x is None:
        local_split_x = cropped.shape[1] // 2
        confidence = 0.0

    # Chữ Hán dọc: trang bên phải trước, trang bên trái sau.
    left = cropped[:, :local_split_x]
    right = cropped[:, local_split_x:]
    right_piece = trim_piece(right, outer_x1 + local_split_x, outer_y1)
    left_piece = trim_piece(left, outer_x1, outer_y1)
    absolute_split_x = outer_x1 + local_split_x
    return [right_piece, left_piece], True, absolute_split_x, confidence


def apply_variant(image: np.ndarray, variant: str) -> np.ndarray:
    if variant == "color":
        return image

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if variant == "gray":
        return gray
    if variant == "clahe":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)
    if variant == "binary":
        return cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            41,
            15,
        )
    raise ValueError(f"Biến thể không hợp lệ: {variant}")


def write_jpeg(path: Path, image: np.ndarray, quality: int, overwrite: bool) -> str:
    if path.exists() and not overwrite:
        return "existing"
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError(f"Không encode được JPEG: {path}")
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_bytes(encoded.tobytes())
    temp_path.replace(path)
    return "written"


def write_manifest(path: Path, rows: Iterable[ManifestRow]) -> None:
    rows = list(rows)
    fieldnames = list(asdict(rows[0]).keys()) if rows else [field.name for field in ManifestRow.__dataclass_fields__.values()]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def main() -> int:
    args = parse_args()
    chapter_id = f"{args.id}_{args.chapter}"
    try:
        catalog = catalog_entry(Path(args.catalog), args.id, chapter_id)
    except (FileNotFoundError, ValueError) as exc:
        print(f"LỖI: {exc}", file=sys.stderr)
        return 1

    input_dir = (
        Path(args.input_root) / args.folder
        if args.folder
        else Path(args.input_root) / args.id / chapter_id
    )
    processed_dir = Path(args.processed_root) / args.id / chapter_id
    intermediate_pages_dir = Path(args.intermediate_root) / args.id / chapter_id / "pages"

    if not input_dir.is_dir():
        print(f"LỖI: Không tìm thấy thư mục input: {input_dir}", file=sys.stderr)
        return 1

    images = list_images(input_dir, args.limit)
    if not images:
        print(f"LỖI: Không có ảnh trong {input_dir}", file=sys.stderr)
        return 1

    processed_dir.mkdir(parents=True, exist_ok=True)
    intermediate_pages_dir.mkdir(parents=True, exist_ok=True)

    rows: list[ManifestRow] = []
    errors: list[dict[str, str]] = []
    hash_to_source: dict[str, str] = {}
    flag_counts: Counter[str] = Counter()
    split_count = 0

    print(f"[{chapter_id}] {catalog['title_vietnamese']}: kiểm tra và tiền xử lý {len(images)} ảnh")
    for index, source_path in enumerate(images, start=1):
        relative_source = source_path.as_posix()
        try:
            source_hash = sha256_file(source_path)
            duplicate_of = hash_to_source.get(source_hash)
            hash_to_source.setdefault(source_hash, relative_source)

            image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("OpenCV không đọc được file ảnh")

            source_metrics = quality_metrics(image)
            pages, split_applied, split_x, split_confidence = split_pages(image, args.split)
            split_count += int(split_applied)

            for page_part, (page_image, bbox) in enumerate(pages, start=1):
                processed = apply_variant(page_image, args.variant)
                metrics = quality_metrics(processed)
                flags = quality_flags(metrics)
                if split_applied and split_confidence < 0.10:
                    flags.append("LOW_SPLIT_CONFIDENCE")
                if duplicate_of:
                    flags.append(f"DUPLICATE_OF:{duplicate_of}")
                for flag in flags:
                    flag_counts[flag.split(":", 1)[0]] += 1

                output_name = f"{source_path.stem}_p{page_part:02d}.jpg"
                output_path = processed_dir / output_name
                status = write_jpeg(output_path, processed, args.jpeg_quality, args.overwrite)
                x1, y1, x2, y2 = bbox
                rows.append(
                    ManifestRow(
                        source_image=relative_source,
                        processed_image=output_path.as_posix(),
                        source_sha256=source_hash,
                        source_width=source_metrics.width,
                        source_height=source_metrics.height,
                        page_part=page_part,
                        reading_order="right_page_then_left_page; columns_right_to_left; top_to_bottom",
                        crop_x1=x1,
                        crop_y1=y1,
                        crop_x2=x2,
                        crop_y2=y2,
                        split_applied=split_applied,
                        split_x=split_x,
                        split_confidence=split_confidence,
                        variant=args.variant,
                        output_width=metrics.width,
                        output_height=metrics.height,
                        blur_score=metrics.blur_score,
                        brightness=metrics.brightness,
                        contrast=metrics.contrast,
                        dark_pixel_ratio=metrics.dark_pixel_ratio,
                        quality_flags=";".join(flags),
                        write_status=status,
                    )
                )
            print(
                f"  [{index:04d}/{len(images):04d}] {source_path.name}: "
                f"{len(pages)} trang, split={'yes' if split_applied else 'no'}, "
                f"confidence={split_confidence:.3f}"
            )
        except Exception as exc:  # tiếp tục batch và ghi lỗi vào summary
            errors.append({"source_image": relative_source, "error": str(exc)})
            print(f"  [{index:04d}/{len(images):04d}] LỖI {source_path.name}: {exc}", file=sys.stderr)

    manifest_path = processed_dir / "manifest.csv"
    write_manifest(manifest_path, rows)
    summary = {
        "corpus_id": args.id,
        "chapter_id": chapter_id,
        "title_vietnamese": catalog["title_vietnamese"],
        "source_volume_id": catalog["source_volume_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": input_dir.as_posix(),
        "processed_dir": processed_dir.as_posix(),
        "input_images_selected": len(images),
        "input_images_succeeded": len(images) - len(errors),
        "input_images_failed": len(errors),
        "spreads_split": split_count,
        "processed_pages": len(rows),
        "variant": args.variant,
        "split_mode": args.split,
        "quality_flag_counts": dict(sorted(flag_counts.items())),
        "errors": errors,
    }
    summary_path = processed_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"\nManifest: {manifest_path}")
    print(f"Tổng kết: {summary_path}")
    print(f"Ảnh input thành công: {summary['input_images_succeeded']}/{len(images)}")
    print(f"Spread đã tách: {split_count}; trang processed: {len(rows)}")
    print(f"Cảnh báo: {dict(flag_counts) if flag_counts else 'không có'}")
    print(f"Thư mục OCR trung gian: {intermediate_pages_dir}")

    if errors:
        return 1
    if args.fail_on_warning and flag_counts:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
