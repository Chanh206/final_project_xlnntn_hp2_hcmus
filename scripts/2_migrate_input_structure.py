#!/usr/bin/env python3
"""Chuyển input phẳng HVH_NNN sang cấu trúc tác phẩm/quyển.

Mặc định chỉ dry-run. Thêm ``--execute`` để thực hiện. Script kiểm tra
số file, số trang, target trùng và SHA-256 tổng hợp trước/sau khi di
chuyển. Nội dung ảnh không bị thay đổi.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
PAGE_RE = re.compile(r"_(\d{4})\.(jpg|jpeg|png|tif|tiff|bmp)$", re.IGNORECASE)


@dataclass(frozen=True)
class CatalogRow:
    work_id: str
    chapter_id: str
    title_vietnamese: str
    source_volume_id: str
    image_count: int
    legacy_folder: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="configs/corpus_catalog.csv")
    parser.add_argument("--input-root", default="data/input")
    parser.add_argument("--report", default="configs/input_migration_report.json")
    parser.add_argument("--execute", action="store_true", help="Thực hiện di chuyển; mặc định chỉ dry-run")
    return parser.parse_args()


def load_catalog(path: Path) -> list[CatalogRow]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = [
        CatalogRow(
            work_id=row["work_id"],
            chapter_id=row["chapter_id"],
            title_vietnamese=row["title_vietnamese"],
            source_volume_id=row["source_volume_id"],
            image_count=int(row["image_count"]),
            legacy_folder=row["legacy_folder"],
        )
        for row in rows
    ]
    if len({row.chapter_id for row in result}) != len(result):
        raise ValueError("chapter_id bị trùng trong catalog")
    if len({row.legacy_folder for row in result}) != len(result):
        raise ValueError("legacy_folder bị trùng trong catalog")
    return result


def direct_images(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return sorted(
        (item for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda item: item.name,
    )


def image_page(path: Path) -> int:
    match = PAGE_RE.search(path.name)
    if not match:
        raise ValueError(f"Tên ảnh không có số trang 4 chữ số: {path}")
    return int(match.group(1))


def validate_pages(images: list[Path], expected_count: int, label: str) -> None:
    if len(images) != expected_count:
        raise ValueError(f"{label}: có {len(images)} ảnh, catalog yêu cầu {expected_count}")
    pages = [image_page(path) for path in images]
    expected_pages = list(range(1, expected_count + 1))
    if pages != expected_pages:
        raise ValueError(f"{label}: số trang không liên tục 0001..{expected_count:04d}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate_digest(images: list[Path]) -> str:
    digest = hashlib.sha256()
    for image in images:
        digest.update(f"{image_page(image):04d}\t{file_sha256(image)}\n".encode("ascii"))
    return digest.hexdigest()


def target_name(chapter_id: str, source: Path) -> str:
    return f"{chapter_id}_{image_page(source):04d}{source.suffix.lower()}"


def main() -> int:
    args = parse_args()
    catalog_path = Path(args.catalog)
    input_root = Path(args.input_root)
    report_path = Path(args.report)
    staging_root = input_root / ".migration_staging"

    try:
        rows = load_catalog(catalog_path)
        plans = []
        already_migrated = []
        for row in rows:
            source_dir = input_root / row.legacy_folder
            target_dir = input_root / row.work_id / row.chapter_id
            source_images = direct_images(source_dir)
            target_images = direct_images(target_dir)

            if source_images:
                if target_images:
                    raise ValueError(f"Cả source và target đều có ảnh: {source_dir} / {target_dir}")
                validate_pages(source_images, row.image_count, row.legacy_folder)
                plans.append((row, source_dir, target_dir, source_images))
            elif target_images:
                validate_pages(target_images, row.image_count, row.chapter_id)
                already_migrated.append((row, target_dir, target_images))
            else:
                raise ValueError(f"Không tìm thấy ảnh source hoặc target cho {row.chapter_id}")

        if plans and already_migrated:
            raise ValueError("Dữ liệu đang ở trạng thái migration dở dang; không tự động tiếp tục")

        print(f"Catalog: {len({row.work_id for row in rows})} tác phẩm, {len(rows)} quyển")
        if already_migrated:
            print("Dữ liệu đã ở cấu trúc mới; chỉ kiểm tra, không cần di chuyển.")
            return 0

        total_images = sum(len(item[3]) for item in plans)
        print(f"Sẽ di chuyển {total_images} ảnh trong {len(plans)} quyển:")
        for row, source_dir, target_dir, images in plans:
            print(f"  {source_dir} -> {target_dir} ({len(images)} ảnh)")

        if not args.execute:
            print("\nDRY-RUN THÀNH CÔNG. Chạy lại với --execute để thực hiện.")
            return 0

        if staging_root.exists():
            raise ValueError(f"Thư mục staging đã tồn tại: {staging_root}")

        before = {}
        for row, _, _, images in plans:
            before[row.chapter_id] = aggregate_digest(images)

        staging_root.mkdir(parents=True)
        for row, _, _, images in plans:
            stage_dir = staging_root / row.legacy_folder
            stage_dir.mkdir()
            for image in images:
                image.replace(stage_dir / image.name)

        # Chỉ xóa thư mục legacy khi chúng hoàn toàn rỗng.
        for _, source_dir, _, _ in plans:
            source_dir.rmdir()

        for row, _, target_dir, _ in plans:
            stage_dir = staging_root / row.legacy_folder
            staged_images = direct_images(stage_dir)
            target_dir.mkdir(parents=True, exist_ok=False)
            for image in staged_images:
                image.replace(target_dir / target_name(row.chapter_id, image))
            stage_dir.rmdir()
        staging_root.rmdir()

        report_rows = []
        for row in rows:
            target_dir = input_root / row.work_id / row.chapter_id
            images = direct_images(target_dir)
            validate_pages(images, row.image_count, row.chapter_id)
            after_digest = aggregate_digest(images)
            if after_digest != before[row.chapter_id]:
                raise RuntimeError(f"Checksum thay đổi sau migration: {row.chapter_id}")
            report_rows.append(
                {
                    "work_id": row.work_id,
                    "chapter_id": row.chapter_id,
                    "title_vietnamese": row.title_vietnamese,
                    "source_volume_id": row.source_volume_id,
                    "image_count": len(images),
                    "aggregate_sha256": after_digest,
                }
            )

        report = {
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "catalog": catalog_path.as_posix(),
            "input_root": input_root.as_posix(),
            "work_count": len({row.work_id for row in rows}),
            "chapter_count": len(rows),
            "image_count": sum(row["image_count"] for row in report_rows),
            "chapters": report_rows,
        }
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nMIGRATION THÀNH CÔNG: {report['image_count']} ảnh; report: {report_path}")
        return 0
    except Exception as exc:
        print(f"LỖI: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
