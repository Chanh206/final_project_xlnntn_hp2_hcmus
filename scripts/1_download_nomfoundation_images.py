"""
Tải ảnh trang sách từ Nom Foundation Library (lib.nomfoundation.org)
và lưu vào thư mục input với tên file tăng dần: <prefix>_01.jpg, <prefix>_02.jpg, ...

Chế độ 1 — hàng loạt từ file danh sách link volume và catalog:
  python scripts/1_download_nomfoundation_images.py \
      --links-file data/input/link.txt \
      --out-root data/input

  Mỗi volume ID phải có trong configs/corpus_catalog.csv. Mỗi dòng catalog
  được giữ thành một folder nguồn độc lập, không gộp theo tác phẩm:
  data/input/HVH_001/HVH_001_0001.jpg ... data/input/HVH_032/...

Chế độ 2 — 1 volume, chỉ định thủ công:
  python scripts/1_download_nomfoundation_images.py \
      --image-code nlvnpf-0501 \
      --pages 48 \
      --prefix HVH_001 \
      --out-dir data/input/HVH_001
"""
import argparse
import csv
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from tqdm import tqdm

IMAGE_URL = "https://lib.nomfoundation.org/site_media/nom/{image_code}/{quality}/{image_code}-{page:03d}.jpg"
VOLUME_PAGE_URL = "https://lib.nomfoundation.org/collection/1/volume/{volume_id}/page/1"

VOLUME_ID_RE = re.compile(r"/volume/(\d+)")
IMAGE_CODE_RE = re.compile(r"site_media/nom/([A-Za-z0-9_-]+)/(?:large|jpeg)/")
TOTAL_PAGES_RE = re.compile(r"Page 1 of (\d+)")


def fetch(url: str) -> requests.Response:
    last_error: requests.RequestException | None = None
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(2 ** attempt)
    assert last_error is not None
    raise last_error


def download_page(image_code: str, quality: str, page: int, dest: Path) -> str:
    """Tải 1 trang, tự fallback sang 'jpeg' nếu bản 'large' không tồn tại (404).
    Trả về quality thực tế đã tải được."""
    qualities = [quality] + [q for q in ("large", "jpeg") if q != quality]
    last_error = None
    for q in qualities:
        url = IMAGE_URL.format(image_code=image_code, quality=q, page=page)
        try:
            resp = fetch(url)
            content = resp.content
            if len(content) < 1024 or not content.startswith(b"\xff\xd8"):
                raise requests.RequestException(f"Nội dung tải về không phải JPEG hợp lệ: {url}")
            temp = dest.with_suffix(dest.suffix + ".part")
            temp.write_bytes(content)
            temp.replace(dest)
            return q
        except requests.HTTPError as e:
            last_error = e
            continue
    raise last_error


def parse_links_file(path: Path) -> list[str]:
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "http" not in line:
            continue
        lines.append(line)
    return lines


def load_catalog_by_volume(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {row["source_volume_id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"source_volume_id bị trùng trong catalog: {path}")
    expected_folders = {f"HVH_{index:03d}" for index in range(1, 33)}
    actual_folders = {row.get("legacy_folder", "") for row in rows}
    if len(rows) != 32 or actual_folders != expected_folders:
        raise ValueError("Catalog phải có đúng legacy_folder HVH_001..HVH_032")
    return result


def validate_batch_inputs(
    links_file: Path, catalog_path: Path
) -> tuple[list[str], dict[str, dict[str, str]]]:
    urls = parse_links_file(links_file)
    if not urls:
        raise ValueError(f"Không tìm thấy link nào trong {links_file}")
    catalog = load_catalog_by_volume(catalog_path)
    url_volume_ids = []
    for url in urls:
        match = VOLUME_ID_RE.search(url)
        if not match:
            raise ValueError(f"Không tìm thấy volume id trong URL: {url}")
        url_volume_ids.append(match.group(1))
    if len(urls) != 32 or len(set(url_volume_ids)) != 32:
        raise ValueError("link.txt phải có đúng 32 volume không trùng")
    if len(catalog) != 32 or set(url_volume_ids) != set(catalog):
        missing = sorted(set(catalog) - set(url_volume_ids))
        extra = sorted(set(url_volume_ids) - set(catalog))
        raise ValueError(f"Link/catalog không khớp; missing={missing}, extra={extra}")
    return urls, catalog


def valid_jpeg(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 1024:
        return False
    with path.open("rb") as handle:
        return handle.read(2) == b"\xff\xd8"


def resolve_volume(url: str) -> tuple[str, str, int]:
    """Trả về (volume_id, image_code, total_pages) bằng cách đọc trang 'page/1' của volume."""
    match = VOLUME_ID_RE.search(url)
    if not match:
        raise ValueError(f"Không tìm thấy volume id trong URL: {url}")
    volume_id = match.group(1)

    page_url = VOLUME_PAGE_URL.format(volume_id=volume_id)
    html = fetch(page_url).text

    code_match = IMAGE_CODE_RE.search(html)
    total_match = TOTAL_PAGES_RE.search(html)
    if not code_match or not total_match:
        raise ValueError(f"Không dò được mã ảnh/tổng số trang cho volume {volume_id}")

    return volume_id, code_match.group(1), int(total_match.group(1))


def download_volume(image_code: str, total_pages: int, prefix: str, out_dir: Path,
                    quality: str, delay: float, start: int = 1, width: int = 4,
                    workers: int = 1) -> tuple[int, list]:
    out_dir.mkdir(parents=True, exist_ok=True)
    width = max(width, len(str(total_pages)))

    def process_page(page: int) -> tuple[int, str | None, str | None]:
        dest = out_dir / f"{prefix}_{page:0{width}d}.jpg"

        if valid_jpeg(dest):
            return page, quality, None
        if dest.exists():
            dest.unlink()

        try:
            used_quality = download_page(image_code, quality, page, dest)
        except requests.RequestException as e:
            return page, None, str(e)

        if delay:
            time.sleep(delay)
        return page, used_quality, None

    pages = range(start, start + total_pages)
    ok, failed, fallbacks = 0, [], 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = executor.map(process_page, pages)
        for page, used_quality, error in tqdm(
            results, total=total_pages, desc=prefix, leave=False
        ):
            if error:
                failed.append((page, error))
                continue
            ok += 1
            if used_quality != quality:
                fallbacks += 1

    if fallbacks:
        print(f"  ({fallbacks} trang phải dùng bản 'jpeg' vì không có bản 'large')")

    return ok, failed


def run_batch(
    links_file: Path, catalog_path: Path, out_root: Path,
    quality: str, delay: float, workers: int,
) -> int:
    urls, catalog = validate_batch_inputs(links_file, catalog_path)
    print(f"Tìm thấy {len(urls)} volume trong {links_file}\n")

    summary = []
    for url in urls:
        try:
            volume_id, image_code, total_pages = resolve_volume(url)
        except (ValueError, requests.RequestException) as e:
            print(f"[BỎ QUA] {url} -> {e}")
            summary.append((url, None, 0, 0, str(e)))
            continue

        row = catalog.get(volume_id)
        if row is None:
            print(f"[BỎ QUA] volume {volume_id} không có trong {catalog_path}")
            summary.append((url, None, 0, total_pages, "thiếu catalog"))
            continue
        expected_pages = int(row["image_count"])
        if total_pages != expected_pages:
            print(f"[BỎ QUA] volume {volume_id}: website={total_pages}, catalog={expected_pages} trang")
            summary.append((url, None, 0, total_pages, "lệch số trang"))
            continue

        prefix = row["legacy_folder"]
        out_dir = out_root / prefix
        print(f"[{prefix}] mã ảnh={image_code}, tổng số trang={total_pages}")

        ok, failed = download_volume(
            image_code, total_pages, prefix, out_dir, quality, delay,
            workers=workers,
        )
        summary.append((url, prefix, ok, total_pages, None if not failed else f"{len(failed)} trang lỗi"))
        print(f"  -> {ok}/{total_pages} ảnh đã lưu vào {out_dir}\n")

    print("=" * 60)
    print("TỔNG KẾT")
    print("=" * 60)
    for url, prefix, ok, total, err in summary:
        status = f"{ok}/{total}" if prefix else "LỖI"
        note = f" ({err})" if err else ""
        print(f"  {prefix or url}: {status}{note}")
    failures = [item for item in summary if item[1] is None or item[2] != item[3] or item[4]]
    if failures:
        print(f"\nKHÔNG ĐẠT: {len(failures)} volume lỗi/thiếu")
        return 1
    print("\nĐẠT: 32/32 folder đã tải đủ ảnh")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--links-file", help="File .txt chứa danh sách link volume, mỗi dòng 1 link")
    parser.add_argument("--catalog", default="configs/corpus_catalog.csv",
                        help="Catalog ánh xạ volume -> work/chapter")
    parser.add_argument("--out-root", default="data/input",
                        help="Thư mục gốc lưu ảnh khi dùng --links-file (mặc định: data/input)")
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Chỉ kiểm tra link/catalog đủ 32 folder, không truy cập mạng",
    )

    parser.add_argument("--image-code", help="Mã ảnh nội bộ Nom Foundation (chế độ 1 volume thủ công)")
    parser.add_argument("--pages", type=int, help="Tổng số trang cần tải (chế độ 1 volume thủ công)")
    parser.add_argument("--start", type=int, default=1, help="Trang bắt đầu (mặc định: 1)")
    parser.add_argument("--prefix", help="Tiền tố tên file, ví dụ: HVH_001_01 (chế độ thủ công)")
    parser.add_argument("--out-dir", help="Thư mục lưu ảnh (chế độ 1 volume thủ công)")

    parser.add_argument("--quality", choices=["large", "jpeg"], default="large",
                        help="large = độ phân giải cao (khuyến nghị cho OCR), jpeg = nhẹ hơn")
    parser.add_argument("--delay", type=float, default=0.3, help="Độ trễ giữa các lần tải (giây)")
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Số ảnh tải song song trong mỗi volume (mặc định: 4)",
    )
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 12:
        parser.error("--workers phải trong khoảng 1..12")

    if args.validate_only:
        if not args.links_file:
            parser.error("--validate-only cần --links-file")
        try:
            urls, catalog = validate_batch_inputs(Path(args.links_file), Path(args.catalog))
        except (OSError, ValueError) as exc:
            print(f"LỖI: {exc}")
            return 1
        print(f"ĐẠT: {len(urls)} link khớp {len(catalog)} folder HVH_001..HVH_032")
        return 0

    if args.links_file:
        try:
            return run_batch(
                Path(args.links_file), Path(args.catalog),
                Path(args.out_root), args.quality, args.delay, args.workers,
            )
        except (OSError, ValueError, requests.RequestException) as exc:
            print(f"LỖI: {exc}")
            return 1

    missing = [name for name in ("image_code", "pages", "prefix", "out_dir") if getattr(args, name) is None]
    if missing:
        parser.error(f"Thiếu tham số cho chế độ thủ công: {', '.join('--' + m.replace('_', '-') for m in missing)}")

    ok, failed = download_volume(
        args.image_code, args.pages, args.prefix, Path(args.out_dir),
        args.quality, args.delay, args.start, workers=args.workers)

    print(f"\nHoàn tất: {ok}/{args.pages} ảnh đã lưu vào {args.out_dir}")
    if failed:
        print(f"Lỗi ({len(failed)} trang):")
        for page, err in failed:
            print(f"  - trang {page}: {err}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
