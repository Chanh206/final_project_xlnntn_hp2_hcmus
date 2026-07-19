#!/usr/bin/env python3
"""OCR ảnh local bằng PaddleOCR hoặc API, có resume và pilot A/B.

Script đọc ảnh gốc trong ``data/input`` và/hoặc ảnh tách trang trong
``data/processed``. Mỗi kết quả được ghi ngay thành một JSON riêng trong
``data/intermediate``; chạy lại sẽ bỏ qua trang đã thành công và có cùng
checksum.

Mặc định dùng PaddleOCR local. Nếu dùng ``--engine api``, cấu hình
bằng .env hoặc biến môi trường:
    OCR_API_KEY=...
    OCR_API_URL=https://example.com/v1/chat/completions
    OCR_MODEL=...

Pilot ba scan ảnh gốc và processed:
    python scripts/4_ocr_local.py --id HVH_001 --chapter 01 --source both --limit 3
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import mimetypes
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
WORK_ID_RE = re.compile(r"^HVH_\d{3}$")
CHAPTER_RE = re.compile(r"^\d{2}$")
PROCESSED_PART_RE = re.compile(r"^(.*)_p(\d{2})$")

PROMPT = """Bạn là bộ máy OCR chữ Hán/Hán Nôm, không phải người dịch hay biên tập.
Hãy chép đúng các ký tự nhìn thấy trong ảnh.

Quy tắc bắt buộc:
- Văn bản viết dọc: đọc mỗi cột từ trên xuống dưới, các cột từ phải sang trái.
- Nếu ảnh có hai trang: đọc trang bên phải trước, sau đó trang bên trái.
- Giữ nguyên chữ phồn thể, chữ Nôm và dấu câu; không giản thể hóa.
- Không dịch, không giải thích, không tự sửa theo ngữ cảnh.
- Ký tự hoàn toàn không đọc được ghi là 〓, không đoán.
- Xuống dòng giữa các cột; không thêm Markdown.

Chỉ trả về JSON hợp lệ:
{"text":"nội dung OCR", "confidence":85, "note":"ghi chú ngắn hoặc rỗng"}
Confidence là số nguyên 0-100 cho toàn ảnh."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--id", required=True, help="Work ID, ví dụ HVH_001")
    parser.add_argument("--chapter", default="01", help="Mã quyển hai chữ số")
    parser.add_argument("--engine", choices=("paddle", "api"), default="paddle")
    parser.add_argument("--source", choices=("original", "processed", "both"), default="both")
    parser.add_argument("--limit", type=int, help="Số scan gốc dùng cho pilot")
    parser.add_argument(
        "--shard-count", type=int, default=1,
        help="Chia task thành N shard độc lập để OCR song song (mặc định: 1)",
    )
    parser.add_argument(
        "--shard-index", type=int, default=0,
        help="Shard cần chạy, đánh số từ 0 (mặc định: 0)",
    )
    parser.add_argument("--catalog", default="configs/corpus_catalog.csv")
    parser.add_argument("--input-root", default="data/input")
    parser.add_argument("--processed-root", default="data/processed")
    parser.add_argument("--intermediate-root", default="data/intermediate")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--api-url", help="Ghi đè OCR_API_URL")
    parser.add_argument("--model", help="Ghi đè OCR_MODEL")
    parser.add_argument("--run-name", help="Tên thư mục run; mặc định sinh từ model")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Chỉ kiểm tra input, không cần API")
    parser.add_argument("--paddle-lang", default="chinese_cht", help="Ngôn ngữ PaddleOCR")
    parser.add_argument("--ocr-version", default="PP-OCRv5")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-cache", default="models/paddlex")
    parser.add_argument("--paddle-score-threshold", type=float, default=0.30)
    parser.add_argument(
        "--ocr-layout",
        choices=("full-page", "columns"),
        default="full-page",
        help="OCR nguyên trang hoặc tách cột dọc rồi OCR từng cột",
    )
    parser.add_argument(
        "--min-column-height-ratio", type=float, default=0.12,
        help="Chiều cao mực tối thiểu của một cột so với trang",
    )
    args = parser.parse_args()
    if not WORK_ID_RE.fullmatch(args.id):
        parser.error("--id phải có dạng HVH_NNN")
    if not CHAPTER_RE.fullmatch(args.chapter):
        parser.error("--chapter phải có hai chữ số")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit phải lớn hơn 0")
    if args.shard_count < 1:
        parser.error("--shard-count phải lớn hơn 0")
    if not 0 <= args.shard_index < args.shard_count:
        parser.error("--shard-index phải từ 0 đến --shard-count - 1")
    return args


def detect_vertical_columns(
    image_path: Path, min_height_ratio: float = 0.25
) -> list[dict[str, int]]:
    """Phát hiện cột chữ dọc, trả box theo thứ tự phải sang trái.

    Chỉ dùng hình học/mực ảnh, không sửa ảnh nguồn. Phép morphology nối các
    ký tự trong cùng cột trước khi lấy connected components.
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV không đọc được ảnh: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, ink = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Xóa component rất nhỏ trước khi nối để hạn chế bụi giấy.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    cleaned = np.zeros_like(ink)
    min_ink_area = max(4, int(height * width * 0.000003))
    max_ink_area = int(height * width * 0.008)
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        # Loại bụi nhỏ và mảng rách/nền đen rất lớn trước khi nối cột.
        if min_ink_area <= area <= max_ink_area:
            cleaned[labels == index] = 255

    kernel_width = max(3, int(width * 0.006)) | 1
    kernel_height = max(15, int(height * 0.060)) | 1
    joined = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, kernel_height)),
    )
    count, _, stats, _ = cv2.connectedComponentsWithStats(joined, 8)
    boxes: list[tuple[int, int, int, int]] = []
    for index in range(1, count):
        x, y, box_width, box_height, area = (int(v) for v in stats[index])
        if box_height < height * min_height_ratio:
            continue
        if box_width < max(8, width * 0.012) or box_width > width * 0.22:
            continue
        if area < height * width * 0.0005:
            continue
        boxes.append((x, y, x + box_width, y + box_height))

    # Gộp component cùng cột nếu khoảng x chồng lấn đáng kể.
    merged: list[list[int]] = []
    for box in sorted(boxes, key=lambda b: b[0]):
        if merged and min(merged[-1][2], box[2]) - max(merged[-1][0], box[0]) > 0:
            merged[-1][0] = min(merged[-1][0], box[0])
            merged[-1][1] = min(merged[-1][1], box[1])
            merged[-1][2] = max(merged[-1][2], box[2])
            merged[-1][3] = max(merged[-1][3], box[3])
        else:
            merged.append(list(box))

    padding_x = max(6, int(width * 0.008))
    padding_y = max(6, int(height * 0.015))
    result = [
        {
            "x1": max(0, x1 - padding_x), "y1": max(0, y1 - padding_y),
            "x2": min(width, x2 + padding_x), "y2": min(height, y2 + padding_y),
        }
        for x1, y1, x2, y2 in merged
    ]
    return sorted(result, key=lambda box: -((box["x1"] + box["x2"]) / 2))


def write_column_crops(
    image_path: Path, boxes: list[dict[str, int]], crop_dir: Path
) -> list[tuple[Path, dict[str, int]]]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV không đọc được ảnh: {image_path}")
    crop_dir.mkdir(parents=True, exist_ok=True)
    crops = []
    for index, box in enumerate(boxes, 1):
        crop = image[box["y1"]:box["y2"], box["x1"]:box["x2"]]
        path = crop_dir / f"col_{index:02d}.png"
        if not cv2.imwrite(str(path), crop):
            raise RuntimeError(f"Không ghi được crop: {path}")
        crops.append((path, box))
    return crops


def load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"").strip("'")
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            values[key] = value
    for key, value in values.items():
        os.environ.setdefault(key, value)


def catalog_entry(path: Path, work_id: str, chapter_id: str) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as handle:
        matches = [
            row
            for row in csv.DictReader(handle)
            if row.get("work_id") == work_id and row.get("chapter_id") == chapter_id
        ]
    if len(matches) != 1:
        raise ValueError(f"Catalog không có duy nhất một dòng cho {work_id}/{chapter_id}")
    return matches[0]


def natural_key(path: Path) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def image_files(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return sorted(
        (item for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES),
        key=natural_key,
    )


def processed_scan_key(path: Path) -> str:
    match = PROCESSED_PART_RE.fullmatch(path.stem)
    if not match:
        raise ValueError(f"Tên processed không có _pNN: {path.name}")
    return match.group(1)


def select_tasks(
    original_dir: Path, processed_dir: Path, source: str, limit: int | None
) -> list[tuple[str, Path]]:
    originals = image_files(original_dir)
    if not originals:
        raise ValueError(f"Không có ảnh gốc trong {original_dir}")
    selected_originals = originals[:limit] if limit else originals
    selected_keys = {path.stem for path in selected_originals}
    tasks: list[tuple[str, Path]] = []
    if source in ("original", "both"):
        tasks.extend(("original", path) for path in selected_originals)
    if source in ("processed", "both"):
        processed = image_files(processed_dir)
        if not processed:
            raise ValueError(f"Không có ảnh processed trong {processed_dir}")
        selected_processed = [path for path in processed if processed_scan_key(path) in selected_keys]
        if not selected_processed:
            raise ValueError("Không ghép được processed với scan gốc đã chọn")
        tasks.extend(("processed", path) for path in selected_processed)
    return tasks


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return slug[:80] or "ocr_run"


def api_endpoint(args: argparse.Namespace) -> tuple[str, str, str]:
    load_env_file(Path(args.env_file))
    api_key = os.environ.get("OCR_API_KEY", "")
    model = args.model or os.environ.get("OCR_MODEL", "")
    url = args.api_url or os.environ.get("OCR_API_URL", "")
    if not url:
        base = os.environ.get("OCR_BASE_URL", "").rstrip("/")
        if base:
            url = base if base.endswith("/chat/completions") else f"{base}/chat/completions"
    missing = [name for name, value in (("OCR_API_KEY", api_key), ("OCR_API_URL", url), ("OCR_MODEL", model)) if not value]
    if missing:
        raise ValueError(f"Thiếu cấu hình: {', '.join(missing)}")
    return url, api_key, model


def extract_json(content: str) -> dict[str, Any]:
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            raise ValueError("Model không trả JSON")
        data = json.loads(match.group(0))
    text = str(data.get("text", "")).strip()
    if not text:
        raise ValueError("JSON OCR không có text")
    try:
        confidence = max(0, min(100, int(data.get("confidence", 0))))
    except (TypeError, ValueError):
        confidence = 0
    return {"text": text, "confidence": confidence, "note": str(data.get("note", "")).strip()}


def call_ocr(
    url: str, api_key: str, model: str, image_path: Path, timeout: int, max_retries: int
) -> tuple[dict[str, Any], str]:
    mime = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
                ],
            }
        ],
        "temperature": 0,
    }
    wait = 2.0
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(
                url,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=timeout,
            )
            if response.status_code == 429 or response.status_code >= 500:
                raise requests.HTTPError(f"HTTP {response.status_code}: {response.text[:300]}", response=response)
            response.raise_for_status()
            body = response.json()
            raw_content = body["choices"][0]["message"]["content"]
            if not isinstance(raw_content, str):
                raise ValueError("message.content không phải chuỗi")
            return extract_json(raw_content), raw_content
        except (requests.RequestException, KeyError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt == max_retries:
                break
            time.sleep(wait)
            wait = min(wait * 2, 30)
    raise RuntimeError(str(last_error))


def init_paddle(args: argparse.Namespace) -> Any:
    cache_dir = Path(args.model_cache).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(cache_dir))
    os.environ.setdefault("MPLCONFIGDIR", str((cache_dir / "matplotlib").resolve()))
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise ValueError(
            "Chưa có PaddleOCR. Hãy chạy bằng Conda env NLP hoặc cài package paddleocr."
        ) from exc
    return PaddleOCR(
        lang=args.paddle_lang,
        ocr_version=args.ocr_version,
        device=args.device,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_rec_score_thresh=args.paddle_score_threshold,
    )


def call_paddle(ocr: Any, image_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    predictions = list(ocr.predict(str(image_path)))
    if not predictions:
        raise ValueError("PaddleOCR không trả result")
    raw = predictions[0].json
    result = raw.get("res", raw)
    texts = list(result.get("rec_texts", []))
    scores = [float(value) for value in result.get("rec_scores", [])]
    polygons = list(result.get("rec_polys", []))
    if not texts:
        return (
            {
                "text": "",
                "confidence": 0,
                "note": "PaddleOCR không phát hiện text; blank candidate",
                "lines": [],
                "is_blank": True,
            },
            result,
        )

    lines = []
    for index, text in enumerate(texts):
        polygon = polygons[index] if index < len(polygons) else []
        xs = [float(point[0]) for point in polygon] if polygon else [0.0]
        ys = [float(point[1]) for point in polygon] if polygon else [float(index)]
        center_x = sum(xs) / len(xs)
        top_y = min(ys)
        score = scores[index] if index < len(scores) else 0.0
        lines.append(
            {
                "text": str(text).strip(),
                "score": score,
                "center_x": center_x,
                "top_y": top_y,
                "polygon": polygon,
            }
        )

    # Paddle thường trả box từ trái sang phải; sách Hán dọc cần đảo
    # thứ tự cột. Gom x xấp xỉ theo bucket 40 px rồi đọc trên -> dưới.
    lines.sort(key=lambda item: (-round(item["center_x"] / 40), item["top_y"]))
    ordered_texts = [line["text"] for line in lines if line["text"]]
    confidence = round(100 * sum(line["score"] for line in lines) / len(lines), 2)
    parsed = {
        "text": "\n".join(ordered_texts),
        "confidence": confidence,
        "note": f"PaddleOCR: {len(lines)} vùng text; đã sắp thứ tự cột phải->trái",
        "lines": lines,
    }
    return parsed, result


def call_column_ocr(
    args: argparse.Namespace,
    image_path: Path,
    crop_dir: Path,
    paddle_ocr: Any,
    url: str,
    api_key: str,
    model: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    boxes = detect_vertical_columns(image_path, args.min_column_height_ratio)
    if not boxes:
        # Trang trắng, trang bìa hoặc trang hư hỏng có thể không đủ hình học
        # để tách cột. Fallback nguyên trang giúp batch không dừng; kết quả vẫn
        # ghi rõ layout thực tế để có thể audit.
        if args.engine == "paddle":
            parsed, raw = call_paddle(paddle_ocr, image_path)
        else:
            parsed, raw = call_ocr(
                url, api_key, model, image_path, args.timeout, args.max_retries
            )
        parsed["note"] = (
            "Không phát hiện cột; fallback OCR nguyên trang. "
            + str(parsed.get("note", ""))
        ).strip()
        parsed["columns"] = []
        parsed["layout_fallback"] = "full-page"
        return parsed, {
            "layout": "columns",
            "layout_fallback": "full-page",
            "columns": [],
            "full_page_response": raw,
        }
    crops = write_column_crops(image_path, boxes, crop_dir)
    columns: list[dict[str, Any]] = []
    for index, (crop_path, box) in enumerate(crops, 1):
        if args.engine == "paddle":
            parsed, raw = call_paddle(paddle_ocr, crop_path)
        else:
            parsed, raw = call_ocr(
                url, api_key, model, crop_path, args.timeout, args.max_retries
            )
        columns.append(
            {
                "column_index": index,
                "crop_path": crop_path.as_posix(),
                "box": box,
                "text": str(parsed.get("text", "")).replace("\n", "").strip(),
                "confidence": float(parsed.get("confidence", 0)),
                "is_blank": bool(parsed.get("is_blank", False)),
                "raw_response": raw,
            }
        )
    nonblank = [column for column in columns if column["text"]]
    if not nonblank:
        # Có box cột không đồng nghĩa các box đó hợp lệ. Với trang in hoặc bố
        # cục khác tập pilot, detector có thể gom sai vùng và làm mọi crop OCR
        # rỗng. Thử lại ảnh nguyên trang trước khi kết luận đây là trang blank.
        if args.engine == "paddle":
            full_page_parsed, full_page_raw = call_paddle(paddle_ocr, image_path)
        else:
            full_page_parsed, full_page_raw = call_ocr(
                url, api_key, model, image_path, args.timeout, args.max_retries
            )
        full_page_parsed["note"] = (
            f"Đã tách {len(columns)} cột nhưng tất cả OCR rỗng; "
            "fallback OCR nguyên trang. "
            + str(full_page_parsed.get("note", ""))
        ).strip()
        full_page_parsed["columns"] = [
            {key: value for key, value in column.items() if key != "raw_response"}
            for column in columns
        ]
        full_page_parsed["layout_fallback"] = "full-page-after-empty-columns"
        return full_page_parsed, {
            "layout": "columns",
            "layout_fallback": "full-page-after-empty-columns",
            "columns": columns,
            "full_page_response": full_page_raw,
        }
    weights = [max(1, len(column["text"])) for column in nonblank]
    confidence = sum(
        column["confidence"] * weight for column, weight in zip(nonblank, weights)
    ) / sum(weights)
    lines = [
        {
            "text": column["text"],
            "score": column["confidence"] / 100,
            "center_x": (column["box"]["x1"] + column["box"]["x2"]) / 2,
            "top_y": column["box"]["y1"],
            "polygon": [
                [column["box"]["x1"], column["box"]["y1"]],
                [column["box"]["x2"], column["box"]["y1"]],
                [column["box"]["x2"], column["box"]["y2"]],
                [column["box"]["x1"], column["box"]["y2"]],
            ],
            "column_index": column["column_index"],
        }
        for column in nonblank
    ]
    parsed = {
        "text": "\n".join(column["text"] for column in nonblank),
        "confidence": round(confidence, 2),
        "note": f"OCR {len(nonblank)}/{len(columns)} cột, thứ tự phải->trái",
        "lines": lines,
        "columns": [
            {key: value for key, value in column.items() if key != "raw_response"}
            for column in columns
        ],
    }
    return parsed, {"layout": "columns", "columns": columns}


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def reusable_result(
    path: Path, image_hash: str, retry_errors: bool, ocr_layout: str
) -> str | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if data.get("image_sha256") != image_hash:
        return None
    if data.get("ocr_layout", "full-page") != ocr_layout:
        return None
    status = data.get("status")
    if status == "blank" and ocr_layout == "columns":
        raw_response = data.get("raw_response", {})
        # Các kết quả tạo bởi phiên bản cũ có thể kết luận BLANK ngay khi đã
        # phát hiện box cột nhưng mọi crop đều rỗng. Không reuse các JSON đó:
        # chạy lại đúng các trang này để áp dụng fallback nguyên trang mới.
        if (
            isinstance(raw_response, dict)
            and raw_response.get("columns")
            and not raw_response.get("layout_fallback")
        ):
            return "retry-empty-columns"
    if status in {"success", "blank"}:
        return str(status)
    return None if retry_errors else "error"


def main() -> int:
    args = parse_args()
    chapter_id = f"{args.id}_{args.chapter}"
    try:
        catalog = catalog_entry(Path(args.catalog), args.id, chapter_id)
        original_dir = Path(args.input_root) / args.id / chapter_id
        processed_dir = Path(args.processed_root) / args.id / chapter_id
        tasks = select_tasks(original_dir, processed_dir, args.source, args.limit)
        if args.shard_count > 1:
            tasks = [
                task for task_index, task in enumerate(tasks)
                if task_index % args.shard_count == args.shard_index
            ]
    except (FileNotFoundError, ValueError) as exc:
        print(f"LỖI: {exc}", file=sys.stderr)
        return 1

    counts = {kind: sum(1 for task_kind, _ in tasks if task_kind == kind) for kind in ("original", "processed")}
    print(f"[{chapter_id}] {catalog['title_vietnamese']}")
    print(f"Tasks: original={counts['original']}, processed={counts['processed']}")
    if args.dry_run:
        for kind, path in tasks:
            print(f"  {kind:9s} {path}")
        print("DRY-RUN THÀNH CÔNG; chưa gọi OCR.")
        return 0

    url = api_key = ""
    paddle_ocr = None
    try:
        if args.engine == "api":
            url, api_key, model = api_endpoint(args)
        else:
            model = f"paddle_{args.ocr_version}_{args.paddle_lang}"
            paddle_ocr = init_paddle(args)
    except ValueError as exc:
        print(f"LỖI: {exc}", file=sys.stderr)
        return 2

    run_name = safe_slug(args.run_name or model)
    run_root = Path(args.intermediate_root) / args.id / chapter_id / "ocr_runs" / run_name
    stats = {
        "success": 0,
        "blank": 0,
        "error": 0,
        "skipped_success": 0,
        "skipped_blank": 0,
        "skipped_error": 0,
    }

    for index, (kind, image_path) in enumerate(tasks, 1):
        output_path = run_root / kind / f"{image_path.stem}.json"
        image_hash = sha256_file(image_path)
        reusable = None if args.overwrite else reusable_result(
            output_path, image_hash, args.retry_errors, args.ocr_layout
        )
        retry_empty_columns = reusable == "retry-empty-columns"
        if reusable and not retry_empty_columns:
            key = f"skipped_{reusable}"
            stats[key] += 1
            print(f"[{index}/{len(tasks)}] SKIP {kind}/{image_path.name} ({reusable})")
            continue

        started = time.monotonic()
        base_result: dict[str, Any] = {
            "chapter_id": chapter_id,
            "source_kind": kind,
            "image_path": image_path.as_posix(),
            "image_sha256": image_hash,
            "model": model,
            "ocr_layout": args.ocr_layout,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        try:
            if retry_empty_columns:
                # JSON cũ đã chứng minh các crop cột đều OCR rỗng. Không tốn
                # thời gian OCR lại những crop đó; dùng trực tiếp fallback
                # nguyên trang và giữ dữ liệu cột cũ để audit.
                previous = json.loads(output_path.read_text(encoding="utf-8"))
                previous_raw = previous.get("raw_response", {})
                old_columns = (
                    previous_raw.get("columns", [])
                    if isinstance(previous_raw, dict)
                    else []
                )
                if args.engine == "paddle":
                    parsed, full_page_raw = call_paddle(paddle_ocr, image_path)
                else:
                    parsed, full_page_raw = call_ocr(
                        url, api_key, model, image_path,
                        args.timeout, args.max_retries,
                    )
                parsed["note"] = (
                    f"Resume từ {len(old_columns)} cột OCR rỗng; "
                    "fallback OCR nguyên trang. "
                    + str(parsed.get("note", ""))
                ).strip()
                parsed["columns"] = [
                    {key: value for key, value in column.items() if key != "raw_response"}
                    for column in old_columns
                    if isinstance(column, dict)
                ]
                parsed["layout_fallback"] = "full-page-after-empty-columns"
                raw_response = {
                    "layout": "columns",
                    "layout_fallback": "full-page-after-empty-columns",
                    "columns": old_columns,
                    "full_page_response": full_page_raw,
                }
            elif args.ocr_layout == "columns":
                crop_dir = run_root / "column_crops" / kind / image_path.stem
                parsed, raw_response = call_column_ocr(
                    args, image_path, crop_dir, paddle_ocr, url, api_key, model
                )
            elif args.engine == "paddle":
                parsed, raw_response = call_paddle(paddle_ocr, image_path)
            else:
                parsed, raw_response = call_ocr(url, api_key, model, image_path, args.timeout, args.max_retries)
            is_blank = bool(parsed.pop("is_blank", False))
            result_status = "blank" if is_blank else "success"
            result = {
                **base_result,
                "status": result_status,
                **parsed,
                "raw_response": raw_response,
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
            stats[result_status] += 1
            label = "BLANK" if is_blank else "OK"
            print(f"[{index}/{len(tasks)}] {label:5s} {kind}/{image_path.name} confidence={parsed['confidence']}")
        except Exception as exc:
            result = {
                **base_result,
                "status": "error",
                "error": str(exc),
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
            stats["error"] += 1
            print(f"[{index}/{len(tasks)}] LỖI {kind}/{image_path.name}: {exc}", file=sys.stderr)
        write_json_atomic(output_path, result)
        if args.delay and index < len(tasks):
            time.sleep(args.delay)

    invocation_stats = stats
    final_status_counts = {"success": 0, "blank": 0, "error": 0, "missing": 0}
    for kind, image_path in tasks:
        result_path = run_root / kind / f"{image_path.stem}.json"
        if not result_path.is_file():
            final_status_counts["missing"] += 1
            continue
        try:
            final_status = json.loads(result_path.read_text(encoding="utf-8")).get("status", "error")
        except (OSError, json.JSONDecodeError):
            final_status = "error"
        if final_status not in final_status_counts:
            final_status = "error"
        final_status_counts[final_status] += 1

    summary = {
        "chapter_id": chapter_id,
        "title_vietnamese": catalog["title_vietnamese"],
        "model": model,
        "engine": args.engine,
        "run_name": run_name,
        "source": args.source,
        "ocr_layout": args.ocr_layout,
        "task_counts": counts,
        "stats": final_status_counts,
        "invocation_stats": invocation_stats,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    summary_path = (
        run_root / f"run_summary.shard_{args.shard_index:03d}_of_{args.shard_count:03d}.json"
        if args.shard_count > 1
        else run_root / "run_summary.json"
    )
    write_json_atomic(summary_path, summary)
    print(f"Tổng kết: {summary_path}")
    return 1 if final_status_counts["error"] or final_status_counts["missing"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
