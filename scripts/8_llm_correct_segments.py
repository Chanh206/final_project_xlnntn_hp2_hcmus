#!/usr/bin/env python3
"""Dùng vision LLM hiệu chỉnh OCR và tách câu theo từng trang, có resume.

Script không sửa ``_raw.txt``. Kết quả từng trang được lưu trong
``data/intermediate/.../llm_corrections``. Chỉ ``--publish`` khi toàn bộ
trang đã qua các guard tự động mới thay các file ``_seg.tsv`` theo từng
ảnh scan chính thức.

Ví dụ pilot không gọi API:
    python scripts/8_llm_correct_segments.py --folder HVH_001 \
      --ocr-run paddle_full_v1 --limit 3 --dry-run

Pilot thật:
    python scripts/8_llm_correct_segments.py --folder HVH_001 \
      --ocr-run paddle_full_v1 --provider ollama --model qwen3-vl:8b \
      --llm-run qwen3vl8b_pilot_v1 --limit 3
"""

from __future__ import annotations

import argparse
import base64
import csv
import difflib
import hashlib
import json
import mimetypes
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


PROMPT_VERSION = "han_nom_vision_correction_v1"
WORK_ID_RE = re.compile(r"^HVH_\d{3}$")
CHAPTER_RE = re.compile(r"^\d{2}$")
LEGACY_FOLDER_RE = re.compile(r"^HVH_(\d{3})$")
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
INCLUDE_CATEGORIES = {"sentence", "title", "heading"}
JAPANESE_SHINJITAI = {"竜", "録"}
# Các dạng giản thể xuất hiện nhiều trong pilot dù prompt yêu cầu giữ mặt chữ.
# Đây là guard bảo thủ: chỉ yêu cầu review, không tự động đổi ký tự.
SUSPICIOUS_SIMPLIFIED = {
    "爱", "边", "长", "发", "红", "来", "气", "强", "为", "无", "远",
    "运", "龙", "庄", "带", "帐", "计", "贴", "头", "护", "绵", "许",
    "莲", "术", "详", "亲", "离", "湾", "异", "体", "万", "后", "国",
}
SIMPLIFIED_TRADITIONAL_PAIRS = (
    ("爱", "愛"), ("来", "來"), ("种", "種"), ("详", "詳"),
    ("发", "發"), ("长", "長"), ("无", "無"), ("边", "邊"),
    ("带", "帶"), ("帐", "帳"), ("计", "計"), ("贴", "貼"),
    ("头", "頭"), ("运", "運"), ("远", "遠"), ("寻", "尋"),
    ("术", "術"), ("莲", "蓮"), ("许", "許"), ("绵", "綿"),
    ("护", "護"), ("红", "紅"), ("脉", "脈"), ("规", "規"),
    ("强", "強"), ("异", "異"), ("气", "氣"), ("献", "獻"),
    ("维", "維"), ("贵", "貴"), ("龙", "龍"), ("为", "為"),
)


class FatalAPIError(RuntimeError):
    """Lỗi cấu hình/quota sẽ không tự hết nếu retry ngay."""

SYSTEM_PROMPT = """Bạn là chuyên gia văn bản học Hán Nôm và OCR bản thảo Việt Nam.
Nhiệm vụ là đối chiếu trực tiếp ảnh với OCR máy, sửa ký tự sai và tách câu.

Quy tắc bắt buộc:
1. Đọc chữ dọc: mỗi cột từ trên xuống, các cột từ phải sang trái.
2. Ảnh chỉ chứa một trang đã được tách; không đảo thứ tự cột.
3. Giữ đúng mặt chữ phồn thể/chữ Nôm nhìn thấy. Không giản thể hóa, không
   đổi sang dạng Nhật, không dịch, không diễn giải và không hiện đại hóa.
4. OCR đầu vào chỉ là gợi ý. Khi OCR và ảnh khác nhau, ưu tiên ảnh.
5. Dùng các dấu son/dấu đỏ làm bằng chứng ngắt câu nhưng phân biệt dấu câu,
   con dấu, khung đỏ, gạch tiêu đề và vết mực.
6. Không tự bổ sung từ không nhìn thấy. Ký tự không chắc chắn ghi 〓 và đặt
   confidence=low; giải thích ngắn trong note.
7. page_transcription chứa toàn bộ phần chữ đã phân loại, theo đúng thứ tự.
   Nối text của tất cả segments sau khi bỏ whitespace phải bằng
   page_transcription sau khi bỏ whitespace.
8. page_number, con dấu thư viện và ghi chú hiện đại được phân loại riêng,
   không trộn vào câu chính.
9. Mỗi segment chỉ chứa một câu/tiêu đề/đơn vị phụ; không dùng Markdown.
"""

OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "page_transcription": {"type": "string"},
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "category": {
                        "type": "string",
                        "enum": [
                            "sentence", "title", "heading", "page_number",
                            "annotation", "stamp", "uncertain",
                        ],
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                    },
                    "note": {"type": "string"},
                },
                "required": ["text", "category", "confidence", "note"],
                "additionalProperties": False,
            },
        },
        "page_note": {"type": "string"},
    },
    "required": ["page_transcription", "segments", "page_note"],
    "additionalProperties": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--id", help="Mã xử lý nội bộ, ví dụ HVH_018")
    parser.add_argument("--folder", help="Folder nguồn 1-1, ví dụ HVH_019")
    parser.add_argument("--chapter", help="Mã quyển nội bộ; chỉ dùng cùng --id")
    parser.add_argument("--ocr-run", required=True, help="Ví dụ paddle_full_v1")
    parser.add_argument("--llm-run", default="vision_correction_v1")
    parser.add_argument(
        "--provider",
        choices=("openai", "ollama", "compatible"),
        default="openai",
        help="openai: Responses API; ollama: local; compatible: /chat/completions",
    )
    parser.add_argument(
        "--model",
        help="Mặc định: LLM_CORRECTION_MODEL/gpt-5.5 hoặc OLLAMA_MODEL/qwen3-vl:8b",
    )
    parser.add_argument(
        "--api-url",
        help="Mặc định theo provider: OpenAI /v1/responses hoặc Ollama /api/chat",
    )
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--catalog", default="configs/corpus_catalog.csv")
    parser.add_argument("--processed-root", default="data/processed")
    parser.add_argument("--intermediate-root", default="data/intermediate")
    parser.add_argument(
        "--output-root",
        default="final_output",
        help="Thư mục nộp chung (mặc định: final_output)",
    )
    parser.add_argument("--detail", choices=("high", "original"), default="original")
    parser.add_argument(
        "--ocr-guidance",
        choices=("full", "none"),
        default="full",
        help="full: đưa Paddle vào prompt; none: đọc ảnh độc lập để tránh neo OCR sai",
    )
    parser.add_argument("--limit", type=int, help="Chỉ xử lý N trang đầu để pilot")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--max-output-tokens", type=int, default=6000)
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--min-similarity", type=float, default=0.80)
    parser.add_argument(
        "--min-segment-candidate-ratio",
        type=float,
        default=0.30,
        help="Tối thiểu số segment chính / số điểm ngắt đỏ gợi ý",
    )
    parser.add_argument("--min-length-ratio", type=float, default=0.45)
    parser.add_argument("--max-length-ratio", type=float, default=1.55)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--retry-review", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Ghi _seg.tsv chính thức; chỉ được phép khi chạy đủ và tất cả trang đạt guard",
    )
    args = parser.parse_args()
    if bool(args.id) == bool(args.folder):
        parser.error("Phải chọn đúng một trong --folder hoặc --id")
    if args.folder:
        if args.chapter:
            parser.error("Không dùng --chapter cùng --folder")
        if not WORK_ID_RE.fullmatch(args.folder):
            parser.error("--folder phải có dạng HVH_NNN")
        with Path(args.catalog).open(encoding="utf-8", newline="") as handle:
            matches = [row for row in csv.DictReader(handle) if row.get("legacy_folder") == args.folder]
        if len(matches) != 1:
            parser.error(f"Catalog không có duy nhất một dòng cho --folder {args.folder}")
        args.id = matches[0]["work_id"]
        args.chapter = matches[0]["chapter_id"].rsplit("_", 1)[-1]
    else:
        args.chapter = args.chapter or "01"
        if not WORK_ID_RE.fullmatch(args.id):
            parser.error("--id phải có dạng HVH_NNN")
        if not CHAPTER_RE.fullmatch(args.chapter):
            parser.error("--chapter phải có hai chữ số")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit phải lớn hơn 0")
    if args.publish and args.limit is not None:
        parser.error("Không thể --publish khi đang dùng --limit")
    if not 0 <= args.min_similarity <= 1:
        parser.error("--min-similarity phải trong khoảng 0..1")
    return args


def catalog_entry(path: Path, work_id: str, chapter_id: str) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as handle:
        matches = [
            row for row in csv.DictReader(handle)
            if row.get("work_id") == work_id and row.get("chapter_id") == chapter_id
        ]
    if len(matches) != 1:
        raise ValueError(f"Catalog không có duy nhất một dòng cho {work_id}/{chapter_id}")
    return matches[0]


def submission_group_id(catalog: dict[str, str]) -> str:
    group_id = str(catalog.get("legacy_folder", ""))
    if not LEGACY_FOLDER_RE.fullmatch(group_id):
        raise ValueError("legacy_folder không hợp lệ trong catalog")
    return group_id


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
        value = value.strip().strip('"').strip("'")
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            # Nếu file có key lặp, dùng dòng cuối; biến môi trường bên ngoài
            # vẫn được ưu tiên hơn file .env.
            values[key] = value
    for key, value in values.items():
        os.environ.setdefault(key, value)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def clean_text(text: str) -> str:
    return re.sub(r"\s+", "", text)


def cjk_ratio(text: str) -> float:
    chars = [char for char in text if not char.isspace()]
    return sum(bool(CJK_RE.fullmatch(char)) for char in chars) / max(1, len(chars))


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(text, encoding="utf-8", newline="\n")
    temp.replace(path)


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    write_text_atomic(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def load_manifest(path: Path, group_id: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy manifest: {path}")
    pages: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            image = Path(row.get("processed_image", ""))
            source = Path(row.get("source_image", ""))
            if not image.is_file():
                raise FileNotFoundError(f"Không tìm thấy ảnh processed: {image}")
            number_match = re.search(r"_(\d{4})$", source.stem)
            if not number_match:
                raise ValueError(f"Không lấy được số ảnh từ source_image: {source}")
            item_id = f"{group_id}_{int(number_match.group(1)):04d}"
            pages.append({"stem": image.stem, "image_path": image, "item_id": item_id})
    if not pages or len(pages) != len({page["stem"] for page in pages}):
        raise ValueError("Manifest rỗng hoặc có processed_image trùng")
    return pages


def load_ocr(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Thiếu OCR JSON: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("status") not in {"success", "blank"}:
        raise ValueError(f"OCR chưa thành công: {path}")
    return data


def load_red_candidates(path: Path) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    if not path.is_file():
        return grouped
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            sentence = row.get("sentence", "").strip()
            for stem in row.get("source_pages", "").split(","):
                if stem and sentence:
                    grouped.setdefault(stem, []).append(sentence)
    return grouped


def input_fingerprint(
    image_hash: str,
    ocr_text: str,
    candidates: list[str],
    provider: str,
    model: str,
    detail: str,
) -> str:
    payload = json.dumps(
        {
            "image_sha256": image_hash,
            "ocr": ocr_text,
            "candidates": candidates,
            "provider": provider,
            "model": model,
            "detail": detail,
            "prompt_version": PROMPT_VERSION,
        },
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    return sha256_bytes(payload)


def reusable_status(path: Path, fingerprint: str, retry_review: bool) -> str | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if data.get("input_fingerprint") != fingerprint:
        return None
    status = str(data.get("status", ""))
    if status in {"accepted", "blank"}:
        return status
    if status == "review" and not retry_review:
        return status
    return None


def make_user_text(
    stem: str, ocr_text: str, candidates: list[str], ocr_guidance: str = "full"
) -> str:
    candidate_text = "\n".join(f"{index}. {text}" for index, text in enumerate(candidates, 1))
    if not candidate_text:
        candidate_text = "(không có)"
    ocr_block = (
        ocr_text
        if ocr_guidance == "full"
        else "(không cung cấp; hãy đọc độc lập trực tiếp từ ảnh)"
    )
    return f"""Trang: {stem}

OCR Paddle theo thứ tự cột:
---
{ocr_block}
---

Các điểm ngắt tự động từ dấu đỏ (chỉ là gợi ý, có thể sai):
---
{candidate_text}
---

Hãy đối chiếu ảnh, chép lại đúng mặt chữ và trả kết quả theo schema."""


def extract_output_text(body: dict[str, Any]) -> str:
    for item in body.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text" and content.get("text"):
                return str(content["text"])
            if content.get("type") == "refusal":
                raise ValueError(f"Model từ chối: {content.get('refusal', '')}")
    raise ValueError("Responses API không có output_text")


def extract_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Model compatible không trả JSON object")
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("JSON của model compatible không phải object")
    return parsed


def call_openai(
    args: argparse.Namespace,
    api_key: str,
    model: str,
    image_path: Path,
    user_text: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    mime = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    payload = {
        "model": model,
        "store": False,
        "instructions": SYSTEM_PROMPT,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_text},
                    {
                        "type": "input_image",
                        "image_url": f"data:{mime};base64,{encoded}",
                        "detail": args.detail,
                    },
                ],
            }
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "han_nom_page_correction",
                "strict": True,
                "schema": OUTPUT_SCHEMA,
            }
        },
        "max_output_tokens": args.max_output_tokens,
    }
    wait = 2.0
    last_error: Exception | None = None
    for attempt in range(1, args.max_retries + 1):
        try:
            response = requests.post(
                args.api_url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=args.timeout,
            )
            if response.status_code >= 400:
                message = f"HTTP {response.status_code}: {response.text[:500]}"
                try:
                    error_code = response.json().get("error", {}).get("code", "")
                except (ValueError, AttributeError):
                    error_code = ""
                if response.status_code in {400, 401, 403, 404} or error_code in {
                    "insufficient_quota", "model_not_found", "invalid_api_key",
                }:
                    raise FatalAPIError(message)
                raise requests.HTTPError(message, response=response)
            body = response.json()
            parsed = json.loads(extract_output_text(body))
            return parsed, body
        except FatalAPIError:
            raise
        except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt == args.max_retries:
                break
            time.sleep(wait)
            wait = min(wait * 2, 30)
    raise RuntimeError(str(last_error))


def call_ollama(
    args: argparse.Namespace,
    model: str,
    image_path: Path,
    user_text: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    schema_text = json.dumps(OUTPUT_SCHEMA, ensure_ascii=False)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"{user_text}\n\nJSON schema bắt buộc:\n{schema_text}"
                ),
                "images": [encoded],
            },
        ],
        "format": OUTPUT_SCHEMA,
        "stream": False,
        "think": False,
        "keep_alive": "10m",
        "options": {
            "temperature": 0,
            "num_predict": args.max_output_tokens,
        },
    }
    wait = 2.0
    last_error: Exception | None = None
    for attempt in range(1, args.max_retries + 1):
        try:
            response = requests.post(args.api_url, json=payload, timeout=args.timeout)
            if response.status_code >= 400:
                message = f"HTTP {response.status_code}: {response.text[:500]}"
                if response.status_code in {400, 404}:
                    raise FatalAPIError(message)
                raise requests.HTTPError(message, response=response)
            body = response.json()
            message = body.get("message", {})
            content = str(message.get("content", "")).strip()
            # Qwen3-VL trên một số bản Ollama 0.31.x trả structured JSON
            # trong message.thinking dù request đã đặt think=false.
            if not content:
                content = str(message.get("thinking", "")).strip()
            if not content:
                raise ValueError(
                    "Ollama không trả message.content hoặc message.thinking; "
                    f"done_reason={body.get('done_reason')}"
                )
            return json.loads(content), body
        except FatalAPIError:
            raise
        except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt == args.max_retries:
                break
            time.sleep(wait)
            wait = min(wait * 2, 30)
    raise RuntimeError(str(last_error))


def call_compatible(
    args: argparse.Namespace,
    api_key: str,
    model: str,
    image_path: Path,
    user_text: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    mime = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    output_example = (
        '{"page_transcription":"toàn bộ chữ theo thứ tự đọc",'
        '"segments":[{"text":"một câu","category":"sentence",'
        '"confidence":"medium","note":""}],"page_note":""}'
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"{user_text}\n\nChỉ trả về một JSON object, không Markdown. "
                            "Không lặp lại mô tả/schema. Điền dữ liệu thật theo mẫu: "
                            f"{output_example}"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{encoded}"},
                    },
                ],
            },
        ],
        "temperature": 0,
        "max_tokens": args.max_output_tokens,
    }
    wait = 2.0
    last_error: Exception | None = None
    for attempt in range(1, args.max_retries + 1):
        try:
            response = requests.post(
                args.api_url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=args.timeout,
            )
            if response.status_code >= 400:
                message = f"HTTP {response.status_code}: {response.text[:500]}"
                if response.status_code in {400, 401, 403, 404}:
                    raise FatalAPIError(message)
                raise requests.HTTPError(message, response=response)
            body = response.json()
            content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
            if isinstance(content, list):
                content = "".join(
                    str(item.get("text", ""))
                    for item in content
                    if isinstance(item, dict)
                )
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Compatible API không trả message.content")
            parsed = extract_json_object(content)
            if "page_transcription" not in parsed or "segments" not in parsed:
                raise ValueError("Model lặp schema hoặc thiếu page_transcription/segments")
            return parsed, body
        except FatalAPIError:
            raise
        except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt == args.max_retries:
                break
            time.sleep(wait)
            wait = min(wait * 2, 30)
    raise RuntimeError(str(last_error))


def call_llm(
    args: argparse.Namespace,
    api_key: str,
    model: str,
    image_path: Path,
    user_text: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if args.provider == "ollama":
        return call_ollama(args, model, image_path, user_text)
    if args.provider == "compatible":
        return call_compatible(args, api_key, model, image_path, user_text)
    return call_openai(args, api_key, model, image_path, user_text)


def validate_correction(
    parsed: dict[str, Any],
    ocr_text: str,
    args: argparse.Namespace,
    red_candidate_count: int = 0,
) -> tuple[list[str], dict[str, Any]]:
    issues: list[str] = []
    transcription = str(parsed.get("page_transcription", "")).strip()
    segments = parsed.get("segments")
    if not transcription:
        issues.append("page_transcription rỗng")
    if not isinstance(segments, list) or not segments:
        issues.append("segments rỗng/không hợp lệ")
        segments = []

    segment_texts: list[str] = []
    included_count = 0
    low_count = 0
    for index, segment in enumerate(segments, 1):
        if not isinstance(segment, dict):
            issues.append(f"segment {index} không phải object")
            continue
        text = str(segment.get("text", "")).strip()
        category = str(segment.get("category", ""))
        confidence = str(segment.get("confidence", ""))
        if not text:
            issues.append(f"segment {index} rỗng")
        if "\t" in text or "\n" in text:
            issues.append(f"segment {index} chứa tab/newline")
        segment_texts.append(text)
        if category in INCLUDE_CATEGORIES:
            included_count += 1
        if confidence == "low" or "〓" in text:
            low_count += 1

    normalized_ocr = clean_text(ocr_text)
    normalized_corrected = clean_text(transcription)
    normalized_segments = clean_text("".join(segment_texts))
    if normalized_segments != normalized_corrected:
        issues.append("Nối segments không khớp page_transcription")
    similarity = difflib.SequenceMatcher(None, normalized_ocr, normalized_corrected).ratio()
    length_ratio = len(normalized_corrected) / max(1, len(normalized_ocr))
    if similarity < args.min_similarity:
        issues.append(
            f"similarity OCR/corrected {similarity:.3f} < {args.min_similarity:.3f}"
        )
    if not args.min_length_ratio <= length_ratio <= args.max_length_ratio:
        issues.append(
            f"length ratio {length_ratio:.3f} ngoài "
            f"{args.min_length_ratio:.3f}..{args.max_length_ratio:.3f}"
        )
    if included_count == 0:
        issues.append("Không có sentence/title/heading để xuất seg")
    candidate_ratio = included_count / max(1, red_candidate_count)
    if (
        red_candidate_count
        and candidate_ratio < args.min_segment_candidate_ratio
    ):
        issues.append(
            f"segment/red-candidate ratio {candidate_ratio:.3f} < "
            f"{args.min_segment_candidate_ratio:.3f}; có thể model chỉ tách theo cột"
        )
    suspicious_japanese = sorted(set(normalized_corrected) & JAPANESE_SHINJITAI)
    if suspicious_japanese:
        issues.append(
            "Có dạng Nhật đáng ngờ cần đối chiếu mặt chữ: "
            + ", ".join(suspicious_japanese)
        )
    suspicious_simplified = sorted(set(normalized_corrected) & SUSPICIOUS_SIMPLIFIED)
    if suspicious_simplified:
        issues.append(
            "Có dạng giản thể đáng ngờ cần đối chiếu mặt chữ: "
            + ", ".join(suspicious_simplified)
        )
    simplification_regressions: list[str] = []
    for simplified, traditional in SIMPLIFIED_TRADITIONAL_PAIRS:
        if (
            traditional in normalized_ocr
            and normalized_corrected.count(simplified) > normalized_ocr.count(simplified)
            and normalized_corrected.count(traditional) < normalized_ocr.count(traditional)
        ):
            simplification_regressions.append(f"{traditional}→{simplified}")
    if simplification_regressions:
        issues.append(
            "Có dấu hiệu giản thể hóa so với OCR: "
            + ", ".join(simplification_regressions)
        )
    metrics = {
        "ocr_characters": len(normalized_ocr),
        "corrected_characters": len(normalized_corrected),
        "similarity": round(similarity, 4),
        "length_ratio": round(length_ratio, 4),
        "cjk_ratio": round(cjk_ratio(transcription), 4),
        "segment_count": len(segments),
        "included_segment_count": included_count,
        "red_candidate_count": red_candidate_count,
        "segment_candidate_ratio": round(candidate_ratio, 4),
        "low_confidence_segment_count": low_count,
        "suspicious_simplified_count": len(suspicious_simplified),
    }
    return issues, metrics


def correction_segments(data: dict[str, Any]) -> list[dict[str, str]]:
    output = data.get("correction", {})
    segments = output.get("segments", []) if isinstance(output, dict) else []
    return [
        segment
        for segment in segments
        if isinstance(segment, dict)
        and segment.get("category") in INCLUDE_CATEGORIES
        and str(segment.get("text", "")).strip()
    ]


def main() -> int:
    args = parse_args()
    chapter_id = f"{args.id}_{args.chapter}"
    try:
        catalog = catalog_entry(Path(args.catalog), args.id, chapter_id)
        output_group_id = submission_group_id(catalog)
    except (FileNotFoundError, ValueError) as exc:
        print(f"LỖI: {exc}", file=sys.stderr)
        return 1
    load_env_file(Path(args.env_file))
    if args.provider == "ollama":
        model = args.model or os.environ.get("OLLAMA_MODEL", "qwen3-vl:8b")
        args.api_url = args.api_url or os.environ.get(
            "OLLAMA_API_URL", "http://127.0.0.1:11434/api/chat"
        )
    elif args.provider == "compatible":
        model = args.model or os.environ.get("OCR_MODEL", "")
        base_url = os.environ.get("OCR_BASE_URL", "").rstrip("/")
        args.api_url = args.api_url or (
            base_url
            if base_url.endswith("/chat/completions")
            else f"{base_url}/chat/completions" if base_url else ""
        )
    else:
        model = args.model or os.environ.get("LLM_CORRECTION_MODEL", "gpt-5.5")
        args.api_url = args.api_url or "https://api.openai.com/v1/responses"
    api_key = (
        os.environ.get("OCR_API_KEY", "")
        if args.provider == "compatible"
        else os.environ.get("OPENAI_API_KEY", "")
    )

    processed_dir = Path(args.processed_root) / args.id / chapter_id
    manifest_path = processed_dir / "manifest.csv"
    ocr_dir = (
        Path(args.intermediate_root)
        / args.id
        / chapter_id
        / "ocr_runs"
        / args.ocr_run
        / "processed"
    )
    build_report_dir = Path(args.intermediate_root) / args.id / chapter_id / "build_reports"
    review_path = build_report_dir / f"{args.ocr_run}.segmentation_review.tsv"
    correction_root = (
        Path(args.intermediate_root)
        / args.id
        / chapter_id
        / "llm_corrections"
        / args.llm_run
    )
    page_dir = correction_root / "pages"

    try:
        all_pages = load_manifest(manifest_path, output_group_id)
        expected_items = {
            f"{output_group_id}_{number:04d}"
            for number in range(1, int(catalog["image_count"]) + 1)
        }
        actual_items = {str(page["item_id"]) for page in all_pages}
        if actual_items != expected_items:
            raise ValueError(
                f"Manifest chưa đủ {len(expected_items)} ảnh nguồn của {output_group_id}"
            )
        red_candidates = load_red_candidates(review_path)
        tasks: list[dict[str, Any]] = []
        for page in all_pages:
            stem = str(page["stem"])
            image_path = Path(page["image_path"])
            ocr_path = ocr_dir / f"{stem}.json"
            ocr = load_ocr(ocr_path)
            tasks.append(
                {
                    "stem": stem,
                    "image_path": image_path,
                    "item_id": page["item_id"],
                    "ocr_path": ocr_path,
                    "ocr": ocr,
                    "candidates": red_candidates.get(stem, []),
                }
            )
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"LỖI: {exc}", file=sys.stderr)
        return 1

    selected = tasks[: args.limit] if args.limit else tasks
    print(
        f"[{chapter_id}] pages={len(selected)}/{len(tasks)} "
        f"provider={args.provider} model={model} detail={args.detail}"
    )
    print(f"Correction run: {correction_root}")
    if args.dry_run:
        for task in selected:
            status = task["ocr"].get("status")
            print(
                f"  {task['stem']} OCR={status} candidates={len(task['candidates'])} "
                f"image={task['image_path']}"
            )
        print("DRY-RUN THÀNH CÔNG; chưa gọi API và chưa sửa output.")
        return 0
    if args.provider in {"openai", "compatible"} and not api_key:
        print(
            f"LỖI: Thiếu {'OCR_API_KEY' if args.provider == 'compatible' else 'OPENAI_API_KEY'} "
            "trong .env hoặc biến môi trường. "
            "Không ghi key trực tiếp vào code.",
            file=sys.stderr,
        )
        return 2
    if args.provider == "compatible" and (not model or not args.api_url):
        print("LỖI: Thiếu OCR_MODEL hoặc OCR_BASE_URL cho provider compatible.", file=sys.stderr)
        return 2

    invocation = Counter()
    fatal_api_error = False
    for index, task in enumerate(selected, 1):
        stem = task["stem"]
        image_path: Path = task["image_path"]
        ocr: dict[str, Any] = task["ocr"]
        output_path = page_dir / f"{stem}.json"
        image_hash = sha256_file(image_path)
        fingerprint = input_fingerprint(
            image_hash,
            str(ocr.get("text", "")),
            task["candidates"],
            args.provider,
            model,
            args.detail,
        )
        reusable = None if args.overwrite else reusable_status(
            output_path, fingerprint, args.retry_review
        )
        if reusable:
            invocation[f"skipped_{reusable}"] += 1
            print(f"[{index}/{len(selected)}] SKIP {stem} ({reusable})")
            continue

        base = {
            "chapter_id": chapter_id,
            "stem": stem,
            "image_path": image_path.as_posix(),
            "image_sha256": image_hash,
            "ocr_path": task["ocr_path"].as_posix(),
            "ocr_run": args.ocr_run,
            "provider": args.provider,
            "model": model,
            "detail": args.detail,
            "prompt_version": PROMPT_VERSION,
            "input_fingerprint": fingerprint,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        if ocr.get("status") == "blank":
            write_json_atomic(output_path, {**base, "status": "blank", "correction": None})
            invocation["blank"] += 1
            print(f"[{index}/{len(selected)}] BLANK {stem}")
            continue

        started = time.monotonic()
        try:
            parsed, raw_response = call_llm(
                args,
                api_key,
                model,
                image_path,
                make_user_text(
                    stem,
                    str(ocr.get("text", "")),
                    task["candidates"],
                    args.ocr_guidance,
                ),
            )
            issues, metrics = validate_correction(
                parsed,
                str(ocr.get("text", "")),
                args,
                red_candidate_count=len(task["candidates"]),
            )
            status = "review" if issues else "accepted"
            result = {
                **base,
                "status": status,
                "issues": issues,
                "metrics": metrics,
                "correction": parsed,
                "response_id": raw_response.get("id"),
                "usage": (
                    raw_response.get("usage", {})
                    if args.provider in {"openai", "compatible"}
                    else {
                        key: raw_response.get(key)
                        for key in (
                            "prompt_eval_count", "eval_count", "total_duration",
                            "load_duration", "prompt_eval_duration", "eval_duration",
                        )
                    }
                ),
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
            invocation[status] += 1
            print(
                f"[{index}/{len(selected)}] {status.upper():8s} {stem} "
                f"similarity={metrics['similarity']} segments={metrics['included_segment_count']}"
            )
        except Exception as exc:
            result = {
                **base,
                "status": "error",
                "error": str(exc),
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
            invocation["error"] += 1
            print(f"[{index}/{len(selected)}] LỖI {stem}: {exc}", file=sys.stderr)
            fatal_api_error = isinstance(exc, FatalAPIError)
        write_json_atomic(output_path, result)
        if fatal_api_error:
            print("DỪNG: lỗi quota/model/key không thể khắc phục bằng retry.", file=sys.stderr)
            break
        if args.delay and index < len(selected):
            time.sleep(args.delay)

    final_counts = Counter()
    page_results: list[tuple[str, dict[str, Any]]] = []
    for task in selected:
        path = page_dir / f"{task['stem']}.json"
        if not path.is_file():
            final_counts["missing"] += 1
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            final_counts["error"] += 1
            continue
        status = str(data.get("status", "error"))
        final_counts[status] += 1
        page_results.append((task["stem"], data))

    item_by_stem = {str(task["stem"]): str(task["item_id"]) for task in selected}
    candidate_segments: dict[str, list[dict[str, str]]] = {}
    for stem, data in page_results:
        if data.get("status") not in {"accepted", "review"}:
            continue
        item_id = item_by_stem[stem]
        candidate_segments.setdefault(item_id, []).extend(correction_segments(data))

    candidate_dir = correction_root / "candidates"
    candidate_paths: list[str] = []
    candidate_line_count = 0
    candidate_lines_by_item: dict[str, list[str]] = {}
    for item_id, segments in candidate_segments.items():
        lines = [
            f"{item_id}_{index:06d}\t{str(segment['text']).strip()}"
            for index, segment in enumerate(segments, 1)
        ]
        candidate_lines_by_item[item_id] = lines
        candidate_path = candidate_dir / f"{item_id}_seg_candidate.tsv"
        write_text_atomic(candidate_path, "\n".join(lines) + ("\n" if lines else ""))
        candidate_paths.append(candidate_path.as_posix())
        candidate_line_count += len(lines)

    complete = len(selected) == len(tasks) and sum(final_counts.values()) == len(tasks)
    publishable = complete and not any(
        final_counts.get(status, 0) for status in ("review", "error", "missing")
    )
    published_paths: list[str] = []
    if args.publish:
        if not publishable:
            print(
                "LỖI: Không publish vì chưa đủ trang hoặc còn review/error/missing.",
                file=sys.stderr,
            )
        elif not candidate_line_count:
            print("LỖI: Không có câu để publish.", file=sys.stderr)
        else:
            expected_items = {str(task["item_id"]) for task in tasks}
            missing_items = sorted(expected_items - set(candidate_lines_by_item))
            if missing_items:
                print(
                    f"LỖI: Không publish vì {len(missing_items)} ảnh không có segment.",
                    file=sys.stderr,
                )
            else:
                for item_id in sorted(expected_items):
                    seg_path = (
                        Path(args.output_root)
                        / output_group_id
                        / item_id
                        / f"{item_id}_seg.tsv"
                    )
                    write_text_atomic(seg_path, "\n".join(candidate_lines_by_item[item_id]) + "\n")
                    published_paths.append(seg_path.as_posix())
                print(f"Đã publish {len(published_paths)} file seg trong {Path(args.output_root) / output_group_id}")

    summary = {
        "chapter_id": chapter_id,
        "legacy_folder": catalog["legacy_folder"],
        "submission_group": output_group_id,
        "submission_items": sorted({str(task["item_id"]) for task in tasks}),
        "ocr_run": args.ocr_run,
        "llm_run": args.llm_run,
        "provider": args.provider,
        "model": model,
        "detail": args.detail,
        "prompt_version": PROMPT_VERSION,
        "selected_pages": len(selected),
        "expected_pages": len(tasks),
        "final_counts": dict(final_counts),
        "invocation_counts": dict(invocation),
        "candidate_sentence_count": candidate_line_count,
        "candidate_paths": candidate_paths,
        "complete": complete,
        "publishable": publishable,
        "published_paths": published_paths,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json_atomic(correction_root / "run_summary.json", summary)
    print(f"Candidate: {len(candidate_paths)} file, {candidate_line_count} câu")
    print(f"Tổng kết: {correction_root / 'run_summary.json'}")
    if args.publish and not published_paths:
        return 1
    return 1 if final_counts.get("error") or final_counts.get("missing") else 0


if __name__ == "__main__":
    raise SystemExit(main())
