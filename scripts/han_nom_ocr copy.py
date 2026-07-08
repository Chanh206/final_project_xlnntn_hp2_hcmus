"""
Han Nom OCR Pipeline
====================
Đọc ảnh từ Google Drive → Gemini Vision OCR → Ghi vào Google Sheets
Hỗ trợ resume: tự động bỏ qua ảnh đã xử lý

Yêu cầu:
  pip3 install google-auth google-auth-oauthlib google-auth-httplib2
               google-api-python-client gspread google-genai

Cách dùng:
  1. Chạy lần đầu để setup OAuth: python3 han_nom_ocr.py --setup
  2. Chạy OCR:                     python3 han_nom_ocr.py
"""

import io
import sys
import time
import json
import re
import argparse
import logging
from pathlib import Path

from google import genai
from google.genai import types
import gspread
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# ─────────────────────────────────────────────
#  CẤU HÌNH — chỉnh sửa phần này
# ─────────────────────────────────────────────

GEMINI_API_KEY   = ""        # Lấy từ https://aistudio.google.com/app/apikey
DRIVE_FOLDER_ID  = "1" # ID folder chứa ảnh trên Drive
SHEET_ID         = ""        # ID Google Sheet (từ URL)
SHEET_TAB_NAME   = "manage"                      # Tên tab trong Sheet

# Tên file credentials OAuth2 tải về từ Google Cloud Console
OAUTH_CLIENT_FILE = "client_secret.json"
TOKEN_FILE        = "token.json"

# Free tier: 15 RPM → sleep 4s mỗi request
SLEEP_BETWEEN_REQUESTS = 4
MAX_RETRIES            = 5

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("ocr_progress.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
#  SCOPES
# ─────────────────────────────────────────────

SCOPES = [
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/spreadsheets",
]

# ─────────────────────────────────────────────
#  AUTHENTICATE GOOGLE (OAuth2)
# ─────────────────────────────────────────────

def get_google_credentials():
    """Lấy hoặc refresh OAuth2 credentials. Lần đầu sẽ mở browser."""
    creds = None

    if Path(TOKEN_FILE).exists():
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            log.info("Refreshing access token...")
            creds.refresh(Request())
        else:
            if not Path(OAUTH_CLIENT_FILE).exists():
                log.error(f"Không tìm thấy file '{OAUTH_CLIENT_FILE}'")
                log.error("Tải file từ Google Cloud Console → APIs & Services → Credentials")
                log.error("→ Create Credentials → OAuth client ID → Desktop app → Download JSON")
                log.error(f"→ Đổi tên thành '{OAUTH_CLIENT_FILE}' và đặt cùng thư mục với script này")
                sys.exit(1)
            log.info("Mở browser để đăng nhập Google...")
            flow = InstalledAppFlow.from_client_secrets_file(OAUTH_CLIENT_FILE, SCOPES)
            creds = flow.run_local_server(port=0)

        with open(TOKEN_FILE, "w") as f:
            f.write(creds.to_json())
        log.info(f"Token đã lưu vào '{TOKEN_FILE}'")

    return creds

# ─────────────────────────────────────────────
#  GOOGLE DRIVE
# ─────────────────────────────────────────────

def list_subfolders(drive_service, folder_id):
    """Lấy danh sách subfolder trực tiếp trong folder_id."""
    subfolders = []
    page_token = None
    query = (
        f"'{folder_id}' in parents "
        f"and mimeType = 'application/vnd.google-apps.folder' "
        f"and trashed = false"
    )
    while True:
        kwargs = dict(
            q=query,
            fields="nextPageToken, files(id, name)",
            pageSize=1000,
        )
        if page_token:
            kwargs["pageToken"] = page_token
        response = drive_service.files().list(**kwargs).execute()
        subfolders.extend(response.get("files", []))
        page_token = response.get("nextPageToken")
        if not page_token:
            break
    return subfolders


def list_images_in_folder(drive_service, folder_id):
    """Liệt kê ảnh đệ quy qua tất cả subfolder.

    Hỗ trợ cấu trúc:
      nlp_project/
        input/
          hvh_001/ -> hvh_001_0001.jpg
          hvh_002/ -> ...
    """
    images = []

    def _collect(fid, path=""):
        # Lấy ảnh trong folder hiện tại
        page_token = None
        query = (
            f"'{fid}' in parents "
            f"and mimeType contains 'image/' "
            f"and trashed = false"
        )
        while True:
            kwargs = dict(
                q=query,
                fields="nextPageToken, files(id, name, mimeType, webViewLink)",
                pageSize=1000,
            )
            if page_token:
                kwargs["pageToken"] = page_token
            response = drive_service.files().list(**kwargs).execute()
            batch = response.get("files", [])
            for f in batch:
                f["folder_path"] = path
            images.extend(batch)
            page_token = response.get("nextPageToken")
            if not page_token:
                break

        # Đệ quy vào subfolder
        for sub in list_subfolders(drive_service, fid):
            sub_path = f"{path}/{sub['name']}" if path else sub["name"]
            log.info(f"  Quét subfolder: {sub_path}")
            _collect(sub["id"], sub_path)

    _collect(folder_id)
    images.sort(key=lambda f: (f.get("folder_path", ""), f["name"]))
    log.info(f"  Tổng cộng: {len(images)} ảnh.")
    return images


def download_image_bytes(drive_service, file_id):
    """Download ảnh về memory (bytes), không lưu ra disk."""
    request = drive_service.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return buf.getvalue()

# ─────────────────────────────────────────────
#  GEMINI OCR (dùng google-genai mới)
# ─────────────────────────────────────────────

PROMPT = """Đây là ảnh chứa chữ Hán Nôm (chữ Nôm Việt cổ hoặc chữ Hán).
Hãy nhận diện các ký tự và đánh giá độ tin cậy của kết quả.
Trả về JSON theo đúng format sau, không thêm gì khác, không có markdown:
{
  "characters": "các ký tự nhận diện được, mỗi dòng ảnh cách nhau bằng \n",
  "confidence": 85,
  "note": "lý do nếu độ tin cậy thấp, để trống nếu rõ ràng"
}
Confidence từ 0-100: 90+ rõ ràng, 70-89 khá chắc, 50-69 không chắc, dưới 50 rất mờ/khó đọc."""


def detect_characters(client, image_bytes, mime_type="image/jpeg"):
    """Gửi ảnh vào Gemini, trả về dict {characters, confidence, note}."""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            PROMPT,
        ],
    )
    raw = response.text.strip()

    # Bóc JSON ra dù Gemini có wrap markdown hay không
    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            return {
                "characters": str(data.get("characters", "")).strip(),
                "confidence": int(data.get("confidence", 0)),
                "note":       str(data.get("note", "")).strip(),
            }
        except (json.JSONDecodeError, ValueError):
            pass

    # Fallback: Gemini không trả JSON đúng format
    return {"characters": raw, "confidence": 0, "note": "parse JSON thất bại"}


def error_result(msg):
    return {"characters": "", "confidence": 0, "note": msg, "error": msg}


def detect_with_retry(client, image_bytes, mime_type, file_name):
    """Gọi Gemini với exponential backoff khi gặp rate limit."""
    wait = 10
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return detect_characters(client, image_bytes, mime_type)
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "resource_exhausted" in err_str:
                if attempt < MAX_RETRIES:
                    log.warning(f"Rate limit ({file_name}), chờ {wait}s... ({attempt}/{MAX_RETRIES})")
                    time.sleep(wait)
                    wait = min(wait * 2, 120)
                else:
                    log.error(f"Vượt quá số lần retry cho '{file_name}'.")
                    return error_result("ERROR: RATE_LIMIT")
            else:
                log.error(f"Lỗi Gemini cho '{file_name}': {e}")
                return error_result(f"ERROR: {e}")
    return error_result("ERROR: MAX_RETRIES")

# ─────────────────────────────────────────────
#  GOOGLE SHEETS
# ─────────────────────────────────────────────

def get_processed_files(worksheet):
    """Đọc cột A để biết file nào đã xử lý (resume logic)."""
    values = worksheet.col_values(1)
    if values and values[0].lower() in ("file_name", "tên file", "filename"):
        values = values[1:]
    # Lấy tên file thuần từ công thức HYPERLINK nếu có
    names = set()
    for v in values:
        m = re.search(r'HYPERLINK\("[^"]+","([^"]+)"\)', v)
        names.add(m.group(1) if m else v)
    return names


def ensure_header(worksheet):
    """Thêm header nếu sheet còn trống."""
    if not worksheet.row_values(1):
        worksheet.append_row(
            ["file_name", "characters", "confidence", "note", "status", "timestamp"],
            value_input_option="RAW",
        )
        log.info("Đã thêm header vào Sheet.")


def write_result(worksheet, file_name, drive_url, result_dict, status="success"):
    """Ghi 1 dòng kết quả vào Sheet."""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if drive_url:
        safe_name  = file_name.replace('"', '""')
        cell_value = f'=HYPERLINK("{drive_url}";"{safe_name}")'
        input_opt  = "USER_ENTERED"
    else:
        cell_value = file_name
        input_opt  = "RAW"

    worksheet.append_row(
        [
            cell_value,
            result_dict.get("characters", ""),
            f"{result_dict.get('confidence', 0)}%",
            result_dict.get("note", ""),
            status,
            timestamp,
        ],
        value_input_option=input_opt,
    )

# ─────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────

def run_ocr():
    if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
        log.error("Chưa điền GEMINI_API_KEY trong script!")
        sys.exit(1)

    # 1. Khởi tạo Gemini client mới
    client = genai.Client(api_key=GEMINI_API_KEY)
    log.info("Khởi tạo Gemini client thành công.")

    # 2. Google OAuth
    creds = get_google_credentials()
    drive_service = build("drive", "v3", credentials=creds)
    gc = gspread.authorize(creds)
    log.info("Kết nối Google Drive & Sheets thành công.")

    # 3. Mở Sheet
    worksheet = gc.open_by_key(SHEET_ID).worksheet(SHEET_TAB_NAME)
    ensure_header(worksheet)

    # 4. Resume logic
    processed = get_processed_files(worksheet)
    log.info(f"Sheet đã có {len(processed)} file được xử lý trước đó.")

    # 5. Liệt kê ảnh
    log.info(f"Đang liệt kê ảnh trong folder: {DRIVE_FOLDER_ID}")
    all_images = list_images_in_folder(drive_service, DRIVE_FOLDER_ID)
    log.info(f"Tổng số ảnh: {len(all_images)}")

    pending = [f for f in all_images if f["name"] not in processed]
    log.info(f"Số ảnh cần xử lý: {len(pending)}")

    if not pending:
        log.info("Tất cả ảnh đã được xử lý.")
        return

    # 6. Vòng lặp chính
    success_count = error_count = 0

    for idx, file in enumerate(pending, start=1):
        file_name = file["name"]
        file_id   = file["id"]
        mime_type = file.get("mimeType", "image/jpeg")
        drive_url = file.get("webViewLink", "")

        log.info(f"[{idx}/{len(pending)}] {file_name}")

        try:
            image_bytes = download_image_bytes(drive_service, file_id)
        except Exception as e:
            log.error(f"  Lỗi download: {e}")
            write_result(worksheet, file_name, drive_url,
                         {"characters": "", "confidence": 0, "note": str(e)},
                         status="ERROR_DOWNLOAD")
            error_count += 1
            continue

        result = detect_with_retry(client, image_bytes, mime_type, file_name)

        if result.get("error"):
            write_result(worksheet, file_name, drive_url, result, status=result["error"])
            error_count += 1
        else:
            write_result(worksheet, file_name, drive_url, result, status="success")
            success_count += 1
            preview = result["characters"][:60]
            log.info(f"  ✓ [{result['confidence']}%] {preview}{'...' if len(result['characters']) > 60 else ''}")

        if idx < len(pending):
            time.sleep(SLEEP_BETWEEN_REQUESTS)

    log.info("=" * 50)
    log.info(f"Hoàn thành! Thành công: {success_count} | Lỗi: {error_count}")


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Han Nom OCR Pipeline")
    parser.add_argument("--setup", action="store_true",
                        help="Chỉ đăng nhập OAuth, không chạy OCR")
    args = parser.parse_args()

    if args.setup:
        log.info("Chế độ setup: đăng nhập Google OAuth...")
        get_google_credentials()
        log.info("Setup hoàn tất! Chạy lại không có --setup để bắt đầu OCR.")
    else:
        run_ocr()