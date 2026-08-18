"""
Duyệt qua facebook_posts_valid.jsonl để biết mỗi post có bao nhiêu ảnh,
tìm ảnh local tương ứng trong `images/{post_id}_{index}.jpg` (do fetch_images.py
tải về), gửi từng ảnh cho Gemini (qua proxy OpenAI-compatible) để OCR,
rồi ghi kết quả tăng dần (resumable) vào facebook_posts_ocr.jsonl.

Chạy: cd nlp_final_project && python3 ocr_images.py [--limit N]
"""
import os
import re
import json
import time
import base64
import argparse
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OCR_API_KEY = os.environ.get("OCR_API_KEY", "")
OCR_BASE_URL = os.environ.get("OCR_BASE_URL", "")
OCR_MODEL = os.environ.get("OCR_MODEL", "")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(SCRIPT_DIR, "facebook_posts_valid.jsonl")
IMAGES_DIR = os.path.join(SCRIPT_DIR, "images")
OUTPUT_JSONL = os.path.join(SCRIPT_DIR, "facebook_posts_ocr.jsonl")
OUTPUT_JSON = os.path.join(SCRIPT_DIR, "facebook_posts_ocr.json")

MAX_RETRIES = 3
RETRY_BACKOFF_SEC = 5
MAX_CONSECUTIVE_FAILURES = 5


class TooManyConsecutiveFailures(Exception):
    """Dừng hẳn khi nhiều ảnh liên tiếp OCR thất bại (nghi ngờ lỗi hệ thống: API/key/quota)."""
    pass

client = OpenAI(api_key=OCR_API_KEY, base_url=OCR_BASE_URL)


def strip_json_fence(text: str) -> str:
    """Model đôi khi bọc JSON trong markdown code fence (```json ... ```)."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def image_file_to_data_uri(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def ocr_local_image(local_path: str):
    """Trả về list kết quả OCR, hoặc None nếu thất bại hoàn toàn (để retry lần sau)."""
    prompt = (
        "Thực hiện OCR chữ viết/thư pháp trong ảnh này.\n"
        "Hãy trích xuất từng từ/dòng chữ và vị trí bounding box của nó.\n"
        "Tọa độ bounding box phải theo thứ tự: [ymin, xmin, ymax, xmax] "
        "trong khoảng từ 0 đến 1000 (chuẩn hóa theo kích thước ảnh).\n\n"
        "Trả về kết quả dưới dạng một JSON Array có định dạng chính xác như sau:\n"
        "[\n"
        "  {\n"
        '    "text": "văn bản trích xuất",\n'
        '    "bounding_box": [ymin, xmin, ymax, xmax]\n'
        "  }\n"
        "]"
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            image_data_uri = image_file_to_data_uri(local_path)
            response = client.chat.completions.create(
                model=OCR_MODEL,
                response_format={"type": "json_object"},
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_data_uri}}
                    ]
                }],
                temperature=0.1
            )
            content = response.choices[0].message.content
            parsed_data = json.loads(strip_json_fence(content))

            if isinstance(parsed_data, dict):
                for value in parsed_data.values():
                    if isinstance(value, list):
                        return value
                return []
            elif isinstance(parsed_data, list):
                return parsed_data
            return []
        except Exception as e:
            print(f"  ⚠️ OCR lần thử {attempt}/{MAX_RETRIES} thất bại: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SEC * attempt)

    return None


def load_done_image_keys(output_jsonl: str) -> set:
    done = set()
    if not os.path.exists(output_jsonl):
        return done
    with open(output_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                done.add(item["image"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def compile_json_array(output_jsonl: str, output_json: str):
    results = []
    if os.path.exists(output_jsonl):
        with open(output_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"📦 Đã gộp {len(results)} ảnh vào: {output_json}")


def main():
    parser = argparse.ArgumentParser(description="OCR ảnh local qua Gemini (proxy bên thứ 3).")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--images-dir", default=IMAGES_DIR)
    parser.add_argument("--output-jsonl", default=OUTPUT_JSONL)
    parser.add_argument("--output-json", default=OUTPUT_JSON)
    parser.add_argument("--limit", type=int, default=None, help="Chỉ OCR tối đa N ảnh mới (để test)")
    args = parser.parse_args()

    if not (OCR_API_KEY and OCR_BASE_URL and OCR_MODEL):
        print("❌ Thiếu biến môi trường OCR_API_KEY / OCR_BASE_URL / OCR_MODEL (kiểm tra file .env).")
        return

    with open(args.input, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    posts = []
    for line in lines:
        try:
            posts.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    done_keys = load_done_image_keys(args.output_jsonl)
    if done_keys:
        print(f"🔁 Resume: đã có {len(done_keys)} ảnh trong '{args.output_jsonl}'.")

    out_f = open(args.output_jsonl, "a", encoding="utf-8")
    ocr_count = 0
    missing_count = 0
    consecutive_failures = 0
    stopped_reason = None

    try:
        for line_idx, post in enumerate(posts, 1):
            post_id = post.get("id", "")
            label = post.get("label", "")
            n_expected = len(post.get("images", []))

            for img_idx in range(n_expected):
                if args.limit is not None and ocr_count >= args.limit:
                    print(f"\n🛑 Đã đạt giới hạn --limit={args.limit}, dừng lại.")
                    raise StopIteration

                image_name = f"/images/{post_id}_{img_idx}.jpg"
                if image_name in done_keys:
                    continue

                local_path = os.path.join(args.images_dir, f"{post_id}_{img_idx}.jpg")
                if not os.path.exists(local_path):
                    missing_count += 1
                    continue  # ảnh chưa được fetch_images.py tải về

                print(f"[Dòng {line_idx}/{len(posts)}] OCR {local_path}...")
                ocr_results = ocr_local_image(local_path)

                if ocr_results is None:
                    consecutive_failures += 1
                    print(f"  ❌ OCR thất bại hoàn toàn ({consecutive_failures}/{MAX_CONSECUTIVE_FAILURES} "
                          f"lỗi liên tiếp), sẽ thử lại ở lần chạy sau.")
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        raise TooManyConsecutiveFailures(
                            f"{MAX_CONSECUTIVE_FAILURES} ảnh liên tiếp OCR thất bại — "
                            f"có thể API/key/quota đang gặp sự cố."
                        )
                    continue

                consecutive_failures = 0
                item = {
                    "id": post_id,
                    "image": image_name,
                    "ground_truth": label,
                    "label": label,
                    "gemini": ocr_results
                }
                out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
                out_f.flush()
                done_keys.add(image_name)
                ocr_count += 1
    except StopIteration:
        pass
    except TooManyConsecutiveFailures as e:
        stopped_reason = str(e)
        print(f"\n🛑 Dừng hẳn: {stopped_reason}")
        print("   Chạy lại script sau khi kiểm tra API/key — các ảnh đã OCR xong sẽ không bị chạy lại.")
    finally:
        out_f.close()

    compile_json_array(args.output_jsonl, args.output_json)
    print(f"\n✅ Hoàn tất phiên chạy này. Đã OCR {ocr_count} ảnh mới. ({missing_count} ảnh chưa có local, "
          f"hãy chạy fetch_images.py trước.)")
    if stopped_reason:
        print(f"⚠️ Dừng sớm do lỗi liên tiếp: {stopped_reason}")


if __name__ == "__main__":
    main()
