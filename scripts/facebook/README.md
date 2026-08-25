# Pipeline tạo ground truth ảnh Facebook

Pipeline này tạo ground truth đề xuất cho ảnh Facebook từ bốn nguồn bằng một
flow có kiểm soát:

```text
valid.jsonl
  ├─ caption gốc
  └─ URL ảnh
       ↓ bước 1: fetch
Images/ (ảnh thật)
       ↓ bước 2: Gemini vision OCR
facebook_posts_ocr.jsonl
       ↓ bước 3: so sánh caption ↔ Gemini
       ├─ Gemini_same_Label/ (record + ảnh thật)
       └─ Gemini_diff_Label/ (record + ảnh thật)
              ↓ bước 4: PP-OCRv6 CPU
          paddle_v6/new_labels.jsonl
              ↓ bước 5: strict join + vision LLM adjudication
          DeepSeek_ground_truth/adjudications.jsonl
```

Chạy mọi lệnh từ thư mục gốc repository. Không chạy script bằng cách `cd` vào
`scripts/facebook` vì các đường dẫn mặc định được thiết kế từ project root.

## 1. Cấu trúc code

| File | Vai trò | Bắt buộc |
|---|---|---|
| `1_fetch_images.py` | Tải và kiểm tra ảnh trong `valid.jsonl`; resume và retry | Có, nếu chưa có ảnh |
| `2_ocr_gemini.py` | Entry point OCR ảnh bằng vision API tương thích OpenAI | Có cho flow Gemini |
| `3_compare_gemini_label.py` | Tính metric, chia `same/diff`, materialize ảnh thật | Có |
| `4_paddlev6_relabel_diff.py` | OCR lại nhóm diff bằng PP-OCRv6 CPU | Có cho nhóm diff |
| `5_llm_adjudicate_ground_truth.py` | Join chặt bốn nguồn và nhờ vision LLM chọn ground truth | Có để phân xử tự động |
| `lib/vision_ocr.py` | Engine API, resize ảnh, resume, retry và JSON validation | Module dùng chung, không sửa output bằng tay |
| `lib/paddle_v6_cpu.py` | Engine PP-OCRv6 CPU, original baseline và fallback `invert/CLAHE` | Module dùng chung |
| `experiments/compare_paddle_preprocessing.py` | Đối chứng preprocessing trên mẫu phân tầng | Không bắt buộc |

## 2. Cấu trúc dữ liệu

```text
data/
├── mrDuc_data/
│   ├── valid.jsonl
│   ├── Images/
│   ├── facebook_posts_ocr.jsonl
│   ├── fetch_errors.jsonl
│   └── ocr_errors.jsonl
└── output/
    ├── Gemini_same_Label/
    │   ├── Images/
    │   ├── records.jsonl
    │   ├── comparison.tsv
    │   └── summary.json
    ├── Gemini_diff_Label/
    │   ├── Images/
    │   ├── records.jsonl
    │   ├── comparison.tsv
    │   ├── summary.json
    │   └── paddle_v6/
    │       ├── ocr_results.jsonl
    │       ├── new_labels.jsonl
    │       └── summary.json
    ├── DeepSeek_ground_truth/
    └── Qwen38_ground_truth/
```

`Gemini_same_Label/Images` và `Gemini_diff_Label/Images` chứa file ảnh vật lý,
không phải symlink hoặc hardlink. Bước 3 tạo các bản sao này theo mặc định.

## 3. Ràng buộc dữ liệu và an toàn join

1. Khóa ảnh chuẩn luôn là `/images/{post_id}_{image_index}.jpg` sinh từ
   `valid.jsonl`; không join theo thứ tự dòng hoặc fuzzy filename.
2. `image` và `post_id` phải khớp tuyệt đối trong caption, Gemini, tập diff và
   PaddleV6. Thiếu, trùng hoặc khác khóa làm bước 5 dừng toàn bộ trước khi gọi API.
3. Caption và Gemini trong các file dẫn xuất phải giống tuyệt đối nguồn gốc.
4. Mỗi request bước 5 lưu `join_fingerprint` SHA-256 của khóa và ba văn bản.
   Resume chỉ hợp lệ khi fingerprint, provider, model và evidence mode không đổi.
5. `--no-image` không tạo ground truth đã xác minh bằng ảnh; mọi record ở chế độ
   này bị đánh dấu `needs_review=true`.
6. Không dùng confidence do Paddle/LLM tự báo như accuracy hoặc CER/WER. Muốn
   công bố CER/WER cần bản chép chuẩn do người đọc xác nhận.
7. Không dùng `--overwrite` khi chỉ muốn resume. Tùy chọn này xóa kết quả cũ của
   đúng bước/lần chạy.

## 4. Cấu hình môi trường

```bash
conda env create -f environment.yml
conda activate NLP
cp .env.example .env
```

Ví dụ cấu hình API dùng chung:

```dotenv
OCR_API_KEY="replace_me"
OCR_BASE_URL="https://provider.example/v1"
OCR_MODEL="gemini-vision-model"

DEEPSEEK_MODEL="deepseek-vision-model"
```

Nếu DeepSeek dùng provider/key riêng, đặt thêm `DEEPSEEK_API_KEY` và
`DEEPSEEK_BASE_URL`. Không commit `.env`.

Qwen chỉ dùng được ở bước 5 khi endpoint thực sự hỗ trợ image input. Provider
hiện đã thử chỉ cung cấp `qwen-3.8-max` text-only, nên không tương đương flow
vision và không được dùng làm ground truth.

## 5. Chạy từng bước

### Bước 1 — Fetch ảnh

Kiểm tra kế hoạch:

```bash
python scripts/facebook/1_fetch_images.py --dry-run
```

Chạy đầy đủ:

```bash
python scripts/facebook/1_fetch_images.py \
  --workers 24 \
  --retries 4 \
  --report-every 100
```

Tham số quan trọng:

- `--workers`: request tải đồng thời; đây là I/O, có thể lớn hơn số worker OCR.
- `--retries`: số lần thử lại lỗi mạng/5xx/429.
- `--verify-existing`: mở và kiểm tra ảnh đã có thay vì chỉ kiểm tra size.
- `--max-fail-rate`: dừng cấp task mới khi tỷ lệ lỗi quá cao.
- `--overwrite`: tải lại ảnh đã tồn tại; không dùng khi resume bình thường.

Output chính: `data/mrDuc_data/Images`; tên ảnh là
`{post_id}_{image_index}.jpg`.

### Bước 2 — Gemini vision OCR

Pilot 10 ảnh:

```bash
python scripts/facebook/2_ocr_gemini.py \
  --workers 4 \
  --limit 10 \
  --retries 4 \
  --timeout 240 \
  --report-every 1
```

Chạy tiếp toàn bộ bằng resume:

```bash
python scripts/facebook/2_ocr_gemini.py \
  --workers 32 \
  --retries 4 \
  --timeout 240 \
  --max-fail-rate 0.10 \
  --fail-rate-min-samples 50 \
  --report-every 50
```

`--workers` bị giới hạn thực tế bởi rate limit của provider, không phải CPU.
Tăng dần `4 → 8 → 16 → 32`; giảm nếu xuất hiện 429/5xx. Output JSONL được flush
từng record và có thể resume. `--max-side`, `--max-upload-mb` và
`--jpeg-quality` kiểm soát payload ảnh; không giảm quá mạnh làm mất nét chữ.

### Bước 3 — So sánh và chia nhóm

```bash
python scripts/facebook/3_compare_gemini_label.py --dry-run

python scripts/facebook/3_compare_gemini_label.py \
  --threshold 0.58 \
  --min-common-han 4
```

Mặc định bước này chép ảnh thật vào hai thư mục `Images/`. Chỉ dùng
`--no-copy-images` khi không cần dataset tự chứa.

Quy tắc phân nhóm:

- Chuỗi dài tối đa 3 ký tự chỉ là `same` khi khớp tuyệt đối.
- Một chuỗi chứa chuỗi còn lại: coverage ngắn ≥ 0,90 và đủ chữ Hán chung.
- Hoặc coverage một chiều ≥ 0,72, Character F1 ≥ 0,55 và đủ chữ Hán chung.
- Hoặc điểm kết hợp ≥ `--threshold` và đủ `--min-common-han`.
- Gemini rỗng luôn vào `diff`.

### Bước 4 — PP-OCRv6 relabel nhóm diff

Dry-run:

```bash
python scripts/facebook/4_paddlev6_relabel_diff.py --dry-run
```

Thiết lập CPU khuyến nghị đã chạy ổn định:

```bash
python scripts/facebook/4_paddlev6_relabel_diff.py \
  --workers 4 \
  --cpu-threads 13 \
  --score-threshold 0.30 \
  --fallback-confidence 0.65
```

- Tổng `workers × cpu-threads` nên gần số lõi vật lý, không phải logical CPU.
- Nhiều process Paddle có thể chậm hơn do tranh RAM/băng thông bộ nhớ.
- Pipeline luôn OCR `original` trước.
- Chỉ khi original rỗng/confidence thấp mới thử một biến thể thích nghi:
  `invert` cho nền tối/chữ sáng hoặc `CLAHE` cho chữ mờ.
- Không dùng adaptive threshold.
- Fallback chỉ được chọn nếu mạnh hơn original theo guard trong engine.

Output `new_labels.jsonl` chỉ là nhãn đề xuất, chưa phải ground truth.

### Bước 5 — Vision LLM adjudication

Kiểm tra strict join mà không gọi API:

```bash
python scripts/facebook/5_llm_adjudicate_ground_truth.py \
  --provider deepseek \
  --dry-run
```

Pilot 10 ảnh:

```bash
python scripts/facebook/5_llm_adjudicate_ground_truth.py \
  --provider deepseek \
  --workers 4 \
  --max-tokens 2000 \
  --retries 4 \
  --timeout 240 \
  --limit 10 \
  --report-every 1
```

Chạy tiếp bằng resume: bỏ `--limit`, giữ `workers=4` vì thử nghiệm 6–8 worker
không ổn định bằng 4 worker trên provider hiện tại.

```bash
python scripts/facebook/5_llm_adjudicate_ground_truth.py \
  --provider deepseek \
  --workers 4 \
  --max-tokens 8192 \
  --retries 4 \
  --timeout 240 \
  --max-fail-rate 0.10 \
  --fail-rate-min-samples 50 \
  --report-every 50
```

Script gửi ảnh cùng ba ứng viên `caption`, `gemini`, `paddle_v6`. Model trả
`selected_source`, bản chép trực quan, ground truth đề xuất, điểm từng nguồn,
confidence, quan hệ caption, reason codes và `needs_review`.

## 6. Metric đánh giá

| Metric | Ý nghĩa | Khoảng |
|---|---|---:|
| Sequence similarity | Mức giống theo thứ tự ký tự (SequenceMatcher) | 0–1 |
| Character F1 | F1 trên số lần xuất hiện ký tự, ít nhạy với thứ tự | 0–1 |
| Bigram Dice | Mức trùng các cặp hai ký tự liên tiếp | 0–1 |
| OCR coverage | Phần OCR được bao phủ bởi chuỗi khớp theo thứ tự | 0–1 |
| Label coverage | Phần caption được bao phủ bởi chuỗi khớp theo thứ tự | 0–1 |
| Common Han | Số chữ Hán chung có tính số lần xuất hiện | số nguyên |
| Similarity score | `0.30*Sequence + 0.25*CharF1 + 0.25*shortCoverage + 0.20*Bigram` | 0–1 |
| Paddle mean confidence | Trung bình score vùng OCR của model Paddle | 0–1 |
| LLM source score | Model tự chấm độ khớp từng ứng viên với ảnh | 0–1 |
| Source score margin | Điểm nguồn cao nhất trừ nguồn thứ hai | 0–1 |

Bước 5 ép `needs_review=true` nếu một trong các điều kiện sau đúng:

- Không gửi ảnh.
- Chọn `merged` hoặc `uncertain`.
- LLM confidence `< 0,85`.
- Điểm nguồn cao nhất `< 0,75`.
- Margin giữa hai nguồn cao nhất `< 0,08`.

Các metric này dùng để phân luồng và kiểm soát rủi ro, không phải accuracy.

## 7. Resume, lỗi và kiểm tra nhanh

- JSONL là nguồn resume; JSON là bản compile để đọc/chia sẻ.
- Error log được compact: ảnh thành công ở lần chạy sau sẽ được bỏ khỏi lỗi.
- HTTP 429/5xx có retry; `model_not_found` dừng ngay vì retry không giúp được.
- Nếu tỷ lệ lỗi vượt `--max-fail-rate` sau đủ mẫu, script dừng cấp task mới.
- Luôn đọc `summary.json` và kiểm tra `succeeded`, `failed`, `done`,
  `unresolved_errors`, `needs_review`, tốc độ và token usage.

Kiểm tra số dòng:

```bash
wc -l \
  data/mrDuc_data/facebook_posts_ocr.jsonl \
  data/output/Gemini_same_Label/records.jsonl \
  data/output/Gemini_diff_Label/records.jsonl \
  data/output/Gemini_diff_Label/paddle_v6/new_labels.jsonl \
  data/output/DeepSeek_ground_truth/adjudications.jsonl
```

Kiểm tra ảnh thật sau bước 3:

```bash
find data/output/Gemini_same_Label/Images -type f | wc -l
find data/output/Gemini_diff_Label/Images -type f | wc -l
find data/output/Gemini_same_Label/Images -type l | wc -l
find data/output/Gemini_diff_Label/Images -type l | wc -l
```

## 8. Git và dữ liệu lớn

Được commit: code trong `scripts/facebook`, README, `.env.example`,
`.gitignore`, requirements và config không chứa bí mật.

Không commit: `.env`, ảnh, `data/mrDuc_data`, `data/output`, `data_packages`,
model cache, archive và file tạm. Các tập dữ liệu tự chứa được chia sẻ qua
Google Drive bằng `data_packages/` và kiểm tra bằng `SHA256SUMS`.
