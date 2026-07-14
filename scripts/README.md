# Thứ tự chạy pipeline

1. `1_download_nomfoundation_images.py`: tải ảnh theo catalog.
2. `2_migrate_input_structure.py`: migration một lần từ cấu trúc cũ.
3. `3_preprocess_images.py`: kiểm tra, cắt viền và tách trang.
4. `4_ocr_local.py`: OCR ảnh local, lưu JSON từng trang và resume.
5. `5_compare_ocr.py`: so sánh pilot OCR gốc/processed.
6. `6_build_outputs.py`: ghép `_raw.txt`, tách câu và tạo `_seg.tsv`.
7. `7_check_output.py`: kiểm tra hai output cuối.
8. `8_llm_correct_segments.py`: vision LLM đối chiếu ảnh, hiệu chỉnh OCR và tách câu có guard/resume.

## OCR pilot bằng PaddleOCR

Kích hoạt Conda env có PaddleOCR:

```bash
conda activate NLP
```

Kiểm tra mapping mà không gọi API:

```bash
python scripts/4_ocr_local.py \
  --id HVH_001 --chapter 01 --source both --limit 3 --dry-run
```

Chạy OCR thật trên ba scan gốc và sáu trang processed:

```bash
python scripts/4_ocr_local.py \
  --id HVH_001 --chapter 01 --engine paddle \
  --source both --limit 3 --run-name paddle_pilot_v1
```

Lệnh in ra `run_name`. Dùng giá trị đó để so sánh:

```bash
python scripts/5_compare_ocr.py \
  --id HVH_001 --chapter 01 --run-name paddle_pilot_v1
```

`comparison.tsv` có hai cột `manual_choice` và `manual_note` để người
kiểm tra quyết định. Chỉ số confidence do model tự báo không thay thế
ground truth hoặc đối chiếu bằng mắt.

PaddleOCR local là engine mặc định. `.env` chỉ cần khi chạy
`--engine api`; xem `.env.example`.

## OCR nguyên trang và OCR từng cột

Bước 4 hỗ trợ `--ocr-layout full-page` (mặc định) và `--ocr-layout columns`.
Chế độ columns lưu crop cột phải sang trái trong thư mục run. Chạy hai run
pilot riêng rồi đối chiếu bằng bước 5 với `--full-page-run` và `--columns-run`.
Hướng dẫn đầy đủ xem `RUN_FROM_SCRATCH.md`.

## Tạo output

Sau khi OCR processed đầy đủ, ghép raw và tách câu. Mặc định script 6
dùng dấu ngắt màu đỏ trên ảnh và polygon OCR để phục hồi ranh giới:

```bash
python scripts/6_build_outputs.py \
  --id HVH_001 --chapter 01 --run-name paddle_full_v1 --overwrite

python scripts/7_check_output.py --id HVH_001_01
```

Script tạo thêm `build_reports/*.segmentation_review.tsv` và
`*.red_mark_detections.tsv` để duyệt nguồn trang, phương pháp ngắt và
tọa độ dấu đỏ. Các dòng `page_tail`/`layout_unit` cần được chú ý khi
hiệu đính. Có thể dùng `--segmentation-mode columns` để tái tạo fallback
cũ, hoặc `--segmentation-mode ocr-punctuation` để chỉ dùng dấu Unicode.

## Hiệu chỉnh bằng vision LLM

Không sửa trực tiếp `_raw.txt`. Trước tiên cấu hình `OPENAI_API_KEY` và
`LLM_CORRECTION_MODEL` trong `.env`, rồi kiểm tra ba trang mà chưa gọi API:

```bash
python scripts/8_llm_correct_segments.py \
  --id HVH_001 --chapter 01 --ocr-run paddle_full_v1 \
  --limit 3 --dry-run
```

Chạy pilot thật bằng cách bỏ `--dry-run`. Kết quả từng trang được lưu để
resume trong `data/intermediate/.../llm_corrections/`. Sau khi pilot ổn,
chạy đủ trang; chỉ thêm `--publish` khi muốn thay `_seg.tsv` chính thức.
Script sẽ từ chối publish nếu còn trang `review`, `error` hoặc `missing`.
Sau khi publish bản đã hiệu chỉnh, validator cần cho phép `_seg.tsv` khác
OCR thô:

```bash
python scripts/7_check_output.py --id HVH_001_01 --allow-corrected
```

Chạy local miễn phí bằng Ollama/Qwen3-VL:

```bash
python scripts/8_llm_correct_segments.py \
  --id HVH_001 --chapter 01 --ocr-run paddle_full_v1 \
  --provider ollama --model qwen3-vl:8b \
  --llm-run qwen3vl8b_pilot_v1 --limit 3
```

Chạy API trung gian tương thích OpenAI bằng các biến `OCR_*` trong `.env`:

```bash
python scripts/8_llm_correct_segments.py \
  --id HVH_001 --chapter 01 --ocr-run paddle_full_v1 \
  --provider compatible --llm-run compatible_pilot_v1 --limit 3
```
