# Chạy lại pipeline từ đầu đến cuối

Quy trình này tạo output cho 32 folder nguồn và 1.994 ảnh. Mỗi ảnh gốc tạo
một folder chứa đúng hai file. Ví dụ `HVH_001` có 48 ảnh nên output chạy từ
`HVH_001_0001` đến `HVH_001_0048`.

Chạy mọi lệnh tại thư mục gốc repository:

```bash
cd xlnntn_hp2_k35
conda activate NLP
```

Không chạy `scripts/2_migrate_input_structure.py`. Script đó chỉ dành cho
dữ liệu legacy từng bị gom sai cấu trúc.

## 0. Kiểm tra môi trường

```bash
python --version
python -c "import cv2, numpy, paddle; print('OpenCV', cv2.__version__); print('NumPy', numpy.__version__); print('Paddle', paddle.__version__)"
python -c "from paddleocr import PaddleOCR; print('PaddleOCR import OK')"
```

Model Paddle đã tải nằm trong `models/paddlex`; giữ thư mục này để không phải
tải lại. Nên còn trống ít nhất 8 GB trước khi chạy toàn bộ.

## 1. Làm sạch kết quả cũ

Không xóa `data/input`, `data/input/link.txt`, `.env`, `configs` hoặc
`models`. Nếu chắc chắn kết quả cũ không cần giữ:

```bash
rm -rf data/processed data/intermediate data/output final_output
find scripts -type d -name '__pycache__' -prune -exec rm -rf {} +
```

Trong workspace hiện tại, bước này đã được thực hiện.

## 2. Kiểm tra input

Không cần tải lại vì hiện đã có đủ ảnh. Chạy kiểm tra:

```bash
python scripts/1_download_nomfoundation_images.py \
  --links-file data/input/link.txt \
  --catalog configs/corpus_catalog.csv \
  --out-root data/input \
  --validate-only

find data/input -mindepth 1 -maxdepth 1 -type d -name 'HVH_*' | wc -l
find data/input -mindepth 2 -maxdepth 2 -type f -name '*.jpg' | wc -l
```

Kết quả bắt buộc:

```text
ĐẠT: 32 link khớp 32 folder HVH_001..HVH_032
32
1994
```

Nếu thiếu ảnh, chạy lại bước tải có resume:

```bash
python scripts/1_download_nomfoundation_images.py \
  --links-file data/input/link.txt \
  --catalog configs/corpus_catalog.csv \
  --out-root data/input \
  --quality large --workers 6
```

## 3. Tiền xử lý toàn bộ 32 folder

Biến thể `color` giữ thông tin màu đỏ dùng để tách câu:

```bash
for n in $(seq -f '%03g' 1 32)
do
  echo "=== PREPROCESS HVH_${n} ==="
  if ! python scripts/3_preprocess_images.py \
    --folder "HVH_${n}" \
    --variant color
  then
    echo "DỪNG: PREPROCESS LỖI TẠI HVH_${n}"
    break
  fi
done
```

Kiểm tra phải có 32 manifest và không có ảnh preprocess lỗi:

```bash
find data/processed -name manifest.csv | wc -l
find data/processed -name summary.json -print0 | \
  xargs -0 jq -r 'select(.input_images_failed != 0) | [.chapter_id, .input_images_failed] | @tsv'
```

Lệnh thứ nhất phải in `32`; lệnh thứ hai không được in dòng nào. Các giá trị
`split_confidence` thấp chỉ phản ánh độ chắc chắn của vị trí tách scan, không
phải confidence OCR.

## 4. Pilot xác nhận layout trên HVH_001

Chạy ba trang bằng cả hai layout:

```bash
python scripts/4_ocr_local.py \
  --folder HVH_001 --engine paddle --source processed \
  --limit 3 --ocr-layout full-page \
  --run-name paddle_fullpage_pilot

python scripts/4_ocr_local.py \
  --folder HVH_001 --engine paddle --source processed \
  --limit 3 --ocr-layout columns \
  --run-name paddle_columns_pilot

python scripts/5_compare_ocr.py \
  --folder HVH_001 \
  --full-page-run paddle_fullpage_pilot \
  --columns-run paddle_columns_pilot
```

Thiết kế hiện tại chọn `columns`. Nếu tất cả crop cột OCR rỗng, script tự
fallback sang full-page trước khi kết luận `BLANK`.

## 5. OCR đầy đủ 32 folder

Tên run dùng thống nhất ở các bước sau là `paddle_columns_full`:

```bash
for n in $(seq -f '%03g' 1 32)
do
  echo "=== OCR HVH_${n} ==="
  if ! python scripts/4_ocr_local.py \
    --folder "HVH_${n}" \
    --engine paddle \
    --source processed \
    --ocr-layout columns \
    --run-name paddle_columns_full \
    --retry-errors
  then
    echo "DỪNG: OCR LỖI TẠI HVH_${n}"
    break
  fi
done
```

Lệnh có resume: chạy lại đúng vòng lặp nếu máy bị dừng. JSON `success` và
`blank` hợp lệ được bỏ qua; `error` được chạy lại nhờ `--retry-errors`.

Kiểm tra phải có 32 summary OCR và không có `error`/`missing`:

```bash
find data/intermediate -path '*/ocr_runs/paddle_columns_full/run_summary.json' | wc -l

find data/intermediate -path '*/ocr_runs/paddle_columns_full/run_summary.json' -print0 | \
  xargs -0 jq -r 'select(.stats.error != 0 or .stats.missing != 0) | [.chapter_id, .stats.error, .stats.missing] | @tsv'
```

Lệnh thứ nhất phải in `32`; lệnh thứ hai không được in dòng nào. Liệt kê các
run còn `BLANK`:

```bash
find data/intermediate -path '*/ocr_runs/paddle_columns_full/run_summary.json' -print0 | \
  xargs -0 jq -r 'select(.stats.blank != 0) | [.chapter_id, .stats.blank] | @tsv'
```

Không tự coi ảnh input là trắng chỉ vì OCR báo `BLANK`. Bước 6 sẽ từ chối
folder có tỷ lệ trang blank vượt 10%, đồng thời từ chối bất kỳ ảnh gốc nào
không tạo được raw/segment.

## 6. Tạo raw/seg theo từng ảnh

```bash
for n in $(seq -f '%03g' 1 32)
do
  echo "=== BUILD HVH_${n} ==="
  if ! python scripts/6_build_outputs.py \
    --folder "HVH_${n}" \
    --run-name paddle_columns_full \
    --overwrite
  then
    echo "DỪNG: BUILD LỖI TẠI HVH_${n}"
    break
  fi
done
```

Ví dụ cấu trúc sau khi build `HVH_001`:

```text
final_output/HVH_001/
├── HVH_001_0001/
│   ├── HVH_001_0001_raw.txt
│   └── HVH_001_0001_seg.tsv
├── ...
└── HVH_001_0048/
    ├── HVH_001_0048_raw.txt
    └── HVH_001_0048_seg.tsv
```

Nếu một scan được tách thành `p01/p02`, hai kết quả OCR được ghép trong cùng
cặp file của ảnh scan gốc.

## 7. Validate từng folder và toàn bộ corpus

Kiểm tra lần lượt; vòng lặp dừng ngay tại folder không đạt:

```bash
for n in $(seq -f '%03g' 1 32)
do
  echo "=== VALIDATE HVH_${n} ==="
  if ! python scripts/7_check_output.py --folder "HVH_${n}"
  then
    echo "DỪNG: VALIDATOR LỖI TẠI HVH_${n}"
    break
  fi
done
```

Cuối cùng:

```bash
python scripts/7_check_output.py --all
```

Chỉ hoàn tất khi nhận được:

```text
ĐẠT TOÀN BỘ: 32 folder, 1994 ảnh trong final_output
```

Kiểm tra nhanh tổng số file:

```bash
find final_output -mindepth 2 -maxdepth 2 -type d | wc -l
find final_output -type f -name '*_raw.txt' | wc -l
find final_output -type f -name '*_seg.tsv' | wc -l
```

Cả ba lệnh phải lần lượt in `1994`, `1994`, `1994`.

## 8. LLM correction tùy chọn

Bước này không bắt buộc để tạo hai output. Luôn pilot trước và không dùng
`--publish`:

```bash
python scripts/8_llm_correct_segments.py \
  --folder HVH_001 \
  --ocr-run paddle_columns_full \
  --provider ollama \
  --model qwen3-vl:32b \
  --llm-run qwen3vl32b_pilot \
  --limit 3
```

Chỉ chạy toàn bộ và thêm `--publish` khi pilot đạt yêu cầu, không còn trang
`review/error/missing`. Bước 8 không sửa `_raw.txt`; nó chỉ thay từng
`_seg.tsv` tương ứng với ảnh scan. Sau khi publish, validate bằng
`--allow-corrected`:

```bash
python scripts/7_check_output.py --folder HVH_001 --allow-corrected
```
