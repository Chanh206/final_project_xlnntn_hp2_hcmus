# Chạy lại pipeline từng bước

Các lệnh dưới đây chạy tại thư mục gốc project:

```bash
cd xlnntn_hp2_k35
conda activate NLP
```

## 0. Làm sạch kết quả sinh ra một cách an toàn

Không xóa `data/input`, `configs/corpus_catalog.csv` hoặc `data/input/link.txt`.
Nên đổi tên kết quả cũ để có thể phục hồi:

```bash
stamp=$(date +%Y%m%d_%H%M%S)
mv data/processed "data/processed_backup_${stamp}" 2>/dev/null || true
mv data/intermediate "data/intermediate_backup_${stamp}" 2>/dev/null || true
mv data/output "data/output_backup_${stamp}" 2>/dev/null || true
```

Nếu chắc chắn không cần backup, có thể tự xóa ba thư mục `data/processed`,
`data/intermediate`, `data/output`; tuyệt đối không xóa `data/input`.

## 1. Kiểm tra/tải ảnh

Ảnh đã có đủ thì không cần tải lại. Muốn script tự bỏ qua file đã tồn tại:

```bash
python scripts/1_download_nomfoundation_images.py \
  --links-file data/input/link.txt --out-root data/input
```

## 2. Kiểm tra cấu trúc input

```bash
python scripts/2_migrate_input_structure.py
```

Nếu script báo dữ liệu đã ở cấu trúc mới thì đó là kết quả đúng. Chỉ dùng
`--execute` khi dữ liệu thực sự còn ở cấu trúc phẳng cũ.

## 3. Tách trang và kiểm tra ảnh

Chạy thử một quyển:

```bash
python scripts/3_preprocess_images.py \
  --id HVH_001 --chapter 01 --variant color
```

Kiểm tra `data/processed/HVH_001/HVH_001_01/summary.json` và `manifest.csv`.

## 4A. Pilot OCR nguyên trang

```bash
python scripts/4_ocr_local.py \
  --id HVH_001 --chapter 01 --engine paddle --source processed \
  --limit 3 --ocr-layout full-page --run-name paddle_fullpage_pilot
```

## 4B. Pilot OCR từng cột

```bash
python scripts/4_ocr_local.py \
  --id HVH_001 --chapter 01 --engine paddle --source processed \
  --limit 3 --ocr-layout columns --run-name paddle_columns_pilot
```

Crop cột nằm tại:

```text
data/intermediate/HVH_001/HVH_001_01/ocr_runs/
└── paddle_columns_pilot/column_crops/processed/<PAGE_ID>/col_NN.png
```

## 5. So sánh full-page với columns

```bash
python scripts/5_compare_ocr.py \
  --id HVH_001 --chapter 01 \
  --full-page-run paddle_fullpage_pilot \
  --columns-run paddle_columns_pilot
```

Đọc `summary.json` và `comparison.tsv` trong:

```text
data/intermediate/HVH_001/HVH_001_01/ocr_comparison/
paddle_fullpage_pilot_vs_paddle_columns_pilot/
```

Không chọn layout chỉ dựa vào confidence. Cần xem thêm số ký tự CJK, số cột,
unknown và nội dung. Pilot v2 hiện tại trên sáu trang khuyến nghị `columns`:
confidence 67.275% so với 66.213%, và 1023 so với 1005 ký tự CJK. Đây vẫn là
heuristic chứ không phải accuracy.

## 6. OCR đầy đủ bằng layout đã chọn

Với kết quả pilot v2 hiện tại, thử chạy đầy đủ bằng từng cột:

```bash
python scripts/4_ocr_local.py \
  --id HVH_001 --chapter 01 --engine paddle --source processed \
  --ocr-layout columns --run-name paddle_columns_full
```

Nếu muốn giữ một run nguyên trang đầy đủ để đối chứng, chạy riêng:

```bash
python scripts/4_ocr_local.py \
  --id HVH_001 --chapter 01 --engine paddle --source processed \
  --ocr-layout full-page --run-name paddle_full_v2
```

## 7. Tạo hai output bắt buộc và kiểm tra

Tên `--run-name` phải trùng run đầy đủ đã chọn ở bước trên:

```bash
python scripts/6_build_outputs.py \
  --id HVH_001 --chapter 01 --run-name paddle_columns_full --overwrite

python scripts/7_check_output.py --id HVH_001_01
```

Output:

```text
data/output/HVH_001/HVH_001_01/
├── HVH_001_01_raw.txt
└── HVH_001_01_seg.tsv
```

## 8. Pilot LLM correction (không publish)

Model hiện tại chưa đủ tốt, nên chỉ pilot ba trang và không thêm `--publish`:

```bash
python scripts/8_llm_correct_segments.py \
  --id HVH_001 --chapter 01 --ocr-run paddle_columns_full \
  --provider compatible --llm-run compatible_pilot_v3 \
  --ocr-guidance full --limit 3
```

Chỉ chạy toàn bộ/publish khi pilot không còn `review` và đã có cách đánh giá
accuracy bằng ground truth. Bước 8 không sửa `_raw.txt`.
