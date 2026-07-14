# Kiểm tra và tiền xử lý ảnh

Script `scripts/3_preprocess_images.py` không thay đổi ảnh gốc trong
`data/input`. Kết quả được ghi sang `data/processed` theo hai cấp
tác phẩm/quyển. Cặp `work_id`/`chapter_id` phải tồn tại trong
`configs/corpus_catalog.csv`.

## Cài đặt

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-preprocess.txt
```

## Chạy pilot

Nên xem thử 10 ảnh đầu tiên trước khi xử lý cả tác phẩm:

```bash
python scripts/3_preprocess_images.py --id HVH_001 --limit 10
```

Mặc định output thuộc quyển/tập `01`. Có thể chọn quyển khác bằng
`--chapter`, ví dụ `--chapter 02`.

Sau khi kiểm tra ảnh tách trang và `manifest.csv`, chạy toàn bộ:

```bash
python scripts/3_preprocess_images.py --id HVH_001 --overwrite
```

## Quy tắc tách trang

- `--split auto` (mặc định): dùng tỷ lệ khung hình và đường gáy.
- `--split always`: luôn tách ảnh tại đường gáy gần tâm.
- `--split never`: chỉ cắt viền, không tách trang.

Khi tách spread, `p01` là trang bên phải và `p02` là trang bên
trái. Sắp xếp tên file vì thế cũng là thứ tự OCR.

## Biến thể ảnh

- `--variant color` (mặc định): giữ màu, chỉ cắt viền/tách trang.
- `--variant gray`: ảnh xám.
- `--variant clahe`: ảnh xám tăng tương phản cục bộ.
- `--variant binary`: nhị phân adaptive; chỉ nên dùng để so sánh OCR.

Không nên ghi nhiều variant vào cùng một thư mục. Khi thử
nghiệm, hãy truyền `--processed-root` khác nhau, ví dụ:

```bash
python scripts/3_preprocess_images.py --id HVH_001 --chapter 01 --limit 10 \
  --variant clahe --processed-root data/processed_clahe
```

## Kết quả

```text
data/input/HVH_001/HVH_001_01/      # ảnh gốc, không bị sửa
data/processed/HVH_001/HVH_001_01/  # ảnh đã cắt/tách trang
├── HVH_001_01_0001_p01.jpg
├── HVH_001_01_0001_p02.jpg
├── manifest.csv
└── summary.json
data/intermediate/HVH_001/HVH_001_01/pages/ # OCR từng trang sẽ ghi ở đây
data/output/HVH_001/HVH_001_01/
├── HVH_001_01_raw.txt
└── HVH_001_01_seg.tsv
```

`manifest.csv` lưu checksum ảnh gốc, tọa độ crop, thứ tự trang, chỉ số
blur/độ sáng/tương phản và các cảnh báo chất lượng. `summary.json`
tổng hợp số ảnh thành công, số spread đã tách và lỗi.

## Tác phẩm nhiều quyển

`HVH_018` (Chế nghệ tinh hoa) có sáu quyển, chạy lần lượt với
`--chapter 01` đến `--chapter 06`. `HVH_024` (Cổ kim truyền lục)
hiện chỉ có quyển 02, vì vậy phải dùng `--chapter 02`; catalog sẽ từ
chối `HVH_024_01`.

## Migration đã áp dụng

Cấu trúc input cũ đã được chuyển bằng
`scripts/2_migrate_input_structure.py`. Báo cáo checksum nằm tại
Khi thực hiện migration, script có thể tạo `configs/input_migration_report.json`
làm báo cáo checksum cục bộ. File này không bắt buộc phải commit. Script
migration mặc định chỉ dry-run;
chỉ `--execute` khi cần thực hiện trên một bản clone cũ.
