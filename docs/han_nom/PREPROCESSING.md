# Kiểm tra và tiền xử lý ảnh

Script `scripts/3_preprocess_images.py` không thay đổi ảnh gốc trong
`data/input`. Mỗi folder nguồn `HVH_001` đến `HVH_032` được giữ độc lập.
Script tra `configs/corpus_catalog.csv` để xác định vị trí xử lý nội bộ,
và tạo output tương ứng một-một với từng ảnh scan trong 32 folder nguồn.

## Cài đặt

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-preprocess.txt
```

## Chạy pilot

Nên xem thử 10 ảnh đầu tiên trước khi xử lý cả tác phẩm:

```bash
python scripts/3_preprocess_images.py --folder HVH_001 --limit 10
```

Sau khi kiểm tra ảnh tách trang và `manifest.csv`, chạy toàn bộ:

```bash
python scripts/3_preprocess_images.py --folder HVH_001 --overwrite
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
python scripts/3_preprocess_images.py --folder HVH_001 --limit 10 \
  --variant clahe --processed-root data/processed_clahe
```

## Kết quả

```text
data/input/HVH_001/                  # ảnh gốc, không bị sửa
data/processed/HVH_001/HVH_001_01/  # ảnh đã cắt/tách trang
├── HVH_001_01_0001_p01.jpg
├── HVH_001_01_0001_p02.jpg
├── manifest.csv
└── summary.json
data/intermediate/HVH_001/HVH_001_01/pages/ # OCR từng trang sẽ ghi ở đây
final_output/HVH_001/HVH_001_0001/
├── HVH_001_0001_raw.txt
└── HVH_001_0001_seg.tsv
```

Nếu `HVH_001` có 48 ảnh, cấu trúc trên tiếp tục đến
`final_output/HVH_001/HVH_001_0048/`. Hai trang `p01/p02` tách từ cùng một
scan vẫn được ghép vào một cặp raw/seg của scan đó.

`manifest.csv` lưu checksum ảnh gốc, tọa độ crop, thứ tự trang, chỉ số
blur/độ sáng/tương phản và các cảnh báo chất lượng. `summary.json`
tổng hợp số ảnh thành công, số spread đã tách và lỗi.

## Lưu ý về 32 folder nguồn

Luôn gọi pipeline bằng `--folder HVH_NNN`. Ví dụ ảnh thứ 1 của folder nguồn
`HVH_019` phải tạo `final_output/HVH_019/HVH_019_0001/`, bất kể ID xử lý
nội bộ của nó là gì. Không chạy
`scripts/2_migrate_input_structure.py` với bộ input hiện tại; script đó chỉ
dành cho bản legacy từng bị gom folder.
