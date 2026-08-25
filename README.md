# NLP K35 — OCR Hán–Nôm và tạo ground truth ảnh Facebook

Repository gồm hai flow độc lập, dùng chung môi trường Python nhưng không dùng
chung dữ liệu hoặc output:

| Flow | Code | Tài liệu vận hành |
|---|---|---|
| Hán–Nôm: tải scan → preprocessing → PaddleOCR → raw/seg → validator | `scripts/1_...py` đến `scripts/8_...py` | `scripts/README.md` và `docs/han_nom/` |
| Facebook: fetch ảnh → Gemini OCR → chia same/diff → PP-OCRv6 → vision LLM adjudication | `scripts/facebook/` | `scripts/facebook/README.md` |

Phần dưới mô tả chi tiết sản phẩm Hán–Nôm. Đối với pipeline Facebook đang xử
lý, xem [hướng dẫn đầy đủ](scripts/facebook/README.md), trong đó có data
contract, strict join, từng lệnh chạy, tham số và metric đánh giá.

Project tải ảnh từ Nom Foundation Library, tách scan thành từng trang, phát
hiện cột chữ dọc, OCR bằng PaddleOCR, ghép văn bản thô và tách câu theo dấu
son/dấu đỏ.

Mỗi folder nguồn là một thư mục cấp ngoài; mỗi ảnh scan gốc là một đơn vị
output riêng. Bộ hiện tại có 32 folder nguồn và 1.994 ảnh:

```text
final_output/
├── HVH_001/
│   ├── HVH_001_0001/
│   │   ├── HVH_001_0001_raw.txt
│   │   └── HVH_001_0001_seg.tsv
│   ├── ...
│   └── HVH_001_0048/
├── HVH_002/                    # 60 ảnh → 60 folder con
├── ...
└── HVH_032/                    # 76 ảnh → 76 folder con
```

- `raw.txt`: OCR thô để đối chiếu.
- `seg.tsv`: `[sentence_id]\t[sentence]`, mã hóa UTF-8.
- Project không tạo file NER.

## 1. Project structure

| File | Nhiệm vụ | Điều kiện |
|---|---|---|
| `README.md` | Tài liệu chính: cài môi trường, chạy pipeline, xử lý lỗi và chuẩn bị GitHub | Bắt buộc |
| `docs/han_nom/RUN_FROM_SCRATCH.md` | Danh sách lệnh ngắn để chạy lại pipeline Hán–Nôm | Không bắt buộc, nên giữ để tra cứu nhanh |
| `docs/han_nom/PREPROCESSING.md` | Giải thích thuật toán cắt viền, tách trang, biến thể ảnh và quality flags | Không bắt buộc, nên giữ cho báo cáo kỹ thuật |
| `docs/han_nom/PIPELINE_AUDIT.md` | Ghi lại kết quả thử nghiệm và giới hạn của Paddle/LLM | Không bắt buộc, dùng để giải thích quyết định thiết kế |
| `.gitignore` | Ngăn Git upload API key, ảnh, model, cache và output sinh ra | Bắt buộc khi dùng Git/GitHub |
| `.env.example` | Mẫu cấu hình API không chứa key thật | Bắt buộc nếu chia sẻ bước API/LLM |
| `.env` | Chứa API key thật trên từng máy | Chỉ cần khi chạy API/LLM; tuyệt đối không commit |
| `environment.yml` | Tạo Conda environment `NLP` với đúng Python và package | Khuyến nghị, cần Conda/Miniforge |
| `requirements.txt` | Phiên bản package chính; được `environment.yml` sử dụng | Bắt buộc khi tạo môi trường |
| `requirements-lock.txt` | Cài package bằng `pip`/`venv` thay cho Conda | Chỉ bắt buộc nếu cài bằng pip |
| `requirements-preprocess.txt` | Chỉ cài NumPy và OpenCV | Chỉ dùng khi muốn chạy riêng bước tiền xử lý |
| `configs/corpus_catalog.csv` | Ánh xạ 32 folder nguồn sang vị trí xử lý nội bộ và mã output chấm máy | Bắt buộc |
| `configs/ocr_policy.json` | Ghi lại engine, phiên bản và quyết định OCR từ pilot | Không bắt buộc khi chạy; nên giữ làm metadata |
| `data/input/link.txt` | Danh sách URL volume để bước 1 tải ảnh | Bắt buộc khi tải batch |
| `data/input/` | Ảnh gốc tải từ nguồn, không được chỉnh sửa | Được sinh bởi bước 1; không commit ảnh JPG |
| `data/processed/` | Ảnh đã cắt viền và tách thành từng trang | Được sinh bởi bước 3; không commit |
| `data/intermediate/` | OCR JSON, crop cột, report và dữ liệu resume | Được sinh bởi bước 4–8; không commit |
| `final_output/` | 32 folder nguồn; bên trong mỗi ảnh có một folder `_NNNN` chứa raw/seg | Được sinh bởi bước 6; tổng dự kiến 1.994 folder con và 3.988 file |
| `models/paddlex/` | Cache model detection/recognition của PaddleOCR | Tự tải ở lần OCR đầu; không commit |
| `scripts/1_download_nomfoundation_images.py` | Tải ảnh từ Nom Foundation, retry và kiểm tra catalog | Bắt buộc khi máy chưa có ảnh input; cần Internet |
| `scripts/2_migrate_input_structure.py` | Công cụ tương thích cho bản dữ liệu cũ từng bị gom theo tác phẩm/quyển | Legacy, không chạy với bộ input 32 folder hiện tại |
| `scripts/3_preprocess_images.py` | Kiểm tra ảnh, cắt viền, tách hai trang và tạo manifest | Bắt buộc |
| `scripts/4_ocr_local.py` | Tách cột, OCR Paddle/API, fallback full-page và resume | Bắt buộc; Paddle local không cần API key |
| `scripts/5_compare_ocr.py` | So sánh pilot full-page với columns | Không bắt buộc cho mỗi lần chạy; nên dùng khi đổi kiểu tài liệu |
| `scripts/6_build_outputs.py` | Ghép OCR, phát hiện dấu son và tạo hai output | Bắt buộc |
| `scripts/7_check_output.py` | Kiểm tra UTF-8, ID câu, TSV và coverage | Bắt buộc trước khi nộp/chia sẻ output |
| `scripts/8_llm_correct_segments.py` | Vision LLM hỗ trợ hiệu chỉnh và tách câu có guard | Không bắt buộc; cần API key hoặc Ollama |
| `scripts/README.md` | Tóm tắt thứ tự và ví dụ gọi các script | Không bắt buộc, dùng để tra cứu nhanh |
| `scripts/facebook/` | Pipeline 5 bước tạo ground truth ảnh Facebook | Bắt buộc cho giai đoạn Facebook |
| `scripts/facebook/README.md` | Flow, cấu trúc, ràng buộc join, lệnh chạy, tham số và metric Facebook | Bắt buộc đọc trước khi chạy giai đoạn Facebook |
| `data_packages/` | Archive dữ liệu để chuyển qua Google Drive | Dữ liệu local; tuyệt đối không commit và không đổi path khi đang tải |
| `.git/` | Lịch sử, branch, index và remote của Git | Git tự quản lý; không chỉnh sửa thủ công |

Các thư mục `__pycache__` và file `.pyc` do Python tự sinh, không thuộc mã
nguồn và đã được `.gitignore` loại trừ.

`images/`, `facebook_posts_valid.jsonl`, `facebook_posts_ocr.jsonl` và
`image_ocr_output/` ở root là dữ liệu lịch sử đã được nhánh nhóm theo dõi từ
trước. Pipeline Facebook mới không đọc các path này; nó chỉ dùng
`data/mrDuc_data/` và `data/output/`. Không thêm dữ liệu mới vào các path
legacy và không di chuyển/xóa hàng nghìn file đã track nếu chưa thống nhất với
trưởng nhóm.

## 2. Pipeline

```text
Ảnh gốc
  → Kiểm tra và tách hai trang
  → Phát hiện cột dọc theo thứ tự phải sang trái
  → OCR từng cột bằng PaddleOCR
  → Fallback OCR nguyên trang nếu không phát hiện được cột
  → Ghép raw.txt
  → Phát hiện dấu son và tạo seg.tsv
  → Validator
  → LLM correction tùy chọn
```

Phương pháp chính là OCR từng cột. Pilot trên sáu trang `HVH_001_01` cho
confidence Paddle trung bình 67,275% và 1.023 ký tự CJK, so với 66,213% và
1.005 ký tự khi OCR nguyên trang. Đây là heuristic, không phải accuracy.

## 3. Môi trường đã kiểm thử

- Linux x86_64, CPU.
- Python 3.10.20.
- PaddlePaddle 3.3.1.
- PaddleOCR 3.7.0.
- PaddleX 3.7.2.
- NumPy 2.2.6.
- OpenCV headless 5.0.0.93.
- Dung lượng trống tối thiểu khoảng 5 GB khi chạy nhiều tác phẩm.

## 4. Clone và cài môi trường

```bash
git clone <URL_REPOSITORY>
cd xlnntn_hp2_k35

conda env create -f environment.yml
conda activate NLP
```

Nếu môi trường `NLP` đã tồn tại:

```bash
conda env update -n NLP -f environment.yml --prune
conda activate NLP
```

Kiểm tra phiên bản:

```bash
python --version
python -c "import cv2, numpy, paddle; print(cv2.__version__, numpy.__version__, paddle.__version__)"
python -c "from paddleocr import PaddleOCR; print('PaddleOCR import OK')"
```

Lần chạy PaddleOCR đầu tiên có thể tải model. Cache nằm trong
`models/paddlex` và không được commit lên GitHub.

Nếu không dùng Conda:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
```

Conda vẫn là phương án được khuyến nghị.

## 5. Cấu hình API tùy chọn

PaddleOCR local không cần API key. Chỉ bước 8 hoặc OCR vision API mới cần
`.env`:

```bash
cp .env.example .env
```

Ví dụ cấu hình nhà cung cấp tương thích OpenAI:

```dotenv
OCR_API_KEY=replace_with_your_key
OCR_BASE_URL=https://your-provider.example/v1
OCR_MODEL=replace_with_your_vision_model
```

Không commit `.env`. Kiểm tra trước khi push:

```bash
git check-ignore -v .env
git grep -n "OCR_API_KEY=" -- ':!.env.example' || true
```

## 6. Cấu trúc thư mục

```text
.
├── configs/
│   ├── corpus_catalog.csv
│   └── ocr_policy.json
├── data/
│   ├── input/          # ảnh gốc, không chỉnh sửa
│   ├── processed/      # ảnh đã tách trang
│   └── intermediate/   # OCR JSON, crop cột và báo cáo
├── final_output/       # 32 folder nguồn, 1.994 đơn vị ảnh
├── scripts/
│   ├── 1_download_nomfoundation_images.py
│   ├── 2_migrate_input_structure.py
│   ├── 3_preprocess_images.py
│   ├── 4_ocr_local.py
│   ├── 5_compare_ocr.py
│   ├── 6_build_outputs.py
│   ├── 7_check_output.py
│   ├── 8_llm_correct_segments.py
│   └── facebook/
│       ├── 1_fetch_images.py
│       ├── 2_ocr_gemini.py
│       ├── 3_compare_gemini_label.py
│       ├── 4_paddlev6_relabel_diff.py
│       ├── 5_llm_adjudicate_ground_truth.py
│       ├── lib/
│       └── experiments/
├── docs/han_nom/
├── data_packages/      # local/Google Drive, không commit
├── environment.yml
├── requirements.txt
└── requirements-lock.txt
```

Ảnh input, processed và intermediate không được lưu trong Git. Người clone
repository sẽ tải lại ảnh bằng bước 1. `final_output` là sản phẩm nộp chung,
có thể commit sau khi validator báo đạt và đã ghép đủ phần của cả nhóm.

## 7. Chạy một quyển từ đầu: HVH_001_01

Chạy mọi lệnh tại thư mục gốc repository sau khi `conda activate NLP`.

### Bước 1 — Tải ảnh

```bash
python scripts/1_download_nomfoundation_images.py \
  --links-file data/input/link.txt \
  --catalog configs/corpus_catalog.csv \
  --out-root data/input
```

Script có retry, timeout, tự bỏ qua ảnh đã tồn tại và ưu tiên ảnh `large`.

Kiểm tra số ảnh một quyển:

```bash
find data/input/HVH_001 -maxdepth 1 -type f -name '*.jpg' | wc -l
```

### Bước 2 — Kiểm tra cấu trúc 32 folder input

```bash
find data/input -mindepth 1 -maxdepth 1 -type d -name 'HVH_*' | sort
python scripts/1_download_nomfoundation_images.py \
  --links-file data/input/link.txt \
  --catalog configs/corpus_catalog.csv \
  --out-root data/input \
  --validate-only
```

Kết quả phải có đúng `HVH_001` đến `HVH_032`. Không chạy
`2_migrate_input_structure.py` cho bộ input này vì script đó chỉ dành cho
bản dữ liệu legacy và có thể gom các folder theo tác phẩm/quyển.

### Bước 3 — Kiểm tra và tiền xử lý ảnh

```bash
python scripts/3_preprocess_images.py \
  --folder HVH_001 --variant color
```

Kết quả:

```text
data/processed/HVH_001/HVH_001_01/
├── manifest.csv
├── summary.json
└── HVH_001_01_NNNN_pNN.jpg
```

`split_confidence` là độ rõ của khe tách trang, không phải confidence OCR.
Kiểm tra `input_images_failed` bằng `0` và số trang processed hợp lý. Không
chỉnh sửa ảnh trong `data/input`.

### Bước 4A — Pilot OCR nguyên trang

```bash
python scripts/4_ocr_local.py \
  --folder HVH_001 \
  --engine paddle --source processed \
  --limit 3 --ocr-layout full-page \
  --run-name paddle_fullpage_pilot
```

### Bước 4B — Pilot OCR từng cột

```bash
python scripts/4_ocr_local.py \
  --folder HVH_001 \
  --engine paddle --source processed \
  --limit 3 --ocr-layout columns \
  --run-name paddle_columns_pilot
```

Crop cột được lưu phải sang trái:

```text
data/intermediate/HVH_001/HVH_001_01/ocr_runs/
└── paddle_columns_pilot/column_crops/processed/<PAGE_ID>/
    ├── col_01.png
    ├── col_02.png
    └── ...
```

Nếu không phát hiện được cột, script fallback OCR nguyên trang và ghi
`layout_fallback: full-page` trong JSON.

### Bước 5 — So sánh hai layout

```bash
python scripts/5_compare_ocr.py \
  --folder HVH_001 \
  --full-page-run paddle_fullpage_pilot \
  --columns-run paddle_columns_pilot
```

Đọc báo cáo:

```bash
jq '{recommended_layout, heuristic_indicators, aggregate_metrics}' \
  data/intermediate/HVH_001/HVH_001_01/ocr_comparison/paddle_fullpage_pilot_vs_paddle_columns_pilot/summary.json
```

Không chọn layout chỉ dựa vào confidence. Cần xem số ký tự CJK, unknown, số
cột và crop đại diện. Không có ground truth thì đây chỉ là heuristic.

### Bước 6 — OCR đầy đủ, có resume

Pipeline hiện chọn OCR từng cột:

```bash
python scripts/4_ocr_local.py \
  --folder HVH_001 \
  --engine paddle --source processed \
  --ocr-layout columns \
  --run-name paddle_columns_full \
  --retry-errors
```

Nếu tiến trình bị dừng, chạy lại đúng lệnh. Các trang `success` và `blank`
được bỏ qua. Kiểm tra:

```bash
jq '.stats' \
  data/intermediate/HVH_001/HVH_001_01/ocr_runs/paddle_columns_full/run_summary.json
```

`error` và `missing` phải bằng `0`. `blank` có thể là trang trắng hoặc bìa,
nhưng bước 6 sẽ từ chối tạo output nếu tỷ lệ `blank` vượt quá 10%.

### Bước 7 — Tạo output

```bash
python scripts/6_build_outputs.py \
  --folder HVH_001 \
  --run-name paddle_columns_full --overwrite
```

Output:

```text
final_output/HVH_001/HVH_001_0001/
├── HVH_001_0001_raw.txt
└── HVH_001_0001_seg.tsv
```

`HVH_001_01` là ID xử lý nội bộ. Khi tạo output, mỗi ảnh nguồn được ánh xạ
một-một: `HVH_001_0001.jpg` thành folder `HVH_001_0001`, ...,
`HVH_001_0048.jpg` thành `HVH_001_0048`. Nếu một scan được tách thành
`p01/p02`, hai phần OCR được ghép lại trong cùng output của ảnh scan đó.

Báo cáo:

```text
data/intermediate/HVH_001/HVH_001_01/build_reports/
├── paddle_columns_full.json
├── paddle_columns_full.red_mark_detections.tsv
└── paddle_columns_full.segmentation_review.tsv
```

### Bước 8 — Validate

```bash
python scripts/7_check_output.py --id HVH_001_0001
```

Kết quả cuối phải là `Output đạt yêu cầu`.

Kiểm tra riêng đủ 48 ảnh của `HVH_001`:

```bash
python scripts/7_check_output.py --folder HVH_001
```

Trước khi nộp, kiểm tra đủ 32 folder nguồn và toàn bộ 1.994 ảnh:

```bash
python scripts/7_check_output.py --all
```

Chỉ nộp khi kết quả là `ĐẠT TOÀN BỘ: 32 folder, 1994 ảnh`.

## 8. Chạy toàn bộ HVH_001 đến HVH_032

Quy trình đầy đủ gồm làm sạch, kiểm tra 1.994 ảnh, preprocess, pilot, OCR,
build theo từng ảnh và validate được trình bày trong
[`RUN_FROM_SCRATCH.md`](docs/han_nom/RUN_FROM_SCRATCH.md). Hãy dùng các vòng lặp trong tài
liệu đó; mỗi bước dừng ngay tại folder lỗi và có kiểm tra số lượng trước khi
chuyển sang bước tiếp theo.

## 9. LLM correction tùy chọn

Bước này không bắt buộc. Model vision phải được pilot trước và không nên
publish nếu kết quả còn `review`.

Provider tương thích OpenAI:

```bash
python scripts/8_llm_correct_segments.py \
  --folder HVH_001 \
  --ocr-run paddle_columns_full \
  --provider compatible --llm-run compatible_pilot \
  --ocr-guidance full --limit 3
```

Ollama local:

```bash
ollama pull qwen3-vl:8b

python scripts/8_llm_correct_segments.py \
  --folder HVH_001 \
  --ocr-run paddle_columns_full \
  --provider ollama --model qwen3-vl:8b \
  --llm-run qwen3vl_pilot --ocr-guidance full --limit 3
```

Không thêm `--publish` cho đến khi toàn bộ trang qua guard và có ground truth.
Bước 8 không sửa `_raw.txt`.

## 10. Phân biệt các chỉ số

- `split_confidence`: độ rõ của khe tách trang.
- Confidence Paddle: độ tự tin nội bộ trên vùng model nhận ra.
- Similarity: độ giống giữa hai output OCR.
- Các chỉ số trên không phải accuracy.

Muốn chứng minh accuracy trên 85%, cần bản chép chuẩn và tính Character Error
Rate: `accuracy = 1 - CER`.

## 11. Lỗi thường gặp

### Không import được PaddleOCR

```bash
conda activate NLP
python -c "from paddleocr import PaddleOCR; print('OK')"
```

Nếu vẫn lỗi, tạo lại env từ `environment.yml`, không cài chồng package.

### Không có `run_summary.json`

Vòng lặp có thể đã dừng ở tác phẩm trước. Xem dòng `LỖI`, sửa rồi chạy lại
với `--retry-errors`; resume giữ các trang đã thành công.

### Không phát hiện được cột

Script tự fallback nguyên trang. Liệt kê các trang fallback:

```bash
find data/intermediate -path '*/paddle_columns_full/processed/*.json' -print0 \
  | xargs -0 jq -r 'select(.layout_fallback=="full-page") | .image_path'
```

### Trang blank

`blank` có thể hợp lệ với trang trắng hoặc bìa. Mở `image_path` trong JSON để
kiểm tra khi cần.

### Validator báo seg khác raw

Output Paddle thông thường không dùng `--allow-corrected`. Chỉ dùng tùy chọn
đó nếu `_seg.tsv` đã được publish bởi bước LLM.

## 12. Chuẩn bị upload GitHub

`.env`, ảnh, model và kết quả sinh ra đã được `.gitignore` loại trừ:

```bash
git status --short
git check-ignore -v .env data/processed data/intermediate models
```

Repository nên chứa:

- `scripts/` và tài liệu Markdown.
- `configs/corpus_catalog.csv`, `configs/ocr_policy.json`.
- `data/input/link.txt`, không chứa JPG.
- `environment.yml`, `requirements.txt`, `requirements-lock.txt`.
- `.env.example`, không chứa key thật.

### Lưu ý về repository hiện tại

Ảnh JPG từng tồn tại trong lịch sử Git, nên thư mục `.git` của bản clone cũ
có thể vẫn rất lớn dù `.gitignore` đã đúng. `.gitignore` không xóa dữ liệu khỏi
lịch sử commit. Cách sạch và dễ chia sẻ nhất là tạo một repository GitHub mới
từ snapshot mã nguồn hiện tại. Nếu tiếp tục dùng repository cũ, cần stage cả
việc xóa các ảnh đã từng được track:

```bash
git add -u data/input
```

Việc này loại ảnh khỏi commit mới nhưng không làm nhỏ lịch sử cũ. Muốn làm nhỏ
lịch sử phải rewrite history bằng `git filter-repo`; chỉ nên thực hiện sau khi
cả nhóm thống nhất vì commit hash và lịch sử remote sẽ thay đổi.

Kiểm tra secret và file lớn trước khi commit:

```bash
git grep -n "OCR_API_KEY=" -- ':!.env.example' || true
find . -type f -size +50M -not -path './.git/*'
```

Stage có chọn lọc:

```bash
git add .gitignore .env.example README.md environment.yml \
  requirements.txt requirements-lock.txt requirements-preprocess.txt \
  configs scripts data/input/link.txt

git add -u data/input

git status --short
git diff --cached --stat
git commit -m "Build reproducible Han-Nom OCR pipeline"
git push
```

Không dùng `git add -f .env` và không commit API key.

## 13. Tài liệu bổ sung

- [Hướng dẫn chạy lại Hán–Nôm](docs/han_nom/RUN_FROM_SCRATCH.md)
- [Chi tiết preprocessing Hán–Nôm](docs/han_nom/PREPROCESSING.md)
- [Báo cáo pipeline Hán–Nôm](docs/han_nom/PIPELINE_AUDIT.md)
- [Pipeline Facebook ground truth](scripts/facebook/README.md)
- [Thứ tự script](scripts/README.md)
