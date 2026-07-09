# HVH Pipeline — Ảnh chữ Hán cổ → OCR → NER

Xây dựng ngữ liệu đơn ngữ chữ Hán chuyên ngành lịch sử VN từ ảnh scan/chụp.

---

## Luồng xử lý

```
Ảnh (.jpg/.png/.tif)
       │
       ▼
[1] Tiền xử lý ảnh
    - Deskew (chỉnh nghiêng)
    - Denoise (khử nhiễu)
    - Adaptive threshold (cân bằng sáng tối)
       │
       ▼
[2] OCR — PaddleOCR (lang=ch)
    → <corpus_id>_raw.txt
       │
       ▼
[3] Tách câu chữ Hán cổ
    - Theo dấu câu: 。！？；…
    - Gộp câu ngắn, fallback 20 ký tự
    → <corpus_id>_seg.tsv
       │
       ▼
[4] NER — HanLP (chính) + Claude LLM (backup)
    Nhãn: PER, LOC, ORG, TITLE, TME, NUM, DYNASTY
    → <corpus_id>_ner.json
```

---

## Cài đặt nhanh (Mac M-series)

```bash
# Bước 1: Cài Docker Desktop
# https://www.docker.com/products/docker-desktop/

# Bước 2: Cấu hình API key (nếu dùng LLM cho NER)
cp .env.example .env
# Mở .env, điền ANTHROPIC_API_KEY hoặc OPENAI_API_KEY

# Bước 3: Build (lần đầu ~20 phút)
make build

# Bước 4: Khởi động
make up
# → Jupyter tại http://localhost:8888/?token=hvhteam
```

---

## Đặt ảnh đầu vào

```
data/input/
└── HVH_001/           ← Tác phẩm
    ├── HVH_001_01/    ← Quyển 1 (đặt ảnh vào đây)
    │   ├── page_001.jpg
    │   ├── page_002.jpg
    │   └── ...
    └── HVH_001_02/    ← Quyển 2
        └── ...
```

**Định dạng ảnh hỗ trợ:** `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`, `.bmp`

---

## Chạy pipeline

```bash
# Xử lý 1 quyển
make run ID=HVH_001_01

# Xử lý tất cả quyển của 1 tác phẩm
make run-all WORK=HVH_001

# Chỉ dùng HanLP, không gọi LLM API
make run-no-llm ID=HVH_001_01

# Kiểm tra chất lượng output
make check ID=HVH_001_01
```

---

## Cấu trúc output

```
data/output/
└── HVH_001_01/
    ├── HVH_001_01_raw.txt     ← OCR thô
    ├── HVH_001_01_seg.tsv     ← Câu đã tách
    └── HVH_001_01_ner.json    ← Kết quả NER
```

### Format _seg.tsv
```
HVH_001_01_000001	春正月帝幸布海口。
HVH_001_01_000002	二月詔諸路收集遺書。
```

### Format _ner.json
```json
[{
  "sentence_id": "HVH_001_01_000001",
  "sentence": "春正月帝幸布海口。",
  "entities": [
    {"text": "春正月", "label": "TME"},
    {"text": "布海口", "label": "LOC"}
  ]
}]
```

---

## Lưu ý quan trọng

| Vấn đề | Giải pháp |
|--------|-----------|
| OCR nhận dạng sai chữ cổ | Thêm bước hiệu đính bằng LLM sau OCR |
| Ảnh quá mờ/xấu | Tăng DPI scan lên ≥ 300 DPI |
| Không có dấu câu | Pipeline tự tách 20 ký tự/câu |
| Model tải lần đầu chậm | Cache vào Docker volume, các lần sau nhanh |

### Dung lượng model (tải lần đầu):
- PaddleOCR (ch): ~300 MB
- HanLP MTL: ~400 MB
- (Nếu dùng LLM): Không tải model, chỉ gọi API
