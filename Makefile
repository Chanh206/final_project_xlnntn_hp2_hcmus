# ============================================================
# Makefile — HVH Pipeline (Mac M-series)
# ============================================================

CONTAINER = hvh_pipeline
ID        ?= HVH_001_01    # Ghi đè bằng: make run ID=HVH_002_01

.PHONY: build up down shell logs run check clean

## ── Docker commands ─────────────────────────────────────────

# Build image (lần đầu ~20 phút)
build:
	docker compose build

# Khởi động container
up:
	docker compose up -d
	@echo ""
	@echo "✅  Jupyter: http://localhost:8888/?token=hvhteam"

# Dừng
down:
	docker compose down

# Mở shell vào container
shell:
	docker exec -it $(CONTAINER) /bin/bash

# Xem logs real-time
logs:
	docker compose logs -f

## ── Pipeline commands ───────────────────────────────────────

# Tải ảnh từ Nom Foundation Library (1 volume, chỉ định thủ công)
# Sử dụng: make download CODE=nlvnpf-0501 PAGES=48 PREFIX=HVH_001 OUT=data/input/HVH_001/HVH_001_01
download:
	docker exec $(CONTAINER) python scripts/download_nomfoundation_images.py \
		--image-code $(CODE) \
		--pages $(PAGES) \
		--prefix $(PREFIX) \
		--out-dir $(OUT)

# Tải hàng loạt ảnh từ danh sách link volume (data/input/link.txt)
# Sử dụng: make download-batch LINKS=data/input/link.txt
download-batch:
	docker exec $(CONTAINER) python scripts/download_nomfoundation_images.py \
		--links-file $(LINKS)

# Chạy pipeline cho 1 quyển
# Sử dụng: make run ID=HVH_001_01
run:
	docker exec $(CONTAINER) python scripts/pipeline_hvh.py \
		--input data/input/$(ID) \
		--id $(ID)

# Chạy pipeline không có LLM (chỉ HanLP)
run-no-llm:
	docker exec $(CONTAINER) python scripts/pipeline_hvh.py \
		--input data/input/$(ID) \
		--id $(ID) \
		--no-llm

# Chạy tất cả quyển của 1 tác phẩm
# Sử dụng: make run-all WORK=HVH_001
run-all:
	docker exec $(CONTAINER) python scripts/run_all_chapters.py \
		--work $(WORK)

# Kiểm tra chất lượng output
# Sử dụng: make check ID=HVH_001_01
check:
	docker exec $(CONTAINER) python scripts/check_output.py --id $(ID)

## ── Cleanup ─────────────────────────────────────────────────

# Xóa toàn bộ (cẩn thận!)
clean:
	docker compose down --rmi all --volumes
	docker system prune -f
