# ============================================================
# Dockerfile — HVH Pipeline (Ảnh → OCR → Tách câu → NER)
# Tối ưu cho Mac M-series (ARM64 / linux/arm64)
# ============================================================
FROM python:3.10-slim

LABEL description="HVH: OCR chữ Hán cổ từ ảnh → tách câu → NER"

# ── Biến môi trường ──────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

WORKDIR /workspace

# ── System dependencies ───────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build tools
    build-essential \
    gcc \
    g++ \
    # Image processing
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    libglib2.0-dev \
    # Xử lý ảnh scan (tiền xử lý trước OCR)
    imagemagick \
    # Tiện ích
    wget \
    curl \
    git \
    poppler-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Python packages ───────────────────────────────────────────
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# ── Copy source ───────────────────────────────────────────────
COPY scripts/ ./scripts/
COPY configs/ ./configs/

EXPOSE 8888

CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", "--port=8888", \
     "--no-browser", "--allow-root", \
     "--NotebookApp.token=hvhteam"]
