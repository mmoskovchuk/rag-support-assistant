FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models

# System packages: Tesseract + Ukrainian/English language data, Poppler for PDF rendering,
# curl for the container health check.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr tesseract-ocr-ukr tesseract-ocr-eng poppler-utils curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch first: the default wheel bundles CUDA and is several GB larger.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

# Dependencies before source code, so code edits do not invalidate this cached layer.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY support_rag ./support_rag

# Run as an unprivileged user. UID 1000 matches the default first user on most
# Linux hosts, so files written to the bind-mounted ./data stay editable.
RUN useradd --uid 1000 --create-home appuser \
    && mkdir -p /data /models \
    && chown appuser:appuser /data /models
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "support_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
