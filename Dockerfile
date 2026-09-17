# syntax=docker/dockerfile:1
# Select cpu (default) or gpu with --build-arg DEVICE=gpu.
ARG DEVICE=cpu

FROM node:22-bookworm-slim AS frontend
WORKDIR /build/webui
COPY webui/package.json webui/package-lock.json ./
RUN npm ci
COPY webui/ ./
RUN npm run build

FROM python:3.11-slim-bookworm AS python-base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SSTRANSCRIBER_WORK_DIR=/data \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch \
    XDG_CACHE_HOME=/cache \
    NUMBA_CACHE_DIR=/cache/numba
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 libgomp1 ca-certificates build-essential git tini \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --no-cache-dir pip==25.3 setuptools==80.9.0 wheel==0.45.1
WORKDIR /app
COPY docker/constraints.txt /opt/constraints.txt
ENV PIP_CONSTRAINT=/opt/constraints.txt

FROM python-base AS cpu
RUN python -m pip install --no-cache-dir \
    torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cpu

FROM python-base AS gpu
# CUDA 12.8 wheels include the CUDA/cuDNN user-space libraries and Blackwell support.
# The host supplies the NVIDIA driver via NVIDIA Container Toolkit / Docker Desktop.
RUN python -m pip install --no-cache-dir \
    torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu128
# CTranslate2 loads these libraries dynamically, independently of PyTorch.
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

FROM ${DEVICE} AS application
COPY requirements-webui.txt ./
COPY docker/requirements-audio.txt /opt/requirements-audio.txt
RUN python -m pip install --no-cache-dir -r requirements-webui.txt -r /opt/requirements-audio.txt \
    && python -m pip check
COPY backends/ ./backends/
COPY web_api.py transcribe_common.py process_audio_folder.py embed_lyrics.py clear_embedded_lyrics.py ./
COPY --from=frontend /build/webui/dist ./webui/dist
COPY docker/verify_runtime.py /opt/verify_runtime.py
ARG APP_UID=1000
ARG APP_GID=1000
RUN groupadd --gid ${APP_GID} app \
    && useradd --uid ${APP_UID} --gid ${APP_GID} --create-home app \
    && mkdir -p /data/input /data/output /data/temp /cache \
    && chown -R app:app /data /cache
USER app
RUN python /opt/verify_runtime.py
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/ready', timeout=3).close()"
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "uvicorn", "web_api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-graceful-shutdown", "20"]

# Build --target test to run the deterministic API/pipeline suite in Linux.
FROM application AS test
USER root
COPY requirements-test.txt ./
RUN python -m pip install --no-cache-dir -r requirements-test.txt
COPY tests/ ./tests/
USER app
RUN python -m unittest discover -s tests -v

FROM application AS runtime