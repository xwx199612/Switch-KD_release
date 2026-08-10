FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    PYTHONPATH=/app \
    VLM_CONFIG_PATH=/config/runtime.yaml \
    HF_HOME=/output/cache/huggingface \
    TRANSFORMERS_CACHE=/output/cache/huggingface \
    TRITON_CACHE_DIR=/output/cache/triton \
    TORCH_HOME=/output/cache/torch \
    XDG_CACHE_HOME=/output/cache

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       software-properties-common ca-certificates curl build-essential \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
       python3.11 python3.11-dev python3.11-venv libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-runtime.lock /app/requirements-runtime.lock
RUN python3.11 -m ensurepip --upgrade \
    && python3.11 -m pip install --no-cache-dir --upgrade pip \
    && python3.11 -m pip install --no-cache-dir \
       --index-url https://download.pytorch.org/whl/cu124 \
       torch==2.6.0 torchvision==0.21.0 \
    && python3.11 -m pip install --no-cache-dir -r /app/requirements-runtime.lock

COPY app/ /app/
COPY config/ /config/
COPY scripts/entrypoint.sh /usr/local/bin/entrypoint.sh
COPY scripts/validate_assets.py /app/scripts/validate_assets.py

RUN useradd --create-home --uid 10001 runtime \
    && mkdir -p /output/cache/huggingface /output/cache/triton /output/cache/torch \
    && chown -R runtime:runtime /output /app /config /usr/local/bin/entrypoint.sh \
    && python3.11 -m compileall -q /app/vlm_distill

USER runtime
EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["python3.11", "-m", "uvicorn", "vlm_distill.docker_service:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
