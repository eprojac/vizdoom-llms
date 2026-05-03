# DGX Spark / Grace Blackwell-friendly wrapper image.
#
# Default base is the public vLLM OpenAI image. On DGX Spark, NVIDIA's vLLM
# playbook notes that ARM64 + Blackwell may require a DGX Spark-specific
# prebuilt image or a source-built vLLM stack. Override BASE_IMAGE if your
# Spark has a recommended NVIDIA/NVCR vLLM image.
#
# Example:
#   docker build --build-arg BASE_IMAGE=vllm/vllm-openai:nightly -t doom-vlm-duel:dgx-spark .
ARG BASE_IMAGE=vllm/vllm-openai:nightly
FROM ${BASE_IMAGE}

USER root
SHELL ["/bin/bash", "-lc"]

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_ROOT_USER_ACTION=ignore \
    SDL_VIDEODRIVER=dummy \
    XDG_RUNTIME_DIR=/tmp/runtime-root \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    HF_HOME=/root/.cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface/hub

# System libs cover ViZDoom, OpenCV video writing, and source-build fallback
# for packages that may not have ARM64 wheels on every image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    ffmpeg \
    git \
    libboost-all-dev \
    libbz2-dev \
    libffi-dev \
    libfluidsynth-dev \
    libgl1 \
    libglib2.0-0 \
    libjpeg-dev \
    liblua5.1-0-dev \
    libopenal-dev \
    libsdl2-2.0-0 \
    libsdl2-dev \
    libsm6 \
    libxext6 \
    libxrender1 \
    ninja-build \
    pkg-config \
    python3-dev \
    unzip \
    wget \
    zlib1g-dev \
 && rm -rf /var/lib/apt/lists/* \
 && mkdir -p /tmp/runtime-root /workspace/doom-vlm-duel

WORKDIR /workspace/doom-vlm-duel
COPY requirements.txt ./requirements.txt
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel \
 && python3 -m pip install --no-cache-dir -r requirements.txt

COPY doom_vlm_duel_dask_vllm.py ./doom_vlm_duel_dask_vllm.py
COPY scenarios ./scenarios
COPY README.md ./README.md
COPY container_run.sh ./container_run.sh
RUN chmod +x ./container_run.sh

ENTRYPOINT ["/workspace/doom-vlm-duel/container_run.sh"]
CMD ["python3", "doom_vlm_duel_dask_vllm.py", "--help"]
