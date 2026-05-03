#!/usr/bin/env bash
set -euo pipefail

cd /workspace/doom-vlm-duel

if [[ "${HF_TOKEN:-}" == "" ]]; then
  unset HF_TOKEN
  echo "[doom-vlm] HF_TOKEN is not set; using anonymous Hugging Face access for public model repos."
else
  echo "[doom-vlm] HF_TOKEN is set; using authenticated Hugging Face access."
fi

export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" runs

echo "[doom-vlm] Start time: $(date -Is)"
echo "[doom-vlm] Working dir: $(pwd)"
echo "[doom-vlm] HF_HOME=$HF_HOME"
echo "[doom-vlm] HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_HUB_CACHE"
echo "[doom-vlm] Existing HF cache size: $(du -sh "$HF_HOME" 2>/dev/null | awk '{print $1}')"
echo "[doom-vlm] Python: $(python3 --version 2>&1)"
echo "[doom-vlm] vLLM: $(python3 -m vllm.entrypoints.openai.api_server --help >/dev/null 2>&1 && vllm --version 2>&1 || true)"
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[doom-vlm] nvidia-smi:"
  nvidia-smi || true
else
  echo "[doom-vlm] nvidia-smi not found inside container. GPU injection may be broken."
fi

echo "[doom-vlm] Command: $*"
exec "$@"
