#!/usr/bin/env bash
set -euo pipefail

# Use native Docker CDI device injection. This avoids the legacy NVIDIA runtime
# hook path that can fail on some DGX Spark host installs when optional files
# such as /usr/bin/nvidia-cuda-mps-control are referenced but absent.

export NVIDIA_CDI_DEVICE="${NVIDIA_CDI_DEVICE:-nvidia.com/gpu=all}"

if command -v nvidia-ctk >/dev/null 2>&1; then
  if ! nvidia-ctk cdi list 2>/dev/null | grep -Fxq "$NVIDIA_CDI_DEVICE"; then
    echo "CDI GPU spec '$NVIDIA_CDI_DEVICE' not found; trying to refresh it with sudo..."
    sudo systemctl restart nvidia-cdi-refresh.service 2>/dev/null || \
      sudo nvidia-ctk cdi generate --output=/var/run/cdi/nvidia.yaml
  fi

  if ! nvidia-ctk cdi list 2>/dev/null | grep -Fxq "$NVIDIA_CDI_DEVICE"; then
    echo "CDI GPU spec '$NVIDIA_CDI_DEVICE' is still not available." >&2
    nvidia-ctk cdi list >&2 || true
    exit 1
  fi
else
  echo "nvidia-ctk not found on host. Install or repair NVIDIA Container Toolkit first." >&2
  exit 1
fi

docker compose -f docker-compose.cdi.yml up --build --abort-on-container-exit doom-vlm-duel
