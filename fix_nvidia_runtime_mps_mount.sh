#!/usr/bin/env bash
set -euo pipefail

# Last-resort repair for:
#   failed to fulfil mount request: open /usr/bin/nvidia-cuda-mps-control: no such file or directory
#
# This app does not use CUDA MPS. Prefer run_dgx_spark_cdi.sh first.
# This script either creates the expected symlink if the binary exists elsewhere,
# or removes stale nvidia-cuda-mps-control entries from NVIDIA Container Toolkit
# host-files CSVs after backing them up.

if [[ $EUID -ne 0 ]]; then
  echo "Run with sudo: sudo ./fix_nvidia_runtime_mps_mount.sh" >&2
  exit 1
fi

if [[ -e /usr/bin/nvidia-cuda-mps-control ]]; then
  echo "/usr/bin/nvidia-cuda-mps-control already exists. Nothing to change."
  exit 0
fi

candidate=""
for p in /usr/local/cuda/bin/nvidia-cuda-mps-control /usr/local/cuda-*/bin/nvidia-cuda-mps-control; do
  if [[ -x "$p" ]]; then
    candidate="$p"
    break
  fi
done

if [[ -n "$candidate" ]]; then
  ln -s "$candidate" /usr/bin/nvidia-cuda-mps-control
  echo "Created symlink: /usr/bin/nvidia-cuda-mps-control -> $candidate"
  exit 0
fi

changed=0
for f in \
  /etc/nvidia-container-runtime/host-files-for-container.d/*.csv \
  /usr/share/nvidia-container-toolkit/host-files-for-container.d/*.csv \
  /etc/nvidia-container-toolkit/host-files-for-container.d/*.csv; do
  [[ -e "$f" ]] || continue
  if grep -q 'nvidia-cuda-mps-control' "$f"; then
    cp -a "$f" "$f.bak.$(date +%Y%m%d-%H%M%S)"
    sed -i '/nvidia-cuda-mps-control/d' "$f"
    echo "Removed stale nvidia-cuda-mps-control entry from $f after backup."
    changed=1
  fi
done

if [[ "$changed" -eq 0 ]]; then
  echo "No candidate binary and no Toolkit CSV entries found. Repair NVIDIA Container Toolkit / CUDA Toolkit on the host."
  exit 2
fi

systemctl restart docker || true
echo "Done. Retry docker compose up."
