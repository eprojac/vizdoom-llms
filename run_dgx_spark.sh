#!/usr/bin/env bash
set -euo pipefail

# Single-command wrapper. You can pass environment variables before it, e.g.:
#   HF_TOKEN=hf_xxx DURATION_S=120 ./run_dgx_spark.sh

docker compose up --build --abort-on-container-exit doom-vlm-duel
