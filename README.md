# Doom VLM Duel on NVIDIA DGX Spark

Run a two-player ViZDoom deathmatch where:

- Player 1 is `RedHatAI/Qwen3.6-35B-A3B-NVFP4`.
- Player 2 is `RedHatAI/gemma-4-26B-A4B-it-NVFP4`.
- Each model receives its own 640x360 first-person frame.
- Each model must answer with exactly one quoted action word, for example `"FORWARD"`.
- If the model returns anything invalid, the frame decision is discarded.
- While inference is pending, the previous valid action remains held.
- Dask runs the two model calls concurrently.
- vLLM serves both models through local OpenAI-compatible endpoints.
- The match records continuous 360p POV videos for both model players with debug overlays.
- The match also records a tactical map/radar stream with ViZDoom sector/linedef geometry, both model players, deduped visible enemies/bots, and live frag counters.

The package is designed for a single DGX Spark. DGX Spark is a Grace Blackwell ARM64 system with 128 GB unified memory, so both vLLM servers share the same accelerator and memory pool.

## Files

```text
.
├── Dockerfile
├── docker-compose.yml
├── docker-compose.cdi.yml
├── doom_vlm_duel_dask_vllm.py
├── scenarios/
│   ├── basic.cfg
│   └── basic_1v1.wad
├── tools/
│   └── make_basic_1v1_wad.py
├── requirements.txt
├── .env.example
├── run_dgx_spark.sh
├── run_dgx_spark_cdi.sh
├── fix_nvidia_runtime_mps_mount.sh
└── README.md
```

## One-command run

From this directory on the DGX Spark:

```bash
HF_TOKEN=hf_your_token_if_needed docker compose up --build --abort-on-container-exit doom-vlm-duel
```

If the model repos are accessible without authentication, this also works:

```bash
docker compose up --build --abort-on-container-exit doom-vlm-duel
```

## Running with no Hugging Face key

The default run path supports no token. Leave `HF_TOKEN` unset:

```bash
docker compose -f docker-compose.cdi.yml up --build --abort-on-container-exit doom-vlm-duel
```

The wrapper inside the container prints this at startup:

```text
[doom-vlm] HF_TOKEN is not set; using anonymous Hugging Face access for public model repos.
```

If a model repo later becomes gated/private, Hugging Face/vLLM will fail with an authentication error in `runs/latest/vllm_qwen.log` or `runs/latest/vllm_gemma.log`. In that case you need to accept the model terms on Hugging Face and run with `HF_TOKEN=...`.

## If the container exits code 0 without downloading models

First check the actual command and startup log:

```bash
docker logs doom-vlm-duel --tail=200
docker compose -f docker-compose.cdi.yml config | sed -n '/command:/,/volumes:/p'
docker inspect doom-vlm-duel --format='Entrypoint={{json .Config.Entrypoint}} Cmd={{json .Config.Cmd}}'
```

With the patched package you should always see `[doom-vlm]` startup lines. If you do not, Docker is running an older image or compose file. Force a clean rebuild:

```bash
docker compose -f docker-compose.cdi.yml down --remove-orphans
DOCKER_BUILDKIT=1 docker compose -f docker-compose.cdi.yml build --no-cache doom-vlm-duel
docker compose -f docker-compose.cdi.yml up --abort-on-container-exit doom-vlm-duel
```

Model downloads happen in the two vLLM subprocess logs:

```bash
tail -f runs/latest/vllm_qwen.log runs/latest/vllm_gemma.log
```

Or use the wrapper:

```bash
HF_TOKEN=hf_your_token_if_needed ./run_dgx_spark.sh
```

The first run will download model weights into `./hf-cache`, which can be very large. This directory is bind-mounted from the host into the container, so downloaded Hugging Face model weights persist on the host filesystem across container rebuilds and reruns.

## Output

By default, outputs are written to `runs/latest/`. If `runs/latest` already exists, the controller renames the previous directory to `runs/latest-yy-mm-dd-hh-mm` before starting the new run, then writes the new run as `runs/latest`.

```text
runs/latest/
├── decisions.csv
├── scoreboard.csv
├── summary.json
├── qwen_player_pov.mp4
├── gemma_player_pov.mp4
├── tactical_map.mp4
├── vllm_qwen.log
└── vllm_gemma.log
```

The videos are continuous game-loop recordings. They are not paused by model thinking. The overlay shows:

- model short name
- model action and actually applied action
- total valid actions
- invalid outputs and errors
- repeat/motionless/escape counters for anti-stuck behavior
- last inference latency
- frame number
- frags and deaths

`scoreboard.csv` records one row per game frame with Qwen/Gemma frags, deaths, inferred head-to-head kills, and the ViZDoom server player frag table.

The third stream, `tactical_map.mp4`, is a publishing/debug radar. It draws an editor-like 2D map background from ViZDoom sector/linedef metadata, then plots the two model-controlled players using their ViZDoom player coordinates. It also plots visible enemies/bots as orange squares when the installed ViZDoom build exposes usable object or label world-position metadata. Bot markers are deduped by object id/proximity, markers too close to the two model players are ignored, and the visible bot count is capped to `BOTS` so the map does not show more extra bots than the match spawned. If metadata is unavailable, the map video still records and shows `visible bots: 0/<BOTS>` instead of failing the run.

Note: the POV MP4 streams are the two model-controlled player POVs. Built-in ViZDoom bots do not expose independent POV streams through this simple two-player controller. To record every non-model bot POV, you would need to spawn each bot as an additional player client rather than using `addbot`.

Before the match timer starts, the controller sends one tiny warmup image to each vLLM server. This absorbs slow first-request multimodal setup, and it fails early if a model cannot return one allowed action.

## ViZDoom maps and configs

The default map setting is `DOOM_MAP=basic.cfg`. Bare `.cfg` names are resolved from the repo's `scenarios/` directory before ViZDoom's installed scenario directory, so this loads `scenarios/basic.cfg`. This repo config points at `scenarios/basic_1v1.wad`, a simple rectangular 1v1 arena with the two model players on opposite sides, a medikit in the middle, a shotgun plus shells near one middle wall, and green armor near the opposite middle wall. The generated WAD can be rebuilt with:

```bash
python3 tools/make_basic_1v1_wad.py
```

`DOOM_MAP` accepts either a Doom map name or a ViZDoom `.cfg` path:

```bash
# Use MAP01 from the default repo scenario.
DOOM_MAP=map01 docker compose up --build --abort-on-container-exit doom-vlm-duel

# Load a specific ViZDoom config. Relative paths are resolved from the repo,
# then from ViZDoom's installed scenarios directory.
DOOM_MAP=basic.cfg docker compose up --build --abort-on-container-exit doom-vlm-duel
```

The compose files mount `./scenarios` into the container, so edits to repo scenario configs are visible without baking a new image. Code changes still require a rebuild.

## Model cache / host filesystem

Yes: by default the compose file keeps model files on the host. It bind-mounts:

```text
${HF_CACHE_DIR:-./hf-cache} -> /root/.cache/huggingface
```

Inside the container, Hugging Face and vLLM use `/root/.cache/huggingface`; on the host this is `./hf-cache` unless you override `HF_CACHE_DIR`. For a durable DGX Spark cache outside the project directory, use an absolute path:

```bash
HF_CACHE_DIR=/home/$USER/.cache/huggingface docker compose up --build --abort-on-container-exit doom-vlm-duel
```


## If container start fails on `nvidia-cuda-mps-control`

If the image builds but Docker fails before the container starts with:

```text
failed to fulfil mount request: open /usr/bin/nvidia-cuda-mps-control: no such file or directory
```

then the app has not started yet. This is a DGX Spark host NVIDIA Container Toolkit/runtime issue: the legacy runtime hook is trying to bind-mount an optional CUDA MPS host binary that is referenced by the host runtime configuration but absent on the host filesystem.

Preferred fix: use the CDI compose file, which asks Docker for the CDI device
directly with `devices: nvidia.com/gpu=all` and does not require a registered
`runtime: nvidia` Docker runtime:

```bash
nvidia-ctk cdi list
HF_TOKEN=hf_your_token_if_needed docker compose -f docker-compose.cdi.yml up --build --abort-on-container-exit doom-vlm-duel
```

If `nvidia.com/gpu=all` is missing from `nvidia-ctk cdi list`, refresh the CDI
spec and retry:

```bash
sudo systemctl restart nvidia-cdi-refresh.service || sudo nvidia-ctk cdi generate --output=/var/run/cdi/nvidia.yaml
```

Or use the helper, which checks for the CDI device and refreshes the spec if
needed:

```bash
HF_TOKEN=hf_your_token_if_needed ./run_dgx_spark_cdi.sh
```

To select a specific CDI device, override `NVIDIA_CDI_DEVICE`:

```bash
NVIDIA_CDI_DEVICE=nvidia.com/gpu=0 ./run_dgx_spark_cdi.sh
```

Only configure Docker's legacy NVIDIA runtime if another workload explicitly
needs `runtime: nvidia`:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Diagnostics:

```bash
nvidia-ctk --version
nvidia-ctk cdi list
which nvidia-cuda-mps-control || true
sudo grep -R "nvidia-cuda-mps-control" \
  /etc/nvidia-container-runtime \
  /etc/nvidia-container-toolkit \
  /usr/share/nvidia-container-toolkit 2>/dev/null || true
```

Last resort, because it edits host NVIDIA Container Toolkit files after backup:

```bash
sudo ./fix_nvidia_runtime_mps_mount.sh
```

This project does not use CUDA MPS, so removing a stale MPS-control mount entry is safe for this workload, but the CDI compose path is cleaner and easier to revert.

## Tuning by environment variables

You can pass settings inline:

```bash
DURATION_S=120 BOTS=2 docker compose up --build --abort-on-container-exit doom-vlm-duel
```

Common variables:

| Variable | Default | Meaning |
|---|---:|---|
| `BOTS` | `0` | Number of extra ViZDoom built-in bots. Default is a true Qwen-vs-Gemma 1v1. |
| `DURATION_S` | `300` | Match duration in seconds. |
| `RUN_NAME` | `latest` | Output subdirectory under `runs/`. |
| `DOOM_MAP` | `basic.cfg` | Doom map name such as `map01`, or a ViZDoom `.cfg` scenario path. |
| `RECORD_FPS` | `35` | MP4 container FPS. This is for playback timing, not model inference FPS. |
| `MAP_VIDEO_SCALE` | `20` | Tactical map world units per pixel. The script auto-zooms out if actors would fall outside the frame. |
| `MAX_MODEL_LEN` | `4096` | vLLM max context length. Lower if memory is tight. |
| `QWEN_GPU_MEM` | `0.34` | vLLM GPU memory fraction for Qwen server. |
| `GEMMA_GPU_MEM` | `0.62` | vLLM GPU memory fraction for Gemma server. |
| `MAX_NUM_BATCHED_TOKENS` | `3072` | vLLM batch token budget. Gemma4 needs this above its per-image token budget. |
| `QWEN_KV_CACHE_MEMORY_BYTES` | `2G` | Explicit Qwen KV cache size; avoids fragile auto-sizing while Gemma is resident. |
| `GEMMA_KV_CACHE_MEMORY_BYTES` | `2G` | Explicit Gemma KV cache size; enough for one 4096-token request with margin. |
| `VLLM_STARTUP_TIMEOUT_S` | `0` | Seconds to wait for vLLM startup. `0` waits indefinitely so first-run downloads are not killed. |
| `HF_CACHE_STATUS_INTERVAL_S` | `10` | Seconds between Hugging Face cache status lines while vLLM starts. |
| `REQUEST_TIMEOUT_S` | `30` | Per-frame model request timeout. |
| `STUCK_ESCAPE_FRAMES` | `24` | Motionless frames before the controller temporarily applies an escape action. |
| `BASE_IMAGE` | `vllm/vllm-openai:nightly` | Base image for the wrapper Dockerfile. |
| `HF_CACHE_DIR` | `./hf-cache` | Host path for Hugging Face cache. This is where model weights persist. Use an absolute path for shared system cache. |
| `VLLM_CACHE_DIR` | `./vllm-cache` | Host path for vLLM compile/cache artifacts so warmup work is reused. |

For longer publishing runs:

```bash
RUN_NAME=pub_001 DURATION_S=900 BOTS=0 RECORD_FPS=35 docker compose up --build --abort-on-container-exit doom-vlm-duel
```

For a short smoke test:

```bash
RUN_NAME=smoke DURATION_S=30 BOTS=0 docker compose up --build --abort-on-container-exit doom-vlm-duel
```

## DGX Spark-specific notes

1. Confirm Docker can see the GPU:

   ```bash
   docker run --rm --gpus all nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04 nvidia-smi
   docker run --rm --device nvidia.com/gpu=all nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04 nvidia-smi
   ```

2. Confirm Docker Compose GPU support. The default `docker-compose.yml` uses Docker GPU device requests through `deploy.resources.reservations.devices` with `capabilities: [gpu]`. The CDI path, `docker-compose.cdi.yml`, uses native CDI syntax through `devices: nvidia.com/gpu=all`.

3. DGX Spark ARM64 + Blackwell vLLM support is newer than mainstream x86 CUDA setups. NVIDIA's DGX Spark vLLM playbook says Spark can use a prebuilt Docker container or a source-built vLLM stack with ARM64-specific LLVM/Triton support. If `vllm/vllm-openai:nightly` does not work on your Spark, set `BASE_IMAGE` to the NVIDIA-recommended DGX Spark vLLM image for your installed software stack:

   ```bash
   BASE_IMAGE=your/dgx-spark-vllm-image:tag docker compose up --build --abort-on-container-exit doom-vlm-duel
   ```

4. Both models run on the same Spark accelerator and memory pool. If either vLLM server exits with OOM, lower these first:

```bash
MAX_MODEL_LEN=3072 MAX_NUM_BATCHED_TOKENS=3072 QWEN_KV_CACHE_MEMORY_BYTES=1G GEMMA_KV_CACHE_MEMORY_BYTES=1G docker compose up --build --abort-on-container-exit doom-vlm-duel
```

5. The script starts two vLLM servers inside the same container. Their logs are:

   ```text
   runs/latest/vllm_qwen.log
   runs/latest/vllm_gemma.log
   ```

## Running without Docker

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Install a DGX Spark-compatible vLLM build separately.
python doom_vlm_duel_dask_vllm.py --bots 0 --duration-s 300 --record-fps 35
```

## Troubleshooting

### `vllm: command not found`

Your base image does not include the vLLM CLI. Use a vLLM base image or set `BASE_IMAGE` to a DGX Spark-compatible vLLM image.

### vLLM rejects launch flags

vLLM CLI syntax changes across versions. This package uses the current JSON form for multimodal limits: `--limit-mm-per-prompt '{"image": 1}'`, avoids the removed `--disable-log-requests` flag and the Qwen reasoning parser for this one-word controller path, sets `--max-num-batched-tokens` high enough for Gemma4 image prompts, and uses explicit `--kv-cache-memory-bytes` values so two resident vLLM servers do not depend on auto KV-cache sizing.

### Structured action choice fails

Run the script manually with `--no-guided-choice`, or edit the compose command to add that flag. The script will still reject ambiguous outputs and will recover if a single allowed action appears inside a longer response.

### OpenCV video writer fails

Confirm `ffmpeg` and `opencv-python-headless` are installed in the image. The Dockerfile installs both.

### ViZDoom cannot start display/audio

The container sets `SDL_VIDEODRIVER=dummy` and disables sound. Do not use `--render-window` in the container unless you also mount X11/Wayland correctly.

## Seeing model download/load progress

From v5 onward, the controller mirrors each vLLM server's stdout/stderr into both:

- `docker compose up` output, prefixed as `[vllm-qwen]` and `[vllm-gemma]`
- host log files under `runs/<RUN_NAME>/vllm_qwen.log` and `runs/<RUN_NAME>/vllm_gemma.log`

The controller prefers complete local Hugging Face snapshots under `HF_CACHE_DIR` and starts vLLM in offline mode with `--served-model-name` set to the original repo ID, so completed model weights are reused instead of downloaded again. If a local snapshot is missing, the run fails before vLLM starts; manually run the Python script with `--allow-hf-download` only when you intentionally want to fetch missing weights.

The first run can spend a long time in Hugging Face downloads and vLLM model loading before ViZDoom starts. Later runs should show stable cache size and reuse local snapshots. You can also watch host-side cache state in another terminal:

```bash
watch -n 2 'du -sh hf-cache runs/latest 2>/dev/null; tail -n 20 runs/latest/vllm_qwen.log runs/latest/vllm_gemma.log 2>/dev/null'
```

If the container is stuck at `Waiting for vLLM servers...`, check whether either vLLM process exited early. v5 now fails fast if a child vLLM server dies before `/v1/models` becomes ready.
