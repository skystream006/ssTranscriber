# Docker deployment

The container serves the built React UI and FastAPI on port 8000, using the existing
CLI subprocess pipeline. Local Python/Node installation, localhost binding, Vite
development, and repository-relative data paths remain the defaults outside Docker.

## Prerequisites

- Docker Engine with Compose v2, or Docker Desktop using Linux containers on Windows.
- GPU deployment: NVIDIA Container Toolkit on Linux, or supported Docker Desktop/WSL2
  GPU passthrough on Windows, plus an NVIDIA driver supporting CUDA 12.8. The GPU image
  uses PyTorch CUDA 12.8 wheels (including Blackwell support) and cuDNN 9. It does not
  use the host Python/CUDA toolkit installation. Older GPUs must also be supported by
  the selected PyTorch and CTranslate2 builds.
- Enough disk space for the image, model caches, audio, and Demucs stems. GPU images
  and large models each consume several GB. First transcription downloads models.

## Start a CPU container

The default build pins CPU-only PyTorch, TorchAudio, and TorchVision wheels across
all dependency installs. Build-time verification rejects CUDA-enabled PyTorch or
installed NVIDIA runtime packages. No NVIDIA driver or GPU passthrough is required.

From the repository root:

```sh
docker compose up --build -d
```

Open http://127.0.0.1:8767. The port deliberately differs from the normal local API.
Use the Transcribe page or POST to `/api/transcribe` (interactive API documentation:
`/docs`). For CPU jobs, select **CPU**, and consider a smaller model. CPU separation
and large-model inference can be slow. `/api/ready` checks the HTTP server only, not
GPU availability or model readiness.

## Start an NVIDIA GPU container

Use the same pair of Compose files for every subsequent GPU command:

```sh
docker compose -f compose.yaml -f compose.gpu.yaml up --build -d
docker compose -f compose.yaml -f compose.gpu.yaml exec sstranscriber nvidia-smi
docker compose -f compose.yaml -f compose.gpu.yaml exec sstranscriber python -c "import torch, ctranslate2; print('torch:', torch.cuda.is_available()); print('CTranslate2 GPUs:', ctranslate2.get_cuda_device_count()); print(torch.ones(1, device='cuda').add_(1).cpu())"
```

Select **Automatic** or the exposed CUDA device in the UI. CUDA 12.8 libraries are
installed through the PyTorch wheels, not a separate CUDA base image. The image adds
their cuBLAS/cuDNN library paths for CTranslate2, which loads these independently.
The viet-lyrics fallback remains a separate process. This avoids sharing its model
runtime with the primary backend but does **not** eliminate combined VRAM usage.

Stop the CPU service with `docker compose down`; for GPU, add the two `-f` arguments.
Do not run CPU and GPU variants simultaneously against the same storage/port.

## Data and configuration

Default mounts are deliberately separate from local-development audio:

| Host / volume | Container | Purpose |
| --- | --- | --- |
| `./docker-data/input` | `/data/input` | Source songs and `lyrics/<song stem>.txt` |
| `./docker-data/output` | `/data/output` | Transcripts, archives, no-vocals songs |
| `./docker-data/temp` | `/data/temp` | Demucs stems and isolated upload workspaces |
| `./docker-data/.endpoint-config.json` | `/data/.endpoint-config.json` | Saved endpoint defaults |
| Compose `model-cache` volume | `/cache` | Hugging Face, Torch/Demucs, Numba caches |

Put source audio in `docker-data/input` (or use upload requests). Input is writable:
Web UI transcription embeds metadata into the source song. Back up important files.
Uploads remain isolated and are removed after their response/failure. Interrupted
containers can leave orphan upload workspaces; remove those only while stopped.
The existing 30-day generated-file cleanup still applies to ordinary output/temp
files, including persisted results. Request-owned upload directories are excluded.

The image runs as UID/GID 1000, not root. **On Linux**, create the bind-mount directory
before starting and ensure it is writable by the configured UID/GID. For a typical
current-user deployment:

```sh
mkdir -p docker-data
export SSTRANSCRIBER_UID=$(id -u)
export SSTRANSCRIBER_GID=$(id -g)
docker compose up --build -d
```

Run this as a normal non-root user. Existing directories and caches must retain
compatible ownership when switching UID/GID. Docker Desktop normally handles host
bind-mount permissions itself. Do not solve permissions by making all data world-writable.

Compose supports these shell variables (or entries in the root git-ignored `.env`):

| Variable | Default | Purpose |
| --- | --- | --- |
| `SSTRANSCRIBER_DOCKER_PORT` | `8767` | Loopback-only host port |
| `SSTRANSCRIBER_DATA_DIR` | `./docker-data` | Host data root |
| `SSTRANSCRIBER_UID` / `SSTRANSCRIBER_GID` | `1000` / `1000` | Runtime user IDs; rebuild when changed |
| `HF_TOKEN` | empty | Optional runtime Hugging Face token |

The container's internal port stays 8000; local `WEB_API_PORT` is not forwarded.
`HF_TOKEN` is passed only at runtime (visible to Docker administrators), not stored
in image layers. The allowlisted build context excludes `.env`, local settings,
audio, outputs, caches, virtual environments, and frontend dependencies.

`docker compose down` retains bind-mounted data and model volumes. Adding `--volumes`
removes the model-cache volume and forces future downloads; it does not remove the
host data directory. Avoid mounting the entire repository onto `/app`: that hides
the image's built UI and can expose local secrets or Windows-specific dependencies.

To reuse local files, copy the desired audio/settings into `docker-data`, or set
`SSTRANSCRIBER_DATA_DIR=.` explicitly. The latter shares local input/output/temp and
endpoint settings, including source-file edits and cleanup. **Never process the same
data concurrently from local and container servers**; their locks are independent.

## Local development is unchanged

Keep using your existing virtual environment and local dependencies:

```sh
python web_api.py --reload
npm --prefix webui run dev
```

Vite still reads `WEB_API_PORT` and proxies to the local API. Docker is not required
for this workflow; no local dependencies are upgraded by the Docker build.

For local UI development against the container instead, stop the local API and
set `WEB_API_PORT=8767` in the root `.env` before starting Vite. The existing CORS
configuration also permits the normal Vite localhost origins. Restore the previous
port when returning to local API development. Rebuild the Docker image after Python
or production frontend changes; the default deployment does not bind-mount source
or run reload mode.

To use a different local data root, export `SSTRANSCRIBER_WORK_DIR` into the server's
environment **before launch**. If it is absent, the API continues using repository
paths and the existing local endpoint settings. A `.env` entry alone is not a general
API environment loader; the API's `.env` parsing remains specific to `WEB_API_PORT`.

CLI jobs can run in a one-off container too (do not overlap with API jobs on the same
data). Supply a device to avoid interactive selection:

```sh
docker compose run --rm sstranscriber python process_audio_folder.py --device cpu --model small --no-vocal-separation
```

The embedding and metadata-clearing utilities also honor the mounted data root.
Do not run the project initialization/installation scripts inside the deployed image;
dependencies and the frontend are already built.

## Dependencies and validation

Included backends: faster-whisper, pho-whisper, viet-lyrics, plus Demucs. NeMo/Parakeet
and FunASR/SenseVoice remain available in local development but are **not installed**
in these baseline images; the UI still lists them. Selecting one in a baseline
container fails with a missing-package error. Extend the image with their dependencies
and revalidate constraints if needed; do not silently install them at startup.

The container-specific audio pins do not change the local installer. TorchAudio 2.8
is kept alongside PyTorch 2.8 because Demucs uses audio APIs changed in TorchAudio 2.9.
These are direct dependency pins/constraints, not a complete transitive lockfile or
digest-pinned supply-chain build. Review and update them together.

For local tests, install `requirements-test.txt` into the existing virtual environment,
then run `python -m unittest discover -s tests -v`. It includes both HTTP test clients
needed by the suite and current Starlette; it does not install audio runtimes.

Every image build runs `pip check`, native-library imports, GPU-variant cuBLAS/cuDNN
dynamic-linker discovery (before Torch imports), and actual WAV/MP3
encoding/decoding without downloading models. The `test` target additionally runs
the real CLI/API/ID3 tests with deterministic inference and separation substitutes:

```sh
docker build --target test --build-arg DEVICE=cpu -t sstranscriber:test .
docker build --target test --build-arg DEVICE=gpu -t sstranscriber:test-gpu .
```

Neither test target needs a GPU at build time. The GitHub Actions container workflow
runs the CPU test build automatically and supports a manual GPU image test build.
Runtime GPU access and actual model inference still require verification on the
deployment host. Test a small real song with separation enabled, then exercise the
viet-lyrics fallback and confirm results before deploying a larger library.

## Operational limits

- Run **one Uvicorn worker and one replica per data root**: queue locks and job history
  are process-local. Restarts do not resume jobs or preserve monitoring history.
- Tini handles PID 1/reaping. Uvicorn has a 20-second graceful request shutdown timeout;
  Compose allows 30 seconds before terminating the container. Stop when idle for
  predictable results; processing can take much longer than the shutdown window.
- Health metrics describe the container's view and may reflect host totals rather
  than exact container quotas. They are not a GPU inference benchmark.
- Loopback publishing is intentional. There is no authentication. Public deployment
  needs an authenticated TLS reverse proxy, upload limits, and long request timeouts.