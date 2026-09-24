# ssTranscriber

Batch-transcribes a folder of audio files and writes synced lyric transcripts to
`output/transcripts/`. A separate, manually-run step (`embed_lyrics.py`) then embeds those
transcripts back into each audio file as ID3 lyrics metadata — both **USLT** (unsynchronized,
plain-text block) and **SYLT** (synchronized, timestamped scrolling lyrics).

The default local pipeline uses [Demucs](https://github.com/facebookresearch/demucs) to isolate
vocals and [faster-whisper](https://github.com/SYSTRAN/faster-whisper) `large-v3` to transcribe
them. Tagging uses [Mutagen](https://mutagen.readthedocs.io/).

Optional `pho-whisper` and `viet-lyrics` backends are also available if you provide a compatible
runtime and model.

## Docker (optional)

CPU and NVIDIA GPU container deployments are available alongside the existing local
Python/Node workflow. See [Docker deployment and development](docker/README.md) for
startup commands, GPU prerequisites, persistent storage, permissions, and validation.
The default container uses a separate data directory and host port `8767`; local
development does not require Docker and retains its existing defaults.

On Windows, start Docker Desktop with Linux containers enabled. Run the commands
below in PowerShell from the repository root. Choose either CPU or GPU; do not run
both variants simultaneously against the same storage and port.

### CPU

The default build uses CPU-only PyTorch wheels and requires no NVIDIA driver or GPU
passthrough. Build and start it with:

```powershell
docker compose up --build -d
```

### NVIDIA GPU

Requires working Docker Desktop/WSL2 GPU passthrough and an NVIDIA driver supporting
CUDA 12.8.

```powershell
docker compose -f compose.yaml -f compose.gpu.yaml up --build -d
```

Open http://127.0.0.1:8767. For CPU jobs, select **CPU** in the UI; for GPU jobs,
select **Automatic** or the exposed CUDA device.

### View logs

```powershell
# CPU
docker compose logs -f

# GPU
docker compose -f compose.yaml -f compose.gpu.yaml logs -f
```

### Stop

```powershell
# CPU
docker compose down

# GPU
docker compose -f compose.yaml -f compose.gpu.yaml down
```

Stopping retains your data and cached models. Source audio goes in `docker-data/input`.

## Requirements

- Python 3.9+
- [FFmpeg](https://ffmpeg.org/) (installed automatically by `_install_audio_tools.py`)
- Optional: an NVIDIA GPU with a matching CUDA driver for faster transcription

Python 3.13 users must select `cu124`, `cu126`, `cu130`, or `cpu`; PyTorch does not publish Python 3.13
wheels on its older `cu118` and `cu121` indexes.

RTX 50-series GPUs require a PyTorch build with Blackwell support. Use `--cuda cu130` with an
NVIDIA driver that reports CUDA 13.0 or newer in `nvidia-smi`.

## Setup

```powershell
python .\_initialize_project.py
python .\_install_audio_tools.py --cuda cu130
```

### Hugging Face token (optional)

Some models are gated or subject to download rate limits. Copy `.env.example` to `.env` and set
your token:

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Both `_install_audio_tools.py` and `process_audio_folder.py` load `.env` automatically (no extra
dependency required) and export it as `HF_TOKEN`/`HUGGING_FACE_HUB_TOKEN` before contacting
Hugging Face. `.env` is git-ignored, so your token is never committed.

If you also want to validate PhoWhisper availability on Hugging Face during setup:

```powershell
python .\_install_audio_tools.py --cuda cu124 --with-phowhisper --pho-model vinai/PhoWhisper-large
```

Other optional backends can be installed the same way:

```powershell
python .\_install_audio_tools.py --cuda cu124 --with-parakeet --parakeet-model nvidia/parakeet-tdt-0.6b-v2
python .\_install_audio_tools.py --cuda cu124 --with-sensevoice --sensevoice-model FunAudioLLM/SenseVoiceSmall
python .\_install_audio_tools.py --cuda cu124 --with-viet-lyrics --viet-lyrics-model kelvinbksoh/whisper-large-v2-vietnamese-lyrics-transcription
```

`_install_audio_tools.py` installs `torch`, `torchaudio`, `torchvision`, `demucs`, `faster-whisper`,
`transformers`, `accelerate`, `librosa`, `soundfile`, `mutagen`, and a bundled FFmpeg build, then
prints whether CUDA is visible. An existing FFmpeg installation on `PATH` is retained; otherwise,
the bundled executable is downloaded and used automatically. `torch`/`torchaudio`/`torchvision`
are always installed together from the same CUDA index so they stay version-matched — a
mismatched `torchvision` build (e.g. left over from an earlier `torch` upgrade) causes
`transformers.pipeline(...)` to fail with
`RuntimeError: operator torchvision::nms does not exist` when loading the `pho-whisper` backend.
On Windows, the installer also provides CUDA 12.8 cuBLAS and cuDNN 9 runtime DLLs required by
Faster-Whisper's CTranslate2 backend, including on RTX 50-series GPUs using PyTorch `cu130`.
`accelerate`/`librosa`/`soundfile` are required for the `pho-whisper` backend, which loads
PhoWhisper checkpoints through `transformers.pipeline(...)` (there is no `pho-whisper` PyPI
package). Pick the `--cuda` value matching your NVIDIA driver
(`cu118`, `cu121`, `cu124`, `cu126`, `cu130`), or use `--cpu` builds via `--cuda cpu`. A full log is
written to `output/_install_audio_tools.log`.

Existing packages that already satisfy the requested versions are retained. Use
`--force-reinstall` only to repair a broken installation or replace an incorrect PyTorch build.

If `cuda_available` prints `False` while `nvidia-smi` shows your GPU, you have a CPU-only or
mismatched PyTorch build — rerun with a different `--cuda` value.

## Local Web UI

### Upload transcription API

Open **Endpoints** (`/endpoints`) to configure server-side defaults for upload requests.
It includes **Recognition** and **Advanced** defaults for the backend, model, device,
Demucs separation, backend profiles, and the Viet Lyrics fallback. Language and the
**Store Demucs stem as MP3** section are request-specific rather than endpoint controls. Settings are
saved in the git-ignored `.endpoint-config.json` and survive server restarts. They are
independent of browser-saved Transcribe configurations. Archiving and Lyrics assist
controls do not apply to uploads. Existing saved configurations remain loadable; retired
language, MP3 stem/bitrate, and no-vocals defaults are ignored and removed on the next save.

- `GET /api/endpoint-config`: read the current defaults.
- `PUT /api/endpoint-config`: save processing defaults as JSON. Unsupported fields,
  including the retired request-specific defaults, and unknown profile groups are rejected.
- `POST /api/transcribe`: upload `multipart/form-data` and wait for the finished download:

  | Field | Required | Meaning |
  | --- | --- | --- |
  | `file` | Yes | Song file in one of the supported audio formats. |
  | `lyrics` | No | Plain-text lyrics. Nonblank data automatically enables known lyrics; absent or blank data disables them. |
  | `lyrics_mode` | No | `align` (default), `prompt`, or `correct`. Ignored when no lyrics are supplied. |
  | `language` | No | Two-letter ISO 639-1 language code such as `vi` or `en` (case-insensitive). Omitted or empty uses auto-detection. |
  | `NoVocals` | No | `true` or `false` (default). Enables **Copy no-vocals song** for this request, automatically enabling vocal separation and MP3 stems at 320 kbps even if saved vocal separation is disabled. |
  | `VietLyricsFallback` | No | `true` or `false`. Enables or disables **Viet Lyrics fallback pass** for this request only; omitted uses the saved endpoint setting. Uses the saved fallback model/profile and runs only when the opening retry triggers. |

The response is the uploaded song with completed USLT/SYLT lyrics embedding and an
attachment filename (`application/octet-stream`). If `NoVocals=true`,
the response is `application/zip`, containing the embedded song and
`[NoVocals] <song stem>.mp3`, also with embedded lyrics. With `NoVocals=false` or omitted,
saved vocal separation is respected and any stems use WAV. Missing requested accompaniment or a transcription/embedding
failure returns an error, not an unprocessed song or partial download.

Requests take a snapshot of saved defaults and share the one-at-a-time processing lock
with Transcribe jobs. Allow a long HTTP timeout for queueing, model downloads, and
inference. Uploaded files, supplied lyrics, transcripts, and Demucs stems use private
temporary directories; existing input songs and previous results are never modified.
Temporary artifacts are removed after the response or processing failure. No separate
transcript file is included in the download; the transcript is embedded in the audio.
The embedding format is the existing pipeline's ID3 USLT/SYLT format, so player support
for these tags in non-MP3 containers can vary.

Interactive request schemas and testing are available at `/docs`. This API, including
configuration changes, is intended for trusted local use and has no authentication;
do not expose it publicly without authentication and upload/request limits.

Open **Endpoint jobs** (`/endpoint-jobs`) to monitor requests received through
`POST /api/transcribe`. The page refreshes automatically and shows running/queued uploads
with original filenames, processing settings, timestamps, and live logs. **Recent history**
also shows completed, failed, and cancelled requests. These jobs remain separate from
Transcribe jobs. Completion means processing and download preparation succeeded; it does
not confirm that the client finished downloading. Downloads still go to the original caller.

Select a queued or running upload and use **Cancel upload** in its details header to
cancel it. Queued uploads leave the queue immediately; running processing and its child
processes are terminated. Cancellation during download preparation waits for packaging
to finish before removing temporary files. The original upload caller receives HTTP
`409` instead of a download, and the job remains in recent history as cancelled.

- `GET /api/endpoint-jobs`: list upload jobs, newest first, without logs.
- `GET /api/endpoint-jobs/{job_id}`: fetch one upload job including captured logs.
- `DELETE /api/endpoint-jobs/{job_id}`: request cancellation (`202`); unknown jobs return
  `404`, and already-finished jobs return `409`. `cancel_requested` indicates a pending cancellation.
- Successful upload responses include an `X-Job-ID` header for correlation.

All active jobs and the latest 100 finished upload records are retained in memory.
Job history resets on API restart, while uploaded audio and lyrics are still cleaned up
after the response. Invalid uploads rejected before processing are not added to the list.

### Running the web interface

The optional local interface uses FastAPI for queued CLI subprocess jobs and React/Vite for the
browser client. Jobs run one at a time to avoid competing for GPU memory. The API process does not
import Torch or the ASR runtimes; each job still executes `process_audio_folder.py` in an isolated
Python subprocess.

Initialize its Python and Node dependencies and build the production bundle once:

```powershell
python .\_initialize_web_api.py
```

The default port is `8000`. To choose another port, pass it during initialization; the script
stores `WEB_API_PORT` in the git-ignored `.env` file and prints the resulting URL:

```powershell
python .\_initialize_web_api.py --port 8765
```

Start the combined production server and open the URL printed by the initializer:

```powershell
python .\web_api.py
```

For frontend development, run the API and Vite in separate terminals. Vite proxies `/api` to the
local FastAPI server:

```powershell
python .\web_api.py --reload
npm --prefix .\webui run dev
```

The API binds to localhost by default. Job state is held in memory and resets when the API server
restarts; transcript files and archived output remain on disk.

While the API server is running, it removes files under `output/` and `temp/` whose modification
time is more than 30 days old. Cleanup runs once during server startup and then every 24 hours.
Source files under `input/` are never included in this cleanup.

Under **Advanced**, expand **Backend profile** to edit the selected backend's runtime options as
JSON. Enabling **Viet Lyrics fallback pass** adds an independent **Fallback profile** for the
isolated fallback worker. Nested objects are merged with the displayed defaults, unknown top-level
option groups are rejected, and **Reset defaults** restores the profile for the selected backend.

The **Configurations** panel stores named transcription setups in the current browser, including
both backend profile editors. A saved setup can be marked as the startup default. Audio selection
is intentionally excluded so loading a configuration cannot silently target an old file.
The navigation sidebar and transcription form can be collapsed independently. Successfully queueing
a job collapses the form automatically so **Live activity** uses the available workspace and height.

The **Music** workspace groups and plays supported audio from `input/` and generated tracks from
`output/songs/`. For MP3 files containing an
ID3 `SYLT` frame with absolute-millisecond timestamps, the lyric timeline follows playback and each
line can be selected to seek to its timestamp. Files without embedded `SYLT` lyrics still play.
Transcription jobs started from the Web UI automatically embed their final transcript as `USLT` and
`SYLT` metadata in each processed source file.

The **Health** workspace at `/health` displays live metrics for the machine running the API:
total and per-logical-processor CPU usage, RAM usage, used/free capacity for accessible mounted
drives, aggregate network receive/send rates and counters, and NVIDIA GPU utilization, VRAM,
and temperature. It refreshes approximately every two seconds while visible and marks retained
readings as stale when a request fails. GPU monitoring requires `nvidia-smi`; unsupported GPUs
or unavailable driver readings are shown as unavailable. The API uses `psutil` for host metrics
without importing Torch. Existing installations should rerun `python .\_initialize_web_api.py`
to install the added dependency and rebuild the frontend, then restart the API.

## Usage

1. Drop audio files into `input/` (subfolders are scanned recursively).
2. Optionally place known lyrics in `input/lyrics/` as UTF-8 text using the same filename stem as
  the audio, for example `input/song.mp3` and `input/lyrics/song.txt`. Faster-Whisper,
  PhoWhisper, and Viet Lyrics can use matching text in `prompt`, `align`, or `correct` mode.
  `align` maps authoritative lyric lines onto ASR-derived timing; it is not a separate acoustic
  forced-alignment engine. Missing or empty files are ignored, Parakeet/SenseVoice continue without
  prompt biasing, and `--no-lyric-prompt` disables all lyric modes.
3. Run:

```powershell
python .\process_audio_folder.py
```

With no `--device` flag the script lists the detected devices and prompts you to pick one:

```
Available processing devices:
  0: CPU
  1: CUDA:0 - NVIDIA GeForce RTX 4090 Laptop GPU
Select device [1]:
```

Pressing Enter accepts the default (the last GPU, or CPU if none). Pass `--device` to skip the
prompt in scripts or scheduled runs; non-interactive runs fall back to `auto`.

After device/CUDA selection, the script prints available models for the selected backend and marks
the currently selected model.

Example output:

```
[2026-08-16 23:40:00] Using device: cuda:0
Available models for backend "faster-whisper":
  - tiny
  - base
  - small
  - medium
  - large-v2
  - large-v3 (selected)
  - large-v3-turbo
  - distil-large-v2
  - distil-large-v3
```

Force a language when auto-detection is unreliable:

```powershell
python .\process_audio_folder.py --device cuda:0 --language vi
```

The default `large-v3` model prioritizes song-lyric accuracy. Its first run downloads the
model, while the first vocal-separation run downloads the Demucs model. Either download can
take several minutes. For faster processing with somewhat lower accuracy, use:

```powershell
python .\process_audio_folder.py --device cuda:0 --model large-v3
```

To use a other backend models:

```powershell
python .\process_audio_folder.py --device cuda:0 --backend pho-whisper --model vinai/PhoWhisper-large --language vi
python .\process_audio_folder.py --device cuda:0 --backend parakeet --model nvidia/parakeet-tdt-0.6b-v2
python .\process_audio_folder.py --device cuda:0 --backend sensevoice --model FunAudioLLM/SenseVoiceSmall
python .\process_audio_folder.py --device cuda:0 --backend viet-lyrics --model kelvinbksoh/whisper-large-v2-vietnamese-lyrics-transcription --language vi
```

### Options

| Flag | Values | Default | Description |
| --- | --- | --- | --- |
| `--device` | `auto`, `cpu`, `cuda:N` | prompt (`auto` if non-interactive) | Processing device. Detected devices are listed before work starts. |
| `--language` | ISO 639-1 code (`vi`, `zh`, `ja`, …) | auto-detect | Forces the transcription language. |
| `--backend` | `faster-whisper`, `pho-whisper`, `parakeet`, `sensevoice`, `viet-lyrics` | `faster-whisper` | Selects the transcription backend. |
| `--model` | backend-specific model name | `large-v3` | Selects the model for the chosen backend. |
| `--no-vocal-separation` | flag | off | Bypasses Demucs and transcribes the original audio. |
| `--demucs-mp3` | flag | off | Writes the Demucs-separated vocals stem as MP3 instead of WAV. |
| `--demucs-mp3-bitrate` | kbps (int) | `320` | MP3 bitrate used when `--demucs-mp3` is set. |
| `--copy-no-vocals` | flag | off | With `--demucs-mp3`, copies the accompaniment to `output/songs/[NoVocals] <name>.mp3` and embeds the same USLT/SYLT lyrics as the source. |
| `--file` | relative path under `input/` | all supported files | Processes a single selected audio file. |
| `--keep-promotions` | flag | off | Keeps known promotional phrases instead of stripping them out. |
| `--no-lyric-prompt` | flag | off | Ignores matching files under `input/lyrics/` and overrides `--lyrics-mode`. |
| `--lyrics-mode` | `prompt`, `align`, `correct` | `prompt` | `prompt` biases decoding; `align` maps authoritative lyric lines to ASR timing; `correct` replaces recognized text while preserving ASR segment timing. |
| `--embed-lyrics` | flag | off | Embeds the final transcript into each processed source file as `USLT` and synchronized `SYLT` metadata. Web UI jobs enable this automatically. |
| `--opening-threshold` | seconds (float) | `1.0` | If the first usable vocal segment starts later than this, retries on the original (non-separated) audio to recover a possibly clipped opening. |
| `--fallback-viet-lyrics` | flag | off | When the `--opening-threshold` retry triggers, runs a 3rd pass with the `viet-lyrics` backend (on the separated vocals) after the primary `--backend`/`--model` retry on the original audio. |
| `--fallback-viet-lyrics-model` | Hugging Face model ID | `kelvinbksoh/whisper-large-v2-vietnamese-lyrics-transcription` | Model used for `--fallback-viet-lyrics` retries. |
| `--backend-options-json` | JSON object | `{}` | Overrides option groups for the selected primary backend. Nested objects merge with backend defaults. |
| `--fallback-viet-lyrics-options-json` | JSON object | `{}` | Overrides option groups passed to the isolated Viet Lyrics fallback worker. |

Common built-in model lists:

- `faster-whisper`: `tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3`, `large-v3-turbo`, `distil-large-v2`, `distil-large-v3`
- `pho-whisper`: `vinai/PhoWhisper-small`, `vinai/PhoWhisper-base`, `vinai/PhoWhisper-large`
- `parakeet`: `nvidia/parakeet-tdt-0.6b-v2`, `nvidia/parakeet-tdt-1.1b`, `nvidia/canary-1b` (requires `_install_audio_tools.py --with-parakeet`)
- `sensevoice`: `FunAudioLLM/SenseVoiceSmall` (requires `_install_audio_tools.py --with-sensevoice`; only supports `zh`, `yue`, `en`, `ja`, `ko` — no Vietnamese)
- `viet-lyrics`: `kelvinbksoh/whisper-small-vietnamese-lyrics-transcription`, `kelvinbksoh/whisper-medium-vietnamese-lyrics-transcription`, `kelvinbksoh/whisper-large-v2-vietnamese-lyrics-transcription` (Whisper fine-tuned specifically on Vietnamese song lyrics; loads through `transformers.pipeline(...)` like `pho-whisper`, but as an independent backend)

Supported extensions: `.mp3`, `.wav`, `.flac`, `.m4a`, `.aac`, `.ogg`, `.opus`, `.wma`.

### Transcript output behavior

- `output/transcripts/` is cleared at the start of each run so stale files do not remain.
- Pass `--save-previous-results` to preserve a non-empty transcript folder before processing. It is
  renamed to the next available numbered folder (`transcripts_01`, `transcripts_02`, and so on),
  and a new empty `output/transcripts/` folder is created for the current run. A non-empty
  `temp/htdemucs/` folder is preserved the same way as `htdemucs_01`, `htdemucs_02`, and so on.
- Each transcription pass writes its backend, model, device, language, and complete backend options
  to `__1initial_model_options.json`, `__2original_model_options.json`, or
  `__3fallback_model_options.json`. Retry manifests are created only when those passes run.
- Transcript files are written as LRC-style synced lyrics (`[mm:ss.xx]text` per line), using the
  same `(text, start_ms)` entries embedded as the mp3's `SYLT` frame via `mutagen.id3.SYLT`, so the
  `.txt` file always matches what is written into the audio file and can be used to re-embed SYLT
  lyrics later if needed.
- If Demucs misses the opening and the script falls back to the original mix, a sibling file with
  the suffix `_1initial.txt` preserves the initial separated-vocals result and `_2original.txt`
  contains the original-mix retry. When `--fallback-viet-lyrics` is also used, a third sibling
  `_3fallback.txt` is created with the viet-lyrics pass's result, so all passes stay individually
  inspectable instead of consecutive retries overwriting earlier transcripts.
- The fallback triggers when the first usable vocal segment starts later than `--opening-threshold` seconds (default `1.0`), forcing the script to expect transcription to begin almost immediately (0:01) instead of tolerating a longer gap. Raise this value if Demucs-separated vocals legitimately start later in your songs.
- When the fallback triggers, the script always retries first with the primary `--backend`/`--model`
  on the original (non-separated) audio. Pass `--fallback-viet-lyrics` to additionally run a third
  pass afterward with the `viet-lyrics` backend (`--fallback-viet-lyrics-model`, default
  `kelvinbksoh/whisper-large-v2-vietnamese-lyrics-transcription`) on the Demucs-separated vocals
  stem — so a triggered fallback with `--fallback-viet-lyrics` means 3 transcription passes total
  (initial separated-vocals pass, original-audio retry, viet-lyrics fallback). The main `.txt` file
  and each sibling file are written to disk immediately after their respective pass, so a crash in
  a later pass never loses an earlier one's result.
  The viet-lyrics fallback runs in its own isolated worker process (loaded lazily once and reused
  for the rest of the run) to avoid a cuDNN conflict with the primary faster-whisper model.
- Demucs stems are written to `temp/htdemucs/...` so the temporary separated vocals can be inspected if needed.
- Pass `--demucs-mp3` to have Demucs write the separated vocals stem as MP3 (default `320` kbps, adjustable with `--demucs-mp3-bitrate`) instead of WAV, which uses less disk space at the cost of a lossy re-encode before transcription.

## Output

A per-file summary is printed and written to `output/processing_log.txt`:

```
Processed root: D:\workspace\ssTranscriber\input
Files found: 18
Successfully updated: 18
Failed: 0
Skipped: 0

Per file:
- Millionaire.mp3 — Success (transcript written, lang=eng)
```

Both `input/` and `output/` are git-ignored and created automatically on startup.

## Embed lyrics into the audio files

By default, `process_audio_folder.py` only writes transcript `.txt` files under
`output/transcripts/`. Web UI jobs pass `--embed-lyrics` and tag processed source files
automatically. For a default CLI run, use `embed_lyrics.py` afterward (manually, once you're happy
with the transcripts) to embed `USLT`/`SYLT` tags:

```powershell
python .\embed_lyrics.py
```

For each audio file under `input/`, it looks for a transcript with the same relative
path/filename (stem) under `output/transcripts/`, parses its `[lang:xx]` header and
`[mm:ss.xx]text` synced lines (written by `process_audio_folder.py`), and embeds them as
`USLT`/`SYLT` — tagging files **in place** and replacing any existing `USLT`/`SYLT` frames. The
`_1initial.txt`/`_2original.txt`/`_3fallback.txt` pass-specific sibling files never match an audio
filename, so they're skipped automatically. Options:

| Flag | Values | Default | Description |
| --- | --- | --- | --- |
| `--language` | ISO 639-1 code (`vi`, `zh`, `ja`, …) | the language recorded in the transcript, or `und` | Overrides the tagged language. |
| `--file` | relative path under `input/` | all supported files | Only embed one selected audio file. |

A per-file summary is printed and written to `output/embed_lyrics_log.txt`.

## Remove embedded metadata

Use `clear_embedded_lyrics.py` to remove all embedded audio metadata, not just lyrics.
This includes ID3 lyric frames and other common tags such as title, artist, album,
and cover art while preserving the actual audio stream.

Preview the affected files without making changes:

```powershell
python .\clear_embedded_lyrics.py --dry-run
```

Remove the metadata after reviewing the preview:

```powershell
python .\clear_embedded_lyrics.py
```

The script scans `input/` recursively and writes a timestamped summary to
`output/clear_metadata_log.txt`.

## Unicode handling

Vietnamese, Chinese, and other non-Latin scripts are fully supported:

- Lyric frames are written as **UTF-16**, the only Unicode encoding ID3v2.3 permits.
- The frame language field is set from the detected language (`vi` → `vie`, `zh` → `chi`),
  falling back to `und`.
- `SYLT` uses `format=2` (absolute milliseconds) so players scroll in sync.
- Console output is forced to UTF-8 so non-ASCII filenames don't crash the run on Windows.

## Copilot prompt

[`.github/prompts/transcribe-sound-files.prompt.md`](.github/prompts/transcribe-sound-files.prompt.md)
describes the same workflow for VS Code Copilot in agent mode.

## Troubleshooting

**`Failed to load audio: ... No such file or directory`**
The file list is snapshotted before processing begins. Moving, renaming, or deleting files
mid-run invalidates the remaining paths. Leave `input/` untouched while the script runs.

**`No transcript produced`**
The model found no vocals or speech — common for instrumentals. The file is left unmodified.

**`Command failed: ... pip install ... VinAIResearch/PhoWhisper.git`**
PhoWhisper setup now validates a Hugging Face model ID (for example
`vinai/PhoWhisper-large`) instead of installing from GitHub. If this fails, check internet
access and verify the model ID passed via `--pho-model`.

**`sensevoice backend does not support language "vi"`**
SenseVoice was only trained on Chinese, Cantonese, English, Japanese, and Korean — it has no
Vietnamese support at all. Requesting `--language vi` with `--backend sensevoice` now fails
fast with this error instead of silently producing hallucinated Chinese/Japanese/Korean text.
Use `--backend faster-whisper` (default), `--backend pho-whisper`, or `--backend viet-lyrics`
for Vietnamese lyrics.

**`Could not load symbol cudnnGetLibConfig. Error code 127`**
On Windows, `faster-whisper` (CTranslate2) and `transformers`/`torch` bundle different,
ABI-incompatible cuDNN builds. Loading both in the same process (e.g. the primary model plus
an in-process fallback model) crashes with this error. `--fallback-viet-lyrics` retries run in
an isolated worker process (`backends/viet_lyrics_worker.py`) specifically to avoid this; if you see this
error elsewhere, avoid loading a transformers-based backend in the same run/process as
`faster-whisper`.
Use `--backend faster-whisper` (default), `--backend pho-whisper`, or `--backend viet-lyrics`
for Vietnamese lyrics.
