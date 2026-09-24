import argparse
import ast
import asyncio
import csv
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
import zipfile
from collections import deque
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import psutil
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from mutagen.id3 import ID3, ID3NoHeaderError
from pydantic import BaseModel, Field, field_validator, model_validator
from starlette.background import BackgroundTask

REPO_ROOT = Path(__file__).resolve().parent
WORK_DIR = Path(os.environ.get('SSTRANSCRIBER_WORK_DIR', str(REPO_ROOT))).resolve()
INPUT_DIR = (WORK_DIR / 'input').resolve()
OUTPUT_DIR = (WORK_DIR / 'output').resolve()
SONGS_DIR = (OUTPUT_DIR / 'songs').resolve()
TEMP_DIR = (WORK_DIR / 'temp').resolve()
WEB_DIST = REPO_ROOT / 'webui' / 'dist'
ENDPOINT_CONFIG_PATH = WORK_DIR / '.endpoint-config.json'
CLEANUP_MAX_AGE_DAYS = 30
CLEANUP_INTERVAL_SECONDS = 24 * 60 * 60
SUPPORTED_AUDIO = {'.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg', '.opus', '.wma'}
BACKEND_FILES = {
    'faster-whisper': 'faster_whisper_backend.py',
    'pho-whisper': 'pho_whisper_backend.py',
    'parakeet': 'parakeet_backend.py',
    'sensevoice': 'sensevoice_backend.py',
    'viet-lyrics': 'viet_lyrics_backend.py',
}
DEFAULT_MODELS = {
    'faster-whisper': 'large-v3',
    'pho-whisper': 'vinai/PhoWhisper-large',
    'parakeet': 'nvidia/parakeet-tdt-0.6b-v2',
    'sensevoice': 'FunAudioLLM/SenseVoiceSmall',
    'viet-lyrics': 'kelvinbksoh/whisper-large-v2-vietnamese-lyrics-transcription',
}


def port_number(value: str):
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError('port must be an integer') from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError('port must be between 1 and 65535')
    return port


def configured_port():
    value = os.environ.get('WEB_API_PORT')
    env_path = REPO_ROOT / '.env'
    if value is None and env_path.is_file():
        for raw_line in env_path.read_text(encoding='utf-8').splitlines():
            line = raw_line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, _, candidate = line.partition('=')
                if key.strip() == 'WEB_API_PORT':
                    value = candidate.strip().strip('"').strip("'")
                    break
    return port_number(value or '8000')


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def json_value(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_value(item) for item in value]
    return repr(value)


def backend_metadata():
    metadata = {}
    for backend, filename in BACKEND_FILES.items():
        tree = ast.parse((REPO_ROOT / 'backends' / filename).read_text(encoding='utf-8'))
        values = {}
        for node in tree.body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value_node = node.value
            for target in targets:
                if not isinstance(target, ast.Name) or not target.id.isupper():
                    continue
                try:
                    values[target.id] = json_value(ast.literal_eval(value_node))
                except (ValueError, TypeError):
                    pass
        metadata[backend] = {
            'models': values.pop('MODELS', []),
            'default_model': DEFAULT_MODELS[backend],
            'options': values,
        }
    return metadata


def validate_profile_options(backend: str, options: dict, label: str):
    available = set(backend_metadata()[backend]['options'])
    unknown = sorted(set(options) - available)
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=f'Unknown {label} option(s): {", ".join(unknown)}',
        )


def available_devices():
    devices = [{'value': 'auto', 'label': 'Automatic'}, {'value': 'cpu', 'label': 'CPU'}]
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                index, separator, name = line.partition(',')
                if separator:
                    devices.append({'value': f'cuda:{index.strip()}', 'label': name.strip()})
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return devices


def relative_files(root: Path, extensions=None):
    if not root.is_dir():
        return []
    return [
        path.relative_to(root).as_posix()
        for path in sorted(root.rglob('*'), key=lambda item: str(item).casefold())
        if path.is_file() and (extensions is None or path.suffix.lower() in extensions)
    ]


def input_audio_path(relative_path: str):
    path = (INPUT_DIR / relative_path).resolve()
    if INPUT_DIR not in path.parents or not path.is_file() or path.suffix.lower() not in SUPPORTED_AUDIO:
        raise HTTPException(status_code=404, detail='Audio file not found')
    return path


def music_audio_path(library: str, relative_path: str):
    root = {'input': INPUT_DIR, 'songs': SONGS_DIR}.get(library)
    if root is None:
        raise HTTPException(status_code=404, detail='Music library not found')
    path = (root / relative_path).resolve()
    if root not in path.parents or not path.is_file() or path.suffix.lower() not in SUPPORTED_AUDIO:
        raise HTTPException(status_code=404, detail='Audio file not found')
    return path


class SharedProcessingSettings(BaseModel):
    backend: str = 'faster-whisper'
    model: str = 'large-v3'
    device: str = 'auto'
    multilingual: bool = False
    vocal_separation: bool = True
    keep_promotions: bool = False
    opening_threshold: float = Field(default=1.0, ge=0, le=300)
    fallback_viet_lyrics: bool = False
    fallback_viet_lyrics_model: str = DEFAULT_MODELS['viet-lyrics']
    backend_options: dict[str, Any] = Field(default_factory=dict)
    fallback_viet_lyrics_options: dict[str, Any] = Field(default_factory=dict)

    @field_validator('backend')
    @classmethod
    def validate_backend(cls, value):
        if value not in BACKEND_FILES:
            raise ValueError('Unsupported backend')
        return value

    @model_validator(mode='after')
    def validate_multilingual_backend(self):
        if self.multilingual and self.backend != 'faster-whisper':
            raise ValueError('multilingual is only supported by faster-whisper')
        return self


class ProcessingSettings(SharedProcessingSettings):
    language: str | None = None
    demucs_mp3: bool = False
    demucs_mp3_bitrate: int = Field(default=320, ge=64, le=512)
    copy_no_vocals: bool = False

    @model_validator(mode='after')
    def validate_no_vocals_copy(self):
        if self.copy_no_vocals and (not self.vocal_separation or not self.demucs_mp3):
            raise ValueError('copy_no_vocals requires vocal_separation and demucs_mp3')
        return self


class EndpointSettings(SharedProcessingSettings):
    model_config = {'extra': 'forbid'}
    opening_threshold: float = Field(default=20.0, ge=0, le=300)


class JobRequest(ProcessingSettings):
    file: str | None = None
    use_lyrics: bool = True
    lyrics_mode: Literal['prompt', 'align', 'correct'] = 'prompt'
    save_previous_results: bool = False

    @field_validator('file')
    @classmethod
    def validate_file(cls, value):
        if value is None:
            return value
        candidate = (INPUT_DIR / Path(value)).resolve()
        if INPUT_DIR not in candidate.parents or not candidate.is_file() or candidate.suffix.lower() not in SUPPORTED_AUDIO:
            raise ValueError('File must be a supported audio file under input/')
        return candidate.relative_to(INPUT_DIR).as_posix()


class Job:
    def __init__(self, request: JobRequest, work_dir: Path | None = None, upload_name: str | None = None):
        self.id = uuid.uuid4().hex[:12]
        self.request = request
        self.status = 'queued'
        self.created_at = utc_now()
        self.started_at = None
        self.finished_at = None
        self.return_code = None
        self.logs = deque(maxlen=4000)
        self.process = None
        self.processing_task = None
        self.cancel_requested = False
        self.work_dir = work_dir
        self.upload_name = upload_name
        self.filename = request.file

    def public(self, include_logs=False):
        result = {
            'id': self.id,
            'filename': self.filename,
            'status': self.status,
            'created_at': self.created_at,
            'started_at': self.started_at,
            'finished_at': self.finished_at,
            'return_code': self.return_code,
            'cancel_requested': self.cancel_requested,
            'request': self.request.model_dump(),
            'log_count': len(self.logs),
        }
        if include_logs:
            result['logs'] = list(self.logs)
        return result


jobs: dict[str, Job] = {}
endpoint_jobs: dict[str, Job] = {}
ENDPOINT_JOB_HISTORY_LIMIT = 100
job_lock = asyncio.Lock()


def prune_endpoint_jobs():
    """Keep all active uploads and only the most recent finished upload records."""
    finished = [job.id for job in endpoint_jobs.values()
                if job.status in {'completed', 'failed', 'cancelled'}]
    for job_id in finished[:-ENDPOINT_JOB_HISTORY_LIMIT]:
        endpoint_jobs.pop(job_id, None)


def build_command(request: JobRequest):
    command = [
        sys.executable,
        '-u',
        str(REPO_ROOT / 'process_audio_folder.py'),
        '--backend',
        request.backend,
        '--model',
        request.model,
        '--device',
        request.device,
        '--multilingual' if request.multilingual else '--no-multilingual',
        '--lyrics-mode',
        request.lyrics_mode,
        '--embed-lyrics',
        '--demucs-mp3-bitrate',
        str(request.demucs_mp3_bitrate),
        '--opening-threshold',
        str(request.opening_threshold),
        '--fallback-viet-lyrics-model',
        request.fallback_viet_lyrics_model,
        '--backend-options-json',
        json.dumps(request.backend_options, ensure_ascii=False, separators=(',', ':')),
        '--fallback-viet-lyrics-options-json',
        json.dumps(request.fallback_viet_lyrics_options, ensure_ascii=False, separators=(',', ':')),
    ]
    if request.file:
        command.extend(['--file', request.file])
    if request.language:
        command.extend(['--language', request.language])
    for enabled, flag in (
        (not request.vocal_separation, '--no-vocal-separation'),
        (request.demucs_mp3, '--demucs-mp3'),
        (request.copy_no_vocals, '--copy-no-vocals'),
        (request.keep_promotions, '--keep-promotions'),
        (not request.use_lyrics, '--no-lyric-prompt'),
        (request.save_previous_results, '--save-previous-results'),
        (request.fallback_viet_lyrics, '--fallback-viet-lyrics'),
    ):
        if enabled:
            command.append(flag)
    return command


async def run_job(job: Job):
    async with job_lock:
        if job.cancel_requested:
            job.status = 'cancelled'
            job.finished_at = utc_now()
            return
        job.status = 'running'
        job.started_at = utc_now()
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
        kwargs = {'creationflags': creationflags} if os.name == 'nt' else {'start_new_session': True}
        try:
            command = build_command(job.request)
            kwargs['env'] = {
                **os.environ,
                'SSTRANSCRIBER_WORK_DIR': str(job.work_dir if job.work_dir is not None else WORK_DIR),
            }
            if job.work_dir is not None:
                command.extend(['--file', job.upload_name])
            job.process = await asyncio.create_subprocess_exec(
                *command,
                cwd=REPO_ROOT,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                **kwargs,
            )
            while line := await job.process.stdout.readline():
                job.logs.append(line.decode('utf-8', errors='replace').rstrip())
            job.return_code = await job.process.wait()
            if job.cancel_requested:
                job.status = 'cancelled'
            else:
                # Uploads remain active until the response artifacts are verified/packaged.
                job.status = ('running' if job.work_dir is not None else 'completed') if job.return_code == 0 else 'failed'
        except asyncio.CancelledError:
            if job.process is not None and job.process.returncode is None:
                if os.name == 'nt':
                    await asyncio.to_thread(subprocess.run, ['taskkill', '/PID', str(job.process.pid), '/T', '/F'], capture_output=True, check=False)
                else:
                    with suppress(ProcessLookupError):
                        os.killpg(job.process.pid, signal.SIGKILL)
                await job.process.wait()
            job.status = 'cancelled'
            raise
        except Exception as exc:
            job.logs.append(f'Web API error: {exc}')
            job.status = 'failed'
        finally:
            if job.status in {'completed', 'failed', 'cancelled'}:
                job.finished_at = utc_now()
            job.process = None


def cleanup_old_generated_files(now: float | None = None):
    cutoff = (time.time() if now is None else now) - CLEANUP_MAX_AGE_DAYS * 24 * 60 * 60
    deleted = 0
    failures = []
    for root in (OUTPUT_DIR, TEMP_DIR):
        if not root.is_dir():
            continue
        for path in root.rglob('*'):
            try:
                # Upload workspaces own their lifecycle. Never age-delete files in
                # a live request (an unusually long upload can span a cleanup run).
                if root == TEMP_DIR and path.relative_to(root).parts[0].startswith('ss-transcriber-api-'):
                    continue
                if path.is_file() and path.stat().st_mtime < cutoff:
                    path.unlink()
                    deleted += 1
            except OSError as exc:
                failures.append(f'{path}: {exc}')
    return deleted, failures


async def scheduled_cleanup():
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
        deleted, failures = await asyncio.to_thread(cleanup_old_generated_files)
        print(f'Generated-file cleanup: deleted {deleted}, failed {len(failures)}', flush=True)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    for directory in (INPUT_DIR, SONGS_DIR, TEMP_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    deleted, failures = await asyncio.to_thread(cleanup_old_generated_files)
    print(f'Generated-file cleanup: deleted {deleted}, failed {len(failures)}', flush=True)
    cleanup_task = asyncio.create_task(scheduled_cleanup())
    try:
        yield
    finally:
        cleanup_task.cancel()
        with suppress(asyncio.CancelledError):
            await cleanup_task


app = FastAPI(title='ssTranscriber API', version='1.0.0', lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:5173', 'http://127.0.0.1:5173'],
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.get('/api/ready')
def readiness():
    """Cheap container liveness check; does not load models or query the GPU."""
    return {'status': 'ok'}


def gpu_health():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=3,
            check=False,
        )
        if result.returncode != 0:
            return {'devices': [], 'unavailable_reason': 'NVIDIA GPU metrics are unavailable.'}
    except (OSError, subprocess.TimeoutExpired):
        return {'devices': [], 'unavailable_reason': 'NVIDIA monitoring is unavailable. A supported GPU and nvidia-smi are required.'}

    def number(value):
        try:
            parsed = float(value)
            return parsed if math.isfinite(parsed) and parsed >= 0 else None
        except ValueError:
            return None

    devices = []
    for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
        if len(row) != 6:
            continue
        index, name, utilization, used, total, temperature = row
        used_mib, total_mib = number(used), number(total)
        devices.append({
            'id': index.strip(), 'name': name.strip(), 'percent': number(utilization),
            'memory_used': used_mib * 1024 ** 2 if used_mib is not None else None,
            'memory_total': total_mib * 1024 ** 2 if total_mib is not None else None,
            'temperature': number(temperature),
        })
    return {'devices': devices, 'unavailable_reason': None if devices else 'No NVIDIA GPUs detected.'}


@app.get('/api/health')
def get_health():
    network_before = psutil.net_io_counters()
    started = time.monotonic()
    cores = psutil.cpu_percent(interval=0.25, percpu=True)
    network_after = psutil.net_io_counters()
    elapsed = time.monotonic() - started
    memory = psutil.virtual_memory()
    storage = []
    for partition in psutil.disk_partitions(all=False):
        if 'cdrom' in partition.opts or not partition.fstype:
            continue
        try:
            usage = psutil.disk_usage(partition.mountpoint)
        except OSError:
            continue
        storage.append({
            'path': partition.mountpoint, 'device': partition.device,
            'total': usage.total, 'used': usage.used, 'free': usage.free, 'percent': usage.percent,
        })
    network = None
    if network_before is not None and network_after is not None:
        network = {
            'received_per_second': max(0, network_after.bytes_recv - network_before.bytes_recv) / elapsed,
            'sent_per_second': max(0, network_after.bytes_sent - network_before.bytes_sent) / elapsed,
            'received_total': network_after.bytes_recv, 'sent_total': network_after.bytes_sent,
        }
    return {
        'sampled_at': utc_now(),
        'cpu': {'percent': sum(cores) / len(cores) if cores else 0, 'cores': cores},
        'memory': {'total': memory.total, 'used': memory.total - memory.available,
                   'available': memory.available, 'percent': memory.percent},
        'storage': storage, 'network': network, 'gpu': gpu_health(),
    }


@app.get('/api/config')
def get_config():
    return {
        'backends': backend_metadata(),
        'devices': available_devices(),
        'lyrics_modes': ['prompt', 'align', 'correct'],
    }


def validate_processing_profiles(settings: SharedProcessingSettings):
    validate_profile_options(settings.backend, settings.backend_options, 'backend')
    validate_profile_options('viet-lyrics', settings.fallback_viet_lyrics_options, 'fallback')


@app.get('/api/endpoint-config', response_model=EndpointSettings)
def get_endpoint_config():
    if not ENDPOINT_CONFIG_PATH.exists():
        return EndpointSettings()
    try:
        data = json.loads(ENDPOINT_CONFIG_PATH.read_text(encoding='utf-8'))
        if isinstance(data, dict):
            # Discard retired defaults from configurations saved before per-upload options.
            for field in ('language', 'demucs_mp3', 'demucs_mp3_bitrate', 'copy_no_vocals'):
                data.pop(field, None)
        settings = EndpointSettings.model_validate(data)
        validate_processing_profiles(settings)
        return settings
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=500, detail='Could not load endpoint configuration') from exc


@app.put('/api/endpoint-config', response_model=EndpointSettings)
def save_endpoint_config(settings: EndpointSettings):
    validate_processing_profiles(settings)
    temporary = ENDPOINT_CONFIG_PATH.with_suffix(f'.{uuid.uuid4().hex}.tmp')
    try:
        ENDPOINT_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(settings.model_dump_json(indent=2) + '\n', encoding='utf-8')
        temporary.replace(ENDPOINT_CONFIG_PATH)
    except OSError as exc:
        raise HTTPException(status_code=500, detail='Could not save endpoint configuration') from exc
    finally:
        temporary.unlink(missing_ok=True)
    return settings


def prepare_upload(work_dir: Path, file, suffix: str, lyrics: str | None):
    input_dir = work_dir / 'input'
    input_dir.mkdir()
    # Fixed internal names avoid traversal, Windows reserved names, and Demucs exclusions.
    song = input_dir / f'song{suffix}'
    with song.open('wb') as target:
        shutil.copyfileobj(file, target, length=1024 * 1024)
    if song.stat().st_size == 0:
        raise HTTPException(status_code=422, detail='Song file is empty')
    if lyrics:
        lyrics_dir = input_dir / 'lyrics'
        lyrics_dir.mkdir()
        (lyrics_dir / 'song.txt').write_text(lyrics, encoding='utf-8')
    return song


def upload_result(work_dir: Path, song: Path, filename: str, copy_no_vocals: bool):
    manifest = work_dir / 'output' / 'processing_results.json'
    if not manifest.is_file():
        raise HTTPException(status_code=500, detail='Processing did not produce a completion record')
    results = json.loads(manifest.read_text(encoding='utf-8'))
    if len(results) != 1 or not results[0]['status'].startswith('Success'):
        raise HTTPException(status_code=500, detail='Song transcription or embedding failed')
    if not copy_no_vocals:
        return song, filename, 'application/octet-stream'
    accompaniment = work_dir / 'output' / 'songs' / '[NoVocals] song.mp3'
    if not accompaniment.is_file():
        raise HTTPException(status_code=500, detail='The requested no-vocals song could not be produced')
    archive = work_dir / 'result.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_STORED) as bundle:
        bundle.write(song, filename)
        bundle.write(accompaniment, f'[NoVocals] {Path(filename).stem}.mp3')
    return archive, f'{Path(filename).stem}.zip', 'application/zip'


@app.post('/api/transcribe', response_class=FileResponse, responses={
    200: {'description': 'Embedded song, or a ZIP containing the song and no-vocals song.',
          'content': {'application/octet-stream': {}, 'application/zip': {}}},
})
async def transcribe_upload(
    file: UploadFile = File(..., description='Song file to transcribe and embed.'),
    lyrics: str | None = Form(None, description='Optional plain-text lyrics; enables known lyrics automatically.'),
    lyrics_mode: Literal['prompt', 'align', 'correct'] = Form('align'),
    language: str | None = Form(
        None, pattern=r'^[A-Za-z]{2}$',
        description='Optional ISO 639-1 language code (e.g. vi, en). Omitted or empty auto-detects the language.',
    ),
    no_vocals: bool = Form(
        False, alias='NoVocals',
        description='Return a ZIP with the embedded song and no-vocals MP3. Enables vocal separation and MP3 stems for this request.',
    ),
    viet_lyrics_fallback: bool | None = Form(
        None, alias='VietLyricsFallback',
        description='Enable or disable the Viet Lyrics fallback pass for this request. Omitted uses the saved endpoint setting; the pass runs only when the opening retry triggers.',
    ),
    multilingual: bool | None = Form(
        None, alias='Multilingual',
        description='Enable or disable per-segment language detection for Faster-Whisper. Omitted uses the saved endpoint setting.',
    ),
):
    """Wait for processing using saved endpoint defaults, then download the completed audio."""
    work_dir = None
    job = None
    response_ready = False
    try:
        filename = Path((file.filename or '').replace('\\', '/')).name
        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_AUDIO or any(ord(char) < 32 or char in '<>:"|?*' for char in filename):
            raise HTTPException(status_code=422, detail='File must have a supported audio filename')
        settings = await asyncio.to_thread(get_endpoint_config)
        validate_processing_profiles(settings)
        lyrics = lyrics.strip().lstrip('\ufeff').strip() if lyrics else None
        processing_options = settings.model_dump()
        if no_vocals:
            processing_options['vocal_separation'] = True
        if viet_lyrics_fallback is not None:
            processing_options['fallback_viet_lyrics'] = viet_lyrics_fallback
        if multilingual is not None:
            if multilingual and settings.backend != 'faster-whisper':
                raise HTTPException(
                    status_code=422,
                    detail='Multilingual is only supported by faster-whisper',
                )
            processing_options['multilingual'] = multilingual
        request = JobRequest(
            **processing_options, language=language.lower() if language else None,
            demucs_mp3=no_vocals, copy_no_vocals=no_vocals,
            use_lyrics=bool(lyrics), lyrics_mode=lyrics_mode, save_previous_results=False,
        )
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        work_dir = Path(tempfile.mkdtemp(prefix='ss-transcriber-api-', dir=TEMP_DIR))
        song = await asyncio.to_thread(prepare_upload, work_dir, file.file, suffix, lyrics)
        job = Job(request, work_dir=work_dir, upload_name=song.name)
        job.filename = filename
        endpoint_jobs[job.id] = job
        prune_endpoint_jobs()
        job.processing_task = asyncio.create_task(run_job(job))
        try:
            await job.processing_task
        except asyncio.CancelledError:
            if not job.cancel_requested:
                raise
        finally:
            job.processing_task = None
        if job.cancel_requested:
            job.status = 'cancelled'
            raise HTTPException(status_code=409, detail='Upload job cancelled')
        if job.status != 'running' or job.return_code != 0:
            raise HTTPException(status_code=500, detail={
                'message': 'Song processing failed', 'logs': list(job.logs)[-30:],
            })
        job.logs.append('Preparing download response…')
        path, download_name, media_type = await asyncio.to_thread(
            upload_result, work_dir, song, filename, request.copy_no_vocals,
        )
        if job.cancel_requested:
            job.status = 'cancelled'
            raise HTTPException(status_code=409, detail='Upload job cancelled')
        response = FileResponse(path, filename=download_name, media_type=media_type,
                                headers={'X-Job-ID': job.id},
                                background=BackgroundTask(shutil.rmtree, work_dir, ignore_errors=True))
        job.logs.append('Processing completed; download is ready for the requesting client.')
        job.status = 'completed'
        job.finished_at = utc_now()
        response_ready = True
        return response
    except asyncio.CancelledError:
        if job is not None:
            job.status = 'cancelled'
            job.finished_at = utc_now()
            job.logs.append('Upload request cancelled.')
        raise
    except Exception as exc:
        if job is not None:
            if job.status != 'cancelled':
                job.status = 'failed'
            job.finished_at = utc_now()
            detail = exc.detail if isinstance(exc, HTTPException) else str(exc)
            if isinstance(detail, str):
                job.logs.append(f'Upload error: {detail}')
        raise
    finally:
        prune_endpoint_jobs()
        await file.close()
        if work_dir is not None and not response_ready:
            await asyncio.to_thread(shutil.rmtree, work_dir, ignore_errors=True)


@app.get('/api/files')
def get_files():
    return {'files': relative_files(INPUT_DIR, SUPPORTED_AUDIO)}


@app.get('/api/music-files')
def get_music_files():
    files = []
    for library, label, root in (
        ('input', 'Input', INPUT_DIR),
        ('songs', '[No Vocals]', SONGS_DIR),
    ):
        files.extend(
            {
                'library': library,
                'group': label,
                'path': relative_path,
                'name': Path(relative_path).name,
            }
            for relative_path in relative_files(root, SUPPORTED_AUDIO)
        )
    return {'files': files}


@app.get('/api/audio/{relative_path:path}')
def get_audio(relative_path: str):
    return FileResponse(input_audio_path(relative_path))


@app.get('/api/audio-lyrics/{relative_path:path}')
def get_audio_lyrics(relative_path: str):
    path = input_audio_path(relative_path)
    return audio_lyrics(path)


def audio_lyrics(path: Path):
    try:
        tags = ID3(path)
    except ID3NoHeaderError:
        tags = ID3()

    sylt_frames = tags.getall('SYLT')
    sylt_frame = next((item for item in sylt_frames if item.desc == 'Transcription'), None)
    if sylt_frame is None and sylt_frames:
        sylt_frame = sylt_frames[0]

    uslt_frames = tags.getall('USLT')
    uslt_frame = next((item for item in uslt_frames if item.desc == 'Transcription'), None)
    if uslt_frame is None and uslt_frames:
        uslt_frame = uslt_frames[0]

    entries = []
    if sylt_frame is not None and sylt_frame.format == 2:
        entries = [
            {'text': text, 'time_ms': time_ms}
            for text, time_ms in sorted(sylt_frame.text, key=lambda item: item[1])
            if text.strip()
        ]

    return {
        'language': sylt_frame.lang if sylt_frame is not None else uslt_frame.lang if uslt_frame is not None else None,
        'entries': entries,
        'uslt': uslt_frame.text.strip() if uslt_frame is not None else '',
    }


@app.get('/api/music-audio/{library}/{relative_path:path}')
def get_music_audio(library: str, relative_path: str):
    return FileResponse(music_audio_path(library, relative_path))


@app.get('/api/music-lyrics/{library}/{relative_path:path}')
def get_music_lyrics(library: str, relative_path: str):
    return audio_lyrics(music_audio_path(library, relative_path))


@app.get('/api/endpoint-jobs')
def get_endpoint_jobs():
    return [job.public() for job in reversed(list(endpoint_jobs.values()))]


@app.get('/api/endpoint-jobs/{job_id}')
def get_endpoint_job(job_id: str):
    job = endpoint_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail='Endpoint job not found')
    return job.public(include_logs=True)


@app.delete('/api/endpoint-jobs/{job_id}', status_code=202)
async def cancel_endpoint_job(job_id: str):
    job = endpoint_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail='Endpoint job not found')
    if job.status not in {'queued', 'running'}:
        raise HTTPException(status_code=409, detail='Endpoint job has already finished')
    if not job.cancel_requested:
        job.cancel_requested = True
        job.logs.append('Cancellation requested from endpoint jobs.')
        if job.processing_task is not None:
            job.processing_task.cancel()
    return job.public(include_logs=True)


@app.get('/api/jobs')
def get_jobs():
    return [job.public() for job in reversed(list(jobs.values()))]


@app.post('/api/jobs', status_code=202)
async def create_job(request: JobRequest):
    validate_profile_options(request.backend, request.backend_options, 'backend profile')
    validate_profile_options(
        'viet-lyrics', request.fallback_viet_lyrics_options, 'fallback profile'
    )
    job = Job(request)
    jobs[job.id] = job
    asyncio.create_task(run_job(job))
    return job.public(include_logs=True)


@app.get('/api/jobs/{job_id}')
def get_job(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail='Job not found')
    return job.public(include_logs=True)


@app.delete('/api/jobs/{job_id}')
async def cancel_job(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail='Job not found')
    if job.status not in {'queued', 'running'}:
        raise HTTPException(status_code=409, detail='Job has already finished')
    job.cancel_requested = True
    if job.process is not None and job.process.returncode is None:
        if os.name == 'nt':
            subprocess.run(
                ['taskkill', '/PID', str(job.process.pid), '/T', '/F'],
                capture_output=True,
                check=False,
            )
        else:
            os.killpg(job.process.pid, signal.SIGTERM)
    return job.public(include_logs=True)


@app.get('/api/transcripts')
def get_transcripts():
    transcript_files = []
    for folder in sorted(OUTPUT_DIR.glob('transcripts*'), reverse=True):
        if not folder.is_dir():
            continue
        for path in sorted(folder.rglob('*')):
            if path.is_file() and path.suffix.lower() in {'.txt', '.json'}:
                transcript_files.append({
                    'path': path.relative_to(OUTPUT_DIR).as_posix(),
                    'size': path.stat().st_size,
                    'modified_at': datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                })
    return {'files': transcript_files}


@app.get('/api/transcripts/{relative_path:path}')
def get_transcript(relative_path: str):
    path = (OUTPUT_DIR / relative_path).resolve()
    if OUTPUT_DIR not in path.parents or not path.is_file():
        raise HTTPException(status_code=404, detail='Transcript not found')
    return FileResponse(path, media_type='text/plain; charset=utf-8')


if WEB_DIST.is_dir():
    @app.get('/endpoint-jobs', include_in_schema=False)
    @app.get('/endpoints', include_in_schema=False)
    @app.get('/health', include_in_schema=False)
    @app.get('/music', include_in_schema=False)
    @app.get('/results', include_in_schema=False)
    def get_webui_route():
        return FileResponse(WEB_DIST / 'index.html')

    app.mount('/', StaticFiles(directory=WEB_DIST, html=True), name='webui')


if __name__ == '__main__':
    import uvicorn

    parser = argparse.ArgumentParser(description='Run the local ssTranscriber Web API.')
    parser.add_argument('--host', default='127.0.0.1', help='Listening address (default: localhost; use 0.0.0.0 in containers).')
    parser.add_argument('--port', type=port_number, default=configured_port(), help='Listening port (default: WEB_API_PORT or 8000).')
    parser.add_argument('--reload', action='store_true', help='Reload the API when Python source files change.')
    args = parser.parse_args()
    uvicorn.run('web_api:app', host=args.host, port=args.port, reload=args.reload)