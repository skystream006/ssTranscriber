import argparse
import ast
import asyncio
import json
import os
import signal
import subprocess
import sys
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

REPO_ROOT = Path(__file__).resolve().parent
INPUT_DIR = (REPO_ROOT / 'input').resolve()
OUTPUT_DIR = (REPO_ROOT / 'output').resolve()
WEB_DIST = REPO_ROOT / 'webui' / 'dist'
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


class JobRequest(BaseModel):
    file: str | None = None
    backend: str = 'faster-whisper'
    model: str = 'large-v3'
    device: str = 'auto'
    language: str | None = None
    vocal_separation: bool = True
    demucs_mp3: bool = False
    demucs_mp3_bitrate: int = Field(default=320, ge=64, le=512)
    keep_promotions: bool = False
    use_lyrics: bool = True
    lyrics_mode: Literal['prompt', 'align', 'correct'] = 'prompt'
    save_previous_results: bool = True
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
    def __init__(self, request: JobRequest):
        self.id = uuid.uuid4().hex[:12]
        self.request = request
        self.status = 'queued'
        self.created_at = utc_now()
        self.started_at = None
        self.finished_at = None
        self.return_code = None
        self.logs = deque(maxlen=4000)
        self.process = None
        self.cancel_requested = False

    def public(self, include_logs=False):
        result = {
            'id': self.id,
            'status': self.status,
            'created_at': self.created_at,
            'started_at': self.started_at,
            'finished_at': self.finished_at,
            'return_code': self.return_code,
            'request': self.request.model_dump(),
            'log_count': len(self.logs),
        }
        if include_logs:
            result['logs'] = list(self.logs)
        return result


jobs: dict[str, Job] = {}
job_lock = asyncio.Lock()


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
        '--lyrics-mode',
        request.lyrics_mode,
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
            job.process = await asyncio.create_subprocess_exec(
                *build_command(job.request),
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
                job.status = 'completed' if job.return_code == 0 else 'failed'
        except Exception as exc:
            job.logs.append(f'Web API error: {exc}')
            job.status = 'failed'
        finally:
            job.finished_at = utc_now()
            job.process = None


app = FastAPI(title='ssTranscriber API', version='1.0.0')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:5173', 'http://127.0.0.1:5173'],
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.get('/api/config')
def get_config():
    return {
        'backends': backend_metadata(),
        'devices': available_devices(),
        'lyrics_modes': ['prompt', 'align', 'correct'],
    }


@app.get('/api/files')
def get_files():
    return {'files': relative_files(INPUT_DIR, SUPPORTED_AUDIO)}


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
            if path.is_file() and path.suffix.lower() in {'.txt', '.txttxt', '.json'}:
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
    app.mount('/', StaticFiles(directory=WEB_DIST, html=True), name='webui')


if __name__ == '__main__':
    import uvicorn

    parser = argparse.ArgumentParser(description='Run the local ssTranscriber Web API.')
    parser.add_argument('--port', type=port_number, default=configured_port(), help='Listening port (default: WEB_API_PORT or 8000).')
    parser.add_argument('--reload', action='store_true', help='Reload the API when Python source files change.')
    args = parser.parse_args()
    uvicorn.run('web_api:app', host='127.0.0.1', port=args.port, reload=args.reload)