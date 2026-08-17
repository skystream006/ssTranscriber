import argparse
import gc
import os
import shutil
import subprocess
import sys
import threading
import time
import re
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import torch
from faster_whisper import WhisperModel
from mutagen.id3 import ID3, SYLT, USLT, Encoding

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass

REPO_ROOT = Path(__file__).resolve().parent


def load_dotenv(path: Path):
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, _, value = line.partition('=')
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_dotenv(REPO_ROOT / '.env')
_hf_token = (
    os.environ.get('HF_TOKEN')
    or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    or os.environ.get('HUGGINGFACE_HUB_TOKEN')
)
if _hf_token:
    os.environ.setdefault('HF_TOKEN', _hf_token)
    os.environ.setdefault('HUGGING_FACE_HUB_TOKEN', _hf_token)

ROOT = (REPO_ROOT / 'input').resolve()
SUPPORTED = {'.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg', '.opus', '.wma'}
LOG_PATH = REPO_ROOT / 'output' / 'processing_log.txt'
TRANSCRIPTS_DIR = REPO_ROOT / 'output' / 'transcripts'
TEMP_DIR = REPO_ROOT / 'temp'
ROOT.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# ID3v2.3 only allows Latin-1 or UTF-16; UTF-16 covers Vietnamese, Chinese and
# every other script, so it is used for all text frames.
TEXT_ENCODING = Encoding.UTF16

# ISO 639-1 (whisper) -> ISO 639-2/B (ID3 frame language field).
LANGUAGE_CODES = {
    'en': 'eng', 'vi': 'vie', 'zh': 'chi', 'ja': 'jpn', 'ko': 'kor',
    'th': 'tha', 'km': 'khm', 'lo': 'lao', 'fr': 'fre', 'de': 'ger',
    'es': 'spa', 'pt': 'por', 'it': 'ita', 'ru': 'rus', 'ar': 'ara',
    'hi': 'hin', 'id': 'ind', 'ms': 'may', 'nl': 'dut', 'pl': 'pol',
    'tr': 'tur', 'uk': 'ukr',
}

PROMOTIONAL_PHRASES = (
    'subscribe cho kênh',
    'đăng ký kênh',
    'like và đăng ký',
    'cảm ơn các bạn đã theo dõi',
    'hẹn gặp lại',
)

AVAILABLE_MODELS = {
    'faster-whisper': (
        'tiny',
        'base',
        'small',
        'medium',
        'large-v2',
        'large-v3',
        'large-v3-turbo',
        'distil-large-v2',
        'distil-large-v3',
    ),
    'pho-whisper': (
        'vinai/PhoWhisper-small',
        'vinai/PhoWhisper-base',
        'vinai/PhoWhisper-large',
    ),
    'parakeet': (
        'nvidia/parakeet-tdt-0.6b-v2',
        'nvidia/parakeet-tdt-1.1b',
        'nvidia/canary-1b',
    ),
    'sensevoice': (
        'FunAudioLLM/SenseVoiceSmall',
    ),
    'viet-lyrics': (
        'kelvinbksoh/whisper-small-vietnamese-lyrics-transcription',
        'kelvinbksoh/whisper-medium-vietnamese-lyrics-transcription',
        'kelvinbksoh/whisper-large-v2-vietnamese-lyrics-transcription',
    ),
}


def colorize(text: str, color: str, *, bold: bool = False) -> str:
    if not sys.stdout.isatty():
        return text
    styles = {'reset': '\033[0m', 'bold': '\033[1m'}
    palette = {
        'cyan': '\033[36m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'red': '\033[31m',
        'magenta': '\033[35m',
        'blue': '\033[34m',
    }
    prefix = styles['bold'] if bold else ''
    return f'{prefix}{palette.get(color, "")}{text}{styles["reset"]}'


def id3_language(code):
    return LANGUAGE_CODES.get((code or '').lower(), 'und')


def normalized_text(text: str):
    return re.sub(r'[^\w]+', ' ', text.casefold()).strip()


def filter_transcript_segments(segments, allow_promotions=False):
    filtered = []
    previous = ''
    removed_promotions = 0
    removed_duplicates = 0

    for segment in segments:
        text = (segment.text or '').strip()
        normalized = normalized_text(text)
        if not normalized:
            continue
        is_promotion = any(phrase in normalized for phrase in PROMOTIONAL_PHRASES)
        if is_promotion and not allow_promotions:
            removed_promotions += 1
            continue
        if normalized == previous:
            removed_duplicates += 1
            continue
        filtered.append(segment)
        previous = normalized

    return filtered, removed_promotions, removed_duplicates


def log_progress(message: str):
    line = f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {message}'
    if 'WARNING:' in message or 'warning' in message.lower():
        print(colorize(line, 'yellow', bold=True), flush=True)
    elif 'Failed:' in message or 'ERROR' in message.upper() or 'failed' in message.lower():
        print(colorize(line, 'red', bold=True), flush=True)
    elif 'Success' in message or 'completed' in message.lower() or 'finished' in message.lower():
        print(colorize(line, 'green', bold=True), flush=True)
    elif 'Model loaded' in message or 'Using device:' in message or 'Processing started' in message:
        print(colorize(line, 'cyan', bold=True), flush=True)
    else:
        print(line, flush=True)
    with LOG_PATH.open('a', encoding='utf-8') as log:
        log.write(line + '\n')


@contextmanager
def progress_heartbeat(message: str, interval=15):
    stopped = threading.Event()
    started = datetime.now()

    def report():
        while not stopped.wait(interval):
            elapsed = (datetime.now() - started).total_seconds()
            log_progress(f'{message} — still running ({elapsed:.0f}s elapsed)')

    thread = threading.Thread(target=report, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join()


def available_devices():
    devices = [('cpu', 'CPU')]
    for i in range(torch.cuda.device_count()):
        devices.append((f'cuda:{i}', f'CUDA:{i} - {torch.cuda.get_device_name(i)}'))
    return devices


def print_devices(devices):
    print(colorize('Available processing devices:', 'blue', bold=True))
    for idx, (_, label) in enumerate(devices):
        print(f'  {idx}: {label}')
    if len(devices) == 1:
        print('  (no CUDA GPU detected)')


def print_available_models(backend: str, selected_model: str):
    models = AVAILABLE_MODELS.get(backend, ())
    if not models:
        return
    print(colorize(f'Available models for backend "{backend}":', 'blue', bold=True))
    for name in models:
        marker = ' (selected)' if name == selected_model else ''
        print(f'  - {name}{marker}')


def prompt_for_device(devices):
    default_index = len(devices) - 1
    while True:
        try:
            choice = input(f'Select device [{default_index}]: ').strip()
        except EOFError:
            log_progress(
                'WARNING: No interactive input available; falling back to default device '
                f'({devices[default_index][1]})'
            )
            return devices[default_index][0]
        if not choice:
            return devices[default_index][0]
        if choice.isdigit() and int(choice) < len(devices):
            return devices[int(choice)][0]
        print(f'Enter a number between 0 and {len(devices) - 1}.')


def _retry_on_windows_file_lock(func, retries=6, initial_delay=0.5, label=None):
    # NeMo extracts .nemo checkpoints (tar archives) into a fresh temp directory
    # on every run. On Windows, newly created files in that directory are
    # commonly locked for a short time by antivirus/real-time scanning or the
    # search indexer, or by a lingering handle NeMo itself keeps open for lazy
    # access (e.g. manifest.json, tokenizer files). Either way this raises
    # OSError WinError 32. A short retry loop with backoff resolves both
    # cases; gc.collect() additionally releases any stale Python-held handle.
    last_exc = None
    delay = initial_delay
    for attempt in range(retries):
        try:
            return func()
        except OSError as exc:
            if getattr(exc, 'winerror', None) != 32:
                raise
            last_exc = exc
            if label:
                log_progress(
                    f'{label} — WARNING: file locked ({exc.filename}); '
                    f'retrying in {delay:.1f}s (attempt {attempt + 1}/{retries})'
                )
            gc.collect()
            time.sleep(delay)
            delay = min(delay * 2, 8.0)
    raise last_exc


def resolve_device(selection: str):
    if selection == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        return 'cpu'

    if selection == 'cpu':
        return 'cpu'

    if selection.startswith('cuda:'):
        index = int(selection.split(':', 1)[1])
        if index < torch.cuda.device_count():
            return selection
        raise ValueError(f'GPU index {index} is not available.')

    raise ValueError(f'Unsupported device selection: {selection}')


def faster_whisper_device(device: str):
    if device == 'cpu':
        return 'cpu', 0, 'int8'

    if device == 'cuda':
        return 'cuda', 0, 'float16'

    return 'cuda', int(device.split(':', 1)[1]), 'float16'


def load_model(model_name: str, backend: str, device: str):
    if backend == 'faster-whisper':
        backend_device, device_index, compute_type = faster_whisper_device(device)
        return WhisperModel(
            model_name,
            device=backend_device,
            device_index=device_index,
            compute_type=compute_type,
        )

    if backend == 'pho-whisper':
        try:
            from pho_whisper import PhoWhisper
            pho_device = 'cuda' if device.startswith('cuda') else 'cpu'
            try:
                return PhoWhisper(model_name, device=pho_device)
            except TypeError:
                try:
                    return PhoWhisper.from_pretrained(model_name, device=pho_device)
                except Exception:
                    raise
        except ImportError:
            # The "pho-whisper" PyPI package does not exist; PhoWhisper models
            # are standard Hugging Face Whisper checkpoints, so fall back to
            # loading them through transformers directly.
            try:
                from transformers import pipeline
            except ImportError as exc:
                raise RuntimeError(
                    f'transformers import failed ({exc}). Run install_audio_tools.py '
                    'or pip install transformers accelerate.'
                ) from exc
            device_id = 0 if device.startswith('cuda') else -1
            try:
                return pipeline(
                    'automatic-speech-recognition',
                    model=model_name,
                    device=device_id,
                )
            except Exception as exc:
                # Surface the real error instead of masking it. Common causes:
                # the model requires accepting terms on Hugging Face, a
                # network/download failure, or an incompatible transformers
                # version for this checkpoint.
                raise RuntimeError(
                    f'Failed to load PhoWhisper model "{model_name}" via transformers: {exc}'
                ) from exc

    if backend == 'parakeet':
        try:
            from nemo.collections.asr.models import ASRModel
        except ImportError as exc:
            raise RuntimeError(
                'NeMo toolkit is not installed. Run install_audio_tools.py --with-parakeet '
                'or pip install "nemo_toolkit[asr]".'
            ) from exc
        model = _retry_on_windows_file_lock(
            lambda: ASRModel.from_pretrained(model_name),
            label=f'Loading parakeet model "{model_name}"',
        )
        model = model.to('cuda' if device.startswith('cuda') else 'cpu')
        model.eval()
        return model

    if backend == 'sensevoice':
        try:
            from funasr import AutoModel
        except ImportError as exc:
            raise RuntimeError(
                'FunASR is not installed. Run install_audio_tools.py --with-sensevoice '
                'or pip install funasr modelscope.'
            ) from exc
        device_str = 'cuda:0' if device.startswith('cuda') else 'cpu'
        # FunASR defaults to resolving model IDs against ModelScope. Hugging Face
        # repo IDs such as "FunAudioLLM/SenseVoiceSmall" must request hub='hf'
        # explicitly, otherwise FunASR 404s against ModelScope and then fails
        # with "model '<id>' is not registered".
        return AutoModel(
            model=model_name,
            hub='hf',
            vad_model='fsmn-vad',
            vad_kwargs={'max_single_segment_time': 30000},
            device=device_str,
        )

    if backend == 'viet-lyrics':
        # Standalone backend for kelvinbksoh's Vietnamese lyrics-transcription
        # checkpoints (fine-tuned Whisper), independent of the pho-whisper backend.
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                f'transformers import failed ({exc}). Run install_audio_tools.py '
                'or pip install transformers accelerate.'
            ) from exc
        device_id = 0 if device.startswith('cuda') else -1
        try:
            return pipeline(
                'automatic-speech-recognition',
                model=model_name,
                device=device_id,
            )
        except Exception as exc:
            raise RuntimeError(
                f'Failed to load viet-lyrics model "{model_name}" via transformers: {exc}'
            ) from exc

    raise ValueError(f'Unsupported backend: {backend}')


def iter_audio_files(root: Path):
    for path in sorted(root.rglob('*')):
        if (
            path.is_file()
            and path.suffix.lower() in SUPPORTED
            and path.name.lower() not in {'vocals.wav', 'no_vocals.wav', 'vocals.mp3', 'no_vocals.mp3'}
        ):
            yield path


def clear_transcripts():
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    for child in TRANSCRIPTS_DIR.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()
    log_progress(f'Cleared transcript output directory: {TRANSCRIPTS_DIR}')


def separate_vocals(path: Path, device: str, output_root: Path, use_mp3=False, mp3_bitrate=320):
    separation_start = datetime.now()
    log_progress(f'{path.name} — Demucs vocal separation started')
    command = [
        sys.executable,
        '-m',
        'demucs',
        '--two-stems=vocals',
        '--device', device,
        '--out', str(output_root),
    ]
    if use_mp3:
        command += ['--mp3', '--mp3-bitrate', str(mp3_bitrate)]
    command.append(str(path))
    child_environment = os.environ.copy()
    child_environment['PYTHONUTF8'] = '1'
    child_environment['PYTHONIOENCODING'] = 'utf-8'
    with progress_heartbeat(f'{path.name} — Demucs vocal separation'):
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            env=child_environment,
            check=False,
        )
    if result.returncode != 0:
            details = result.stdout.strip()
            with LOG_PATH.open('a', encoding='utf-8') as log:
                log.write(f'\n--- Demucs output for {path.name} ---\n')
                log.write(details + '\n--- End Demucs output ---\n')
            lines = [line.strip() for line in details.splitlines() if line.strip()]
            reason = ' | '.join(lines[-3:]) if lines else f'exit code {result.returncode}'
            raise RuntimeError(f'Demucs exited with code {result.returncode}: {reason}')

    vocals_filename = 'vocals.mp3' if use_mp3 else 'vocals.wav'
    vocals = output_root / 'htdemucs' / path.stem / vocals_filename
    if not vocals.is_file():
        matches = list(output_root.rglob(vocals_filename))
        if len(matches) != 1:
            raise RuntimeError(f'Demucs completed but {vocals_filename} was not found.')
        vocals = matches[0]
    elapsed = (datetime.now() - separation_start).total_seconds()
    log_progress(f'{path.name} — Demucs vocal separation completed ({elapsed:.1f}s)')
    return vocals


def transcribe_audio(model, path: Path, language, label: str, backend: str):
    log_progress(f'{label} — transcription started')
    if backend == 'faster-whisper':
        segment_iterator, info = model.transcribe(
            str(path),
            language=language,
            beam_size=5,
            # beam_size: Number of candidate decodings retained at each step. Higher values
                # Larger numbers can improve accuracy but increase GPU memory use and
                # processing time; smaller numbers are faster but less exhaustive.
            repetition_penalty=1.1,
            # repetition_penalty: Discourages the model from repeating the same phrase in music or
                # quiet sections. Values near 1.0 are neutral; higher values suppress
                # Larger numbers suppress repetition more aggressively but may remove
                # intentional repeats; values near 1.0 preserve more repetitions.
            word_timestamps=True,
            # word_timestamps: Return segment timestamps suitable for SYLT synchronized lyrics.
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            # temperature: Enable temperature fallback by passing a tuple.
            #   Prevent the model from getting stuck in a loop of non-speech tokens.
            no_speech_threshold=0.01,
            # no_speech_threshold: Lower than the default so quiet singing is less likely to be rejected
                # Larger values reject more quiet audio as silence; lower values capture
                # more quiet singing but can increase hallucinations.
            log_prob_threshold=-5.0,
            # log_prob_threshold: Accept segments with lower average log probability. This helps unclear
                # Larger values reject more low-confidence text; more negative values
                # accept less reliable singing or instrumental guesses.
            vad_filter=False,
            # vad_filter: Disabled because VAD is tuned for speech and can remove sung vocals.
            condition_on_previous_text=False,
            # condition_on_previous_text: Prevents text from one segment contaminating the next segment and
            #   reduces repeated promotional or hallucinated phrases.
        )
        segments = []
        last_reported = 0
        duration = max(float(info.duration), 0.001)
        for segment in segment_iterator:
            segments.append(segment)
            percent = min(100, int(float(segment.end) / duration * 100))
            report_at = percent // 10 * 10
            if report_at >= last_reported + 10:
                last_reported = report_at
                log_progress(f'{label} — transcription {report_at}%')
        if segments and last_reported < 100:
            log_progress(f'{label} — transcription 100%')
        return segments, info

    if backend == 'pho-whisper':
        kwargs = {
            'return_timestamps': True,
            'chunk_length_s': 30,
            'stride_length_s': 5,
        }
        if language:
            kwargs['language'] = language
        result = model(str(path), **kwargs)

        segments = []
        if isinstance(result, dict):
            chunks = result.get('chunks') or []
            if chunks:
                for chunk in chunks:
                    text = (chunk.get('text') or '').strip()
                    if not text:
                        continue
                    start = float(chunk.get('timestamp', (0.0, 0.0))[0])
                    end = float(chunk.get('timestamp', (0.0, 0.0))[1] or start)
                    segments.append(SimpleNamespace(start=start, end=end, text=text))
            else:
                text = (result.get('text') or '').strip()
                if text:
                    segments.append(SimpleNamespace(start=0.0, end=0.0, text=text))
        elif isinstance(result, list):
            for item in result:
                if isinstance(item, dict):
                    text = (item.get('text') or '').strip()
                    if not text:
                        continue
                    ts = item.get('timestamp') or (0.0, 0.0)
                    start = float(ts[0])
                    end = float(ts[1] or start)
                    segments.append(SimpleNamespace(start=start, end=end, text=text))
        elif isinstance(result, str):
            text = result.strip()
            if text:
                segments.append(SimpleNamespace(start=0.0, end=0.0, text=text))

        duration = max(float(path.stat().st_size or 0.0), 0.001)
        info = SimpleNamespace(duration=duration, language=language or 'vi')
        if segments:
            last_reported = 0
            for segment in segments:
                percent = min(100, int(float(segment.end) / max(float(duration), 0.001) * 100))
                report_at = percent // 10 * 10
                if report_at >= last_reported + 10:
                    last_reported = report_at
                    log_progress(f'{label} — transcription {report_at}%')
            if last_reported < 100:
                log_progress(f'{label} — transcription 100%')
        return segments, info

    if backend == 'parakeet':
        # NeMo's transcribe() writes its own internal manifest.json into a
        # fresh temp directory on every call and immediately reads it back
        # via a DataLoader. On Windows, if that DataLoader uses worker
        # processes (num_workers > 0, the default), those workers can try to
        # reopen the manifest before the main process's write is flushed,
        # causing a deterministic WinError 32 (a new temp dir fails every
        # time, not just occasionally). Forcing num_workers=0 keeps everything
        # single-process and avoids the race; the retry wrapper remains as a
        # safety net for any other transient lock (e.g. antivirus scanning).
        # num_workers=0 is applied unconditionally (not via inspect.signature)
        # because decorators used inside NeMo can hide the real parameter
        # list, which previously caused the guard to silently skip it.
        def _transcribe_parakeet():
            try:
                return model.transcribe([str(path)], timestamps=True, num_workers=0)
            except TypeError as exc:
                if 'num_workers' not in str(exc):
                    raise
                return model.transcribe([str(path)], timestamps=True)

        hypotheses = _retry_on_windows_file_lock(_transcribe_parakeet, label=label)
        hypothesis = hypotheses[0] if hypotheses else None
        text_value = getattr(hypothesis, 'text', hypothesis)
        segments = []
        timestamp_info = getattr(hypothesis, 'timestamp', None) if hypothesis is not None else None
        segment_timestamps = None
        if isinstance(timestamp_info, dict):
            segment_timestamps = timestamp_info.get('segment') or timestamp_info.get('word')
        if segment_timestamps:
            for item in segment_timestamps:
                text = (item.get('segment') or item.get('word') or '').strip()
                if not text:
                    continue
                start = float(item.get('start', 0.0))
                end = float(item.get('end', start))
                segments.append(SimpleNamespace(start=start, end=end, text=text))
        elif text_value:
            text = str(text_value).strip()
            if text:
                segments.append(SimpleNamespace(start=0.0, end=0.0, text=text))

        duration = max(float(path.stat().st_size or 0.0), 0.001)
        info = SimpleNamespace(duration=duration, language=language or 'en')
        if segments:
            log_progress(f'{label} — transcription 100%')
        return segments, info

    if backend == 'sensevoice':
        # SenseVoice was only trained on Chinese, Cantonese, English, Japanese, and
        # Korean. Requesting any other language (e.g. Vietnamese) silently produces
        # hallucinated text in one of those trained languages instead of failing,
        # so this is rejected explicitly rather than returning garbage output.
        sensevoice_supported_languages = {'auto', 'zh', 'yue', 'en', 'ja', 'ko'}
        if language and language not in sensevoice_supported_languages:
            raise ValueError(
                f'sensevoice backend does not support language "{language}". '
                f'Supported languages: {sorted(sensevoice_supported_languages)}. '
                'Use --backend faster-whisper, --backend pho-whisper, or --backend viet-lyrics for Vietnamese.'
            )
        results = model.generate(
            input=str(path),
            cache={},
            language=language or 'auto',
            use_itn=True,
            batch_size_s=60,
        )
        segments = []
        for item in results or []:
            if not isinstance(item, dict):
                continue
            text = (item.get('text') or '').strip()
            if not text:
                continue
            try:
                from funasr.utils.postprocess_utils import rich_transcription_postprocess
                text = rich_transcription_postprocess(text)
            except ImportError:
                pass
            start_ms = item.get('start')
            end_ms = item.get('end')
            start = float(start_ms) / 1000 if start_ms is not None else 0.0
            end = float(end_ms) / 1000 if end_ms is not None else start
            segments.append(SimpleNamespace(start=start, end=end, text=text))

        duration = max(float(path.stat().st_size or 0.0), 0.001)
        info = SimpleNamespace(duration=duration, language=language or 'auto')
        if segments:
            log_progress(f'{label} — transcription 100%')
        return segments, info

    if backend == 'viet-lyrics':
        kwargs = {
            'return_timestamps': True,
            'chunk_length_s': 30,
            'stride_length_s': 5,
        }
        if language:
            kwargs['language'] = language
        result = model(str(path), **kwargs)

        segments = []
        if isinstance(result, dict):
            chunks = result.get('chunks') or []
            if chunks:
                for chunk in chunks:
                    text = (chunk.get('text') or '').strip()
                    if not text:
                        continue
                    start = float(chunk.get('timestamp', (0.0, 0.0))[0])
                    end = float(chunk.get('timestamp', (0.0, 0.0))[1] or start)
                    segments.append(SimpleNamespace(start=start, end=end, text=text))
            else:
                text = (result.get('text') or '').strip()
                if text:
                    segments.append(SimpleNamespace(start=0.0, end=0.0, text=text))
        elif isinstance(result, list):
            for item in result:
                if isinstance(item, dict):
                    text = (item.get('text') or '').strip()
                    if not text:
                        continue
                    ts = item.get('timestamp') or (0.0, 0.0)
                    start = float(ts[0])
                    end = float(ts[1] or start)
                    segments.append(SimpleNamespace(start=start, end=end, text=text))
        elif isinstance(result, str):
            text = result.strip()
            if text:
                segments.append(SimpleNamespace(start=0.0, end=0.0, text=text))

        duration = max(float(path.stat().st_size or 0.0), 0.001)
        info = SimpleNamespace(duration=duration, language=language or 'vi')
        if segments:
            last_reported = 0
            for segment in segments:
                percent = min(100, int(float(segment.end) / max(float(duration), 0.001) * 100))
                report_at = percent // 10 * 10
                if report_at >= last_reported + 10:
                    last_reported = report_at
                    log_progress(f'{label} — transcription {report_at}%')
            if last_reported < 100:
                log_progress(f'{label} — transcription 100%')
        return segments, info

    raise ValueError(f'Unsupported backend for transcription: {backend}')


def build_sylt_entries(segments):
    # Shared with the .txt transcript writer so the on-disk lyrics file always
    # matches exactly what gets embedded as the mp3's SYLT frame.
    entries = []
    for seg in segments:
        text = (seg.text or '').strip()
        if not text:
            continue
        start_ms = int(float(seg.start) * 1000)
        entries.append((text, start_ms))
    return entries


def format_sylt_as_lrc(sylt_entries):
    lines = []
    for text, start_ms in sylt_entries:
        minutes, remainder_ms = divmod(max(start_ms, 0), 60_000)
        seconds, centiseconds = divmod(remainder_ms, 1000)
        centiseconds //= 10
        lines.append(f'[{minutes:02d}:{seconds:02d}.{centiseconds:02d}]{text}')
    return '\n'.join(lines)


def write_lyrics_to_file(path: Path, transcript: str, segments, language='und'):
    try:
        tags = ID3(path)
    except Exception:
        tags = ID3()

    tags.delall('USLT')
    tags.delall('SYLT')

    uslt = USLT(encoding=TEXT_ENCODING, lang=language, desc='Transcription', text=transcript)
    tags.add(uslt)

    sylt_entries = build_sylt_entries(segments)
    if sylt_entries:
        sylt = SYLT(
            encoding=TEXT_ENCODING,
            lang=language,
            format=2,
            type=1,
            desc='Transcription',
            text=sylt_entries,
        )
        tags.add(sylt)

    tags.save(path, v2_version=3)
    return True


def main():
    start_time = datetime.now()
    parser = argparse.ArgumentParser(description='Transcribe audio files and embed USLT/SYLT lyrics metadata.')
    devices = available_devices()

    parser.add_argument(
        '--device',
        default=None,
        help='Processing device: auto, cpu, or cuda:N. Omit to choose interactively; falls back to auto when non-interactive.',
    )
    parser.add_argument(
        '--language',
        default=None,
        help='Force a transcription language (ISO 639-1, e.g. vi, zh, ja). Auto-detected when omitted.',
    )
    parser.add_argument(
        '--backend',
        choices=('faster-whisper', 'pho-whisper', 'parakeet', 'sensevoice', 'viet-lyrics'),
        default='faster-whisper',
        help=(
            'Choose the transcription backend: faster-whisper (Whisper models), '
            'pho-whisper (PhoWhisper), parakeet (NVIDIA NeMo Parakeet/Canary), '
            'sensevoice (FunASR SenseVoice), or viet-lyrics (kelvinbksoh Vietnamese '
            'lyrics-transcription Whisper fine-tunes).'
        ),
    )
    parser.add_argument(
        '--model',
        default='large-v3',
        help=(
            'Model name for the selected backend, e.g. large-v3 (faster-whisper), '
            'vinai/PhoWhisper-base (pho-whisper), nvidia/parakeet-tdt-0.6b-v2 (parakeet), '
            'FunAudioLLM/SenseVoiceSmall (sensevoice), or '
            'kelvinbksoh/whisper-large-v2-vietnamese-lyrics-transcription (viet-lyrics).'
        ),
    )
    parser.add_argument(
        '--no-vocal-separation',
        action='store_true',
        help='Transcribe original audio without first isolating vocals with Demucs.',
    )
    parser.add_argument(
        '--demucs-mp3',
        action='store_true',
        help='Have Demucs write the separated vocals stem as MP3 instead of WAV.',
    )
    parser.add_argument(
        '--demucs-mp3-bitrate',
        type=int,
        default=320,
        help='MP3 bitrate (kbps) used when --demucs-mp3 is set. Default 320.',
    )
    parser.add_argument(
        '--file',
        type=Path,
        default=None,
        help='Process one audio file under input/ (relative path) instead of the full folder.',
    )
    parser.add_argument(
        '--keep-promotions',
        action='store_true',
        help='Keep promotional phrases such as "subscribe" and "đăng ký" instead of removing them.',
    )
    parser.add_argument(
        '--opening-threshold',
        type=float,
        default=10.0,
        help=(
            'Seconds. If the first usable vocal segment from Demucs-separated audio starts '
            'later than this, retry transcription on the original (non-separated) audio to '
            'recover a possibly clipped opening. Default 1.0 forces transcription to expect '
            'the opening at essentially 0:01 instead of tolerating a longer detected gap.'
        ),
    )
    args = parser.parse_args()

    LOG_PATH.write_text('', encoding='utf-8')
    log_progress('Processing started')

    print_devices(devices)
    if args.device is None:
        if sys.stdin is not None and sys.stdin.isatty():
            device = prompt_for_device(devices)
        else:
            device = resolve_device('auto')
    else:
        device = resolve_device(args.device)
    log_progress(f'Using device: {device}')
    print_available_models(args.backend, args.model)

    results = []
    log_progress(f'Scanning for audio files under {ROOT}')
    files = list(iter_audio_files(ROOT))
    if args.file is not None:
        requested_path = (ROOT / args.file).resolve()
        if ROOT not in requested_path.parents or requested_path not in files:
            parser.error('--file must name a supported audio file under input/.')
        files = [requested_path]
    log_progress(f'Found {len(files)} files to process')
    clear_transcripts()

    if not files:
        end_time = datetime.now()
        summary = f'[{start_time.strftime("%Y-%m-%d %H:%M:%S")}] Processed root: {ROOT}\nFiles found: 0\nSuccessfully updated: 0\nFailed: 0\nSkipped: 0\nNo supported sound files were found.\n[{end_time.strftime("%Y-%m-%d %H:%M:%S")}] Duration: {(end_time - start_time).total_seconds():.2f}s'
        with LOG_PATH.open('a', encoding='utf-8') as log:
            log.write(summary + '\n')
        print(summary)
        return

    backend_device, device_index, compute_type = faster_whisper_device(device)
    log_progress(
        f'Loading {args.backend} model "{args.model}" '
        f'(first run may download several GB)'
    )
    try:
        with progress_heartbeat(f'Loading {args.backend} model "{args.model}"'):
            model = load_model(args.model, args.backend, device)
    except Exception as load_error:
        log_progress(f'Failed to load {args.backend} model "{args.model}": {load_error}')
        raise
    log_progress(f'Model loaded; starting transcription with {args.backend} model "{args.model}"')
    for idx, path in enumerate(files, 1):
        file_start = datetime.now()
        log_progress(f'[{idx}/{len(files)}] {path.name} — processing started using model "{args.model}"')
        try:
            transcription_path = path
            used_original_fallback = False
            if not args.no_vocal_separation:
                try:
                    transcription_path = separate_vocals(
                        path,
                        device,
                        TEMP_DIR,
                        use_mp3=args.demucs_mp3,
                        mp3_bitrate=args.demucs_mp3_bitrate,
                    )
                except Exception as separation_error:
                    log_progress(
                        f'{path.name} — WARNING: {separation_error}; '
                        'falling back to original audio'
                    )

            segments, info = transcribe_audio(
                model, transcription_path, args.language, path.name, args.backend
            )
            candidate_segments, _, _ = filter_transcript_segments(segments, allow_promotions=args.keep_promotions)
            if (
                transcription_path != path
                and (
                    not candidate_segments
                    or float(candidate_segments[0].start) > args.opening_threshold
                )
            ):
                used_original_fallback = True
                if candidate_segments:
                    fallback_message = (
                        f'{path.name} — first usable vocal segment starts at '
                        f'{candidate_segments[0].start:.1f}s; retrying original audio '
                        'to recover the opening'
                    )
                else:
                    fallback_message = (
                        f'{path.name} — no usable vocal segments; retrying original audio'
                    )
                log_progress(fallback_message)
                try:
                    segments, info = transcribe_audio(model, path, args.language, path.name, args.backend)
                except Exception as fallback_error:
                    # The original file can occasionally fail to decode (e.g. an
                    # unusual WAV encoding faster-whisper's loader can't parse)
                    # even though Demucs read it fine. Keep the Demucs-based
                    # transcript already produced instead of losing it entirely.
                    used_original_fallback = False
                    log_progress(
                        f'{path.name} — WARNING: could not retry original audio '
                        f'({fallback_error}); keeping separated-vocal transcript'
                    )
            segments, removed_promotions, removed_duplicates = filter_transcript_segments(
                segments, allow_promotions=args.keep_promotions
            )
            if removed_promotions or removed_duplicates:
                log_progress(
                    f'{path.name} — removed {removed_promotions} promotional and '
                    f'{removed_duplicates} duplicate segment(s)'
                )
            transcript = '\n'.join(
                segment.text.strip() for segment in segments if segment.text.strip()
            ).strip()
            if not transcript:
                status = 'Failed: no transcript produced'
                results.append((path.name, status))
                log_progress(f'[{idx}/{len(files)}] {path.name} — {status}')
                continue

            language = id3_language(args.language or info.language)
            log_progress(f'{path.name} — writing USLT/SYLT metadata and transcript')
            write_lyrics_to_file(path, transcript, segments, language)

            # Write transcript as LRC-style synced lyrics, matching the mp3's SYLT frame exactly.
            sylt_entries = build_sylt_entries(segments)
            transcript_output = format_sylt_as_lrc(sylt_entries) if sylt_entries else transcript
            relative_path = path.relative_to(ROOT).with_suffix('.txt')
            transcript_path = TRANSCRIPTS_DIR / relative_path
            transcript_path.parent.mkdir(parents=True, exist_ok=True)
            transcript_path.write_text(transcript_output, encoding='utf-8')

            if used_original_fallback:
                original_relative = path.relative_to(ROOT).with_name(f'{path.stem}_original.txt')
                original_transcript_path = TRANSCRIPTS_DIR / original_relative
                original_transcript_path.parent.mkdir(parents=True, exist_ok=True)
                original_transcript_path.write_text(transcript_output, encoding='utf-8')
                log_progress(f'{path.name} — wrote original-audio fallback transcript: {original_transcript_path.name}')

            status = f'Success (USLT + SYLT embedded, lang={language})'
            results.append((path.name, status))
            elapsed = (datetime.now() - file_start).total_seconds()
            log_progress(f'[{idx}/{len(files)}] {path.name} — {status} ({elapsed:.1f}s)')
        except Exception as exc:
            status = f'Failed: {exc}'
            results.append((path.name, status))
            elapsed = (datetime.now() - file_start).total_seconds()
            log_progress(f'[{idx}/{len(files)}] {path.name} — {status} ({elapsed:.1f}s)')

    end_time = datetime.now()
    log_progress('Generating summary')
    summary_lines = [
        f'[{start_time.strftime("%Y-%m-%d %H:%M:%S")}] Processed root: {ROOT}',
        f'Files found: {len(files)}',
        f'Successfully updated: {sum(1 for _, status in results if status.startswith("Success"))}',
        f'Failed: {sum(1 for _, status in results if status.startswith("Failed"))}',
        f'Skipped: {sum(1 for _, status in results if status.startswith("Skipped"))}',
        f'Duration: {(end_time - start_time).total_seconds():.2f}s',
        f'[{end_time.strftime("%Y-%m-%d %H:%M:%S")}] Completed',
        '',
        'Per file:'
    ]
    for name, status in results:
        summary_lines.append(f'- {name} — {status}')

    summary = '\n'.join(summary_lines) + '\n'
    with LOG_PATH.open('a', encoding='utf-8') as log:
        log.write('\n' + summary)
    print(summary)


if __name__ == '__main__':
    main()
