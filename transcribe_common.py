"""Shared utilities used by process_audio_folder.py and the backends package.

Keeps device/logging/ID3/demucs helpers in one place so backend modules only
need to implement model-specific loading and transcription.
"""
import gc
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from types import SimpleNamespace

import torch
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
    'please subscribe',
    'subscribe to the channel',
    'like and subscribe',
    'thanks for watching',
    'thank you for watching',
    'see you next time',
)


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


def _words_with_normalized_text(text):
    words = []
    for word in re.findall(r'\S+', text):
        normalized = normalized_text(word)
        if normalized:
            words.append((word, normalized))
    return words


def _alignment_anchors(lyric_words, transcript_words):
    matcher = SequenceMatcher(None, lyric_words, transcript_words, autojunk=False)
    anchors = {(0, 0), (len(lyric_words), len(transcript_words))}
    for lyric_start, transcript_start, size in matcher.get_matching_blocks():
        for offset in range(size + 1):
            anchors.add((lyric_start + offset, transcript_start + offset))
    return sorted(anchors)


def _map_position(position, anchors):
    before = anchors[0]
    for after in anchors[1:]:
        if position <= after[0]:
            if after[0] == before[0]:
                return float(after[1])
            fraction = (position - before[0]) / (after[0] - before[0])
            return before[1] + fraction * (after[1] - before[1])
        before = after
    return float(anchors[-1][1])


def _transcript_word_times(segments):
    word_times = []
    for segment in segments:
        words = _words_with_normalized_text(segment.text or '')
        if not words:
            continue
        start = float(segment.start)
        end = max(float(segment.end), start)
        duration = end - start
        for index, (_, normalized) in enumerate(words):
            word_start = start + duration * index / len(words)
            word_end = start + duration * (index + 1) / len(words)
            word_times.append((normalized, word_start, word_end))
    return word_times


def _time_at_word_position(position, word_times):
    if not word_times:
        return 0.0
    if position <= 0:
        return word_times[0][1]
    if position >= len(word_times):
        return word_times[-1][2]
    lower = int(position)
    fraction = position - lower
    left = word_times[lower - 1][2] if lower else word_times[0][1]
    right = word_times[lower][1]
    return left + fraction * (right - left)


def apply_lyrics_mode(segments, lyrics_text, mode):
    segments = list(segments)
    if not lyrics_text or mode == 'prompt' or not segments:
        return segments

    lyric_lines = [line.strip() for line in lyrics_text.splitlines() if line.strip()]
    lyric_words = _words_with_normalized_text('\n'.join(lyric_lines))
    word_times = _transcript_word_times(segments)
    if not lyric_lines or not lyric_words or not word_times:
        return segments

    lyric_normalized = [normalized for _, normalized in lyric_words]
    transcript_normalized = [normalized for normalized, _, _ in word_times]
    anchors = _alignment_anchors(lyric_normalized, transcript_normalized)

    if mode == 'align':
        aligned = []
        lyric_position = 0
        for line in lyric_lines:
            line_word_count = len(_words_with_normalized_text(line))
            if not line_word_count:
                continue
            start_position = _map_position(lyric_position, anchors)
            lyric_position += line_word_count
            end_position = _map_position(lyric_position, anchors)
            start = _time_at_word_position(start_position, word_times)
            end = max(_time_at_word_position(end_position, word_times), start)
            aligned.append(SimpleNamespace(start=start, end=end, text=line))
        return aligned or segments

    if mode == 'correct':
        reverse_anchors = sorted((transcript, lyric) for lyric, transcript in anchors)
        corrected = []
        transcript_position = 0
        previous_lyric_end = 0
        for index, segment in enumerate(segments):
            segment_word_count = len(_words_with_normalized_text(segment.text or ''))
            transcript_position += segment_word_count
            lyric_end = round(_map_position(transcript_position, reverse_anchors))
            if index == len(segments) - 1:
                lyric_end = len(lyric_words)
            lyric_end = min(max(lyric_end, previous_lyric_end), len(lyric_words))
            text = ' '.join(word for word, _ in lyric_words[previous_lyric_end:lyric_end])
            corrected.append(SimpleNamespace(
                start=float(segment.start),
                end=float(segment.end),
                text=text or segment.text,
            ))
            previous_lyric_end = lyric_end
        return corrected

    raise ValueError(f'Unsupported lyrics mode: {mode}')


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


def print_available_models(backend: str, selected_model: str, models):
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


def retry_on_windows_file_lock(func, retries=6, initial_delay=0.5, label=None):
    # Windows reports transient file handles as either sharing violations
    # (WinError 32) or access denied (WinError 5), including directory renames.
    # A short retry loop handles antivirus/indexer scans and recently exited
    # child processes; gc.collect() also releases stale Python-held handles.
    last_exc = None
    delay = initial_delay
    for attempt in range(retries):
        try:
            return func()
        except OSError as exc:
            if os.name != 'nt' or getattr(exc, 'winerror', None) not in {5, 32}:
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


def iter_audio_files(root: Path):
    for path in sorted(root.rglob('*')):
        if (
            path.is_file()
            and path.suffix.lower() in SUPPORTED
            and path.name.lower() not in {'vocals.wav', 'no_vocals.wav', 'vocals.mp3', 'no_vocals.mp3'}
        ):
            yield path


def clear_transcripts(save_previous=False):
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    if save_previous and any(TRANSCRIPTS_DIR.iterdir()):
        suffix = 1
        while True:
            archive_dir = TRANSCRIPTS_DIR.with_name(f'{TRANSCRIPTS_DIR.name}_{suffix:02d}')
            if not archive_dir.exists():
                retry_on_windows_file_lock(
                    lambda: TRANSCRIPTS_DIR.rename(archive_dir),
                    label='Archiving previous transcripts',
                )
                TRANSCRIPTS_DIR.mkdir(parents=True)
                log_progress(f'Archived previous transcripts: {archive_dir}')
                return archive_dir
            suffix += 1

    for child in TRANSCRIPTS_DIR.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()
    log_progress(f'Cleared transcript output directory: {TRANSCRIPTS_DIR}')
    return None


def archive_demucs_results():
    demucs_dir = TEMP_DIR / 'htdemucs'
    demucs_dir.mkdir(parents=True, exist_ok=True)
    if not any(demucs_dir.iterdir()):
        return None

    suffix = 1
    while True:
        archive_dir = demucs_dir.with_name(f'{demucs_dir.name}_{suffix:02d}')
        if not archive_dir.exists():
            archive_dir.mkdir()
            for child in demucs_dir.iterdir():
                retry_on_windows_file_lock(
                    lambda child=child: child.rename(archive_dir / child.name),
                    label=f'Archiving previous Demucs result ({child.name})',
                )
            log_progress(f'Archived previous Demucs results: {archive_dir}')
            return archive_dir
        suffix += 1


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


_LANG_HEADER_RE = re.compile(r'^\[lang:([a-z]{2,3})\]$')
_LRC_LINE_RE = re.compile(r'^\[(\d{2}):(\d{2})\.(\d{2})\](.*)$')


def write_transcript_file(path: Path, language: str, transcript_output: str):
    # Prepends a "[lang:xx]" header so embed_lyrics.py can recover the detected
    # language from the transcript file alone, without re-running transcription.
    path.parent.mkdir(parents=True, exist_ok=True)
    header = f'[lang:{language}]\n' if language else ''
    path.write_text(header + transcript_output, encoding='utf-8')


def read_transcript_file(path: Path):
    # Inverse of write_transcript_file(): returns (language_or_None, sylt_entries, plain_transcript).
    lines = path.read_text(encoding='utf-8').splitlines()
    language = None
    if lines:
        header_match = _LANG_HEADER_RE.match(lines[0].strip())
        if header_match:
            language = header_match.group(1)
            lines = lines[1:]

    entries = []
    plain_lines = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        match = _LRC_LINE_RE.match(line)
        if match:
            minutes, seconds, centiseconds, lyric = match.groups()
            start_ms = (int(minutes) * 60 + int(seconds)) * 1000 + int(centiseconds) * 10
            lyric = lyric.strip()
            entries.append((lyric, start_ms))
            plain_lines.append(lyric)
        else:
            plain_lines.append(line)

    return language, entries, '\n'.join(plain_lines).strip()


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


def load_transformers_asr_pipeline(model_name: str, device: str, label: str):
    # Shared by any backend that is just a standard Hugging Face Whisper
    # checkpoint loaded through transformers.pipeline(...).
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError(
            f'transformers import failed ({exc}). Run _install_audio_tools.py '
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
            f'Failed to load {label} model "{model_name}" via transformers: {exc}'
        ) from exc


def run_transformers_pipeline_transcription(
    model,
    path: Path,
    language,
    label: str,
    lyrics_text=None,
    default_language='und',
    chunk_length_s=30,
    stride_length_s=5,
    line_pause_threshold=0.6,
    max_words_per_line=14,
    max_line_duration=12.0,
    generation_kwargs=None,
):
    # Shared chunked-ASR-pipeline transcription logic (pho-whisper, viet-lyrics).
    # Uses word-level timestamps rather than one timestamp per 30s chunk,
    # since this pipeline (unlike faster-whisper) has no built-in phrase/pause
    # segmentation; words are then regrouped into lyric-line-sized segments
    # below using pause gaps, so the transcript/SYLT output isn't one giant
    # blob of text per chunk.
    kwargs = {
        'return_timestamps': 'word',
        'chunk_length_s': chunk_length_s,
        'stride_length_s': stride_length_s,
    }
    kwargs.update(generation_kwargs or {})
    if lyrics_text:
        prompt_ids = model.tokenizer.get_prompt_ids(lyrics_text, return_tensors='pt')
        prompt_ids = prompt_ids[-128:]
        kwargs['prompt_ids'] = prompt_ids
        max_target_positions = getattr(model.model.config, 'max_target_positions', 448)
        prompt_budget = max_target_positions - len(prompt_ids) - 4
        if kwargs.get('max_new_tokens') is not None:
            kwargs['max_new_tokens'] = min(kwargs['max_new_tokens'], prompt_budget)
    if language:
        kwargs['language'] = language
    result = model(str(path), **kwargs)

    words = []
    if isinstance(result, dict):
        chunks = result.get('chunks') or []
        for chunk in chunks:
            text = (chunk.get('text') or '').strip()
            if not text:
                continue
            ts = chunk.get('timestamp') or (0.0, 0.0)
            start = float(ts[0]) if ts[0] is not None else 0.0
            end = float(ts[1]) if ts[1] is not None else start
            words.append((text, start, end))
        if not words:
            text = (result.get('text') or '').strip()
            if text:
                words.append((text, 0.0, 0.0))
    elif isinstance(result, list):
        for item in result:
            if isinstance(item, dict):
                text = (item.get('text') or '').strip()
                if not text:
                    continue
                ts = item.get('timestamp') or (0.0, 0.0)
                start = float(ts[0]) if ts[0] is not None else 0.0
                end = float(ts[1]) if ts[1] is not None else start
                words.append((text, start, end))
    elif isinstance(result, str):
        text = result.strip()
        if text:
            words.append((text, 0.0, 0.0))

    segments = []
    line_words = []
    line_start = None
    previous_end = None
    for text, start, end in words:
        if line_words and (
            (previous_end is not None and start - previous_end > line_pause_threshold)
            or len(line_words) >= max_words_per_line
            or (line_start is not None and end - line_start > max_line_duration)
        ):
            segments.append(SimpleNamespace(start=line_start, end=previous_end, text=' '.join(line_words)))
            line_words = []
            line_start = None
        if line_start is None:
            line_start = start
        line_words.append(text)
        previous_end = end
    if line_words:
        segments.append(SimpleNamespace(start=line_start, end=previous_end, text=' '.join(line_words)))

    duration = max(float(path.stat().st_size or 0.0), 0.001)
    info = SimpleNamespace(duration=duration, language=language or default_language)
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

