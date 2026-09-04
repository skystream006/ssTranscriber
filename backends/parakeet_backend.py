from types import SimpleNamespace

from transcribe_common import log_progress, retry_on_windows_file_lock

MODELS = (
    'nvidia/parakeet-tdt-0.6b-v2',
    'nvidia/parakeet-tdt-1.1b',
    'nvidia/canary-1b',
)

TIMESTAMPS = True
# TIMESTAMPS: Request segment or word timestamps from NeMo. Must remain True for synchronized lyrics.
NUM_WORKERS = 0
# NUM_WORKERS: DataLoader worker processes used during transcription. Larger values may improve
# throughput but can race on NeMo's temporary manifest on Windows; zero uses the reliable main process.
DEFAULT_LANGUAGE = 'en'
# DEFAULT_LANGUAGE: Language recorded when no language is supplied. Parakeet models default to English.


def load(model_name: str, device: str):
    try:
        from nemo.collections.asr.models import ASRModel
    except ImportError as exc:
        raise RuntimeError(
            'NeMo toolkit is not installed. Run install_audio_tools.py --with-parakeet '
            'or pip install "nemo_toolkit[asr]".'
        ) from exc
    model = retry_on_windows_file_lock(
        lambda: ASRModel.from_pretrained(model_name),
        label=f'Loading parakeet model "{model_name}"',
    )
    model = model.to('cuda' if device.startswith('cuda') else 'cpu')
    model.eval()
    return model


def transcribe(model, path, language, label: str, lyrics_text=None):
    def _transcribe_parakeet():
        try:
            return model.transcribe(
                [str(path)],
                timestamps=TIMESTAMPS,
                num_workers=NUM_WORKERS,
            )
        except TypeError as exc:
            if 'num_workers' not in str(exc):
                raise
            return model.transcribe([str(path)], timestamps=TIMESTAMPS)

    hypotheses = retry_on_windows_file_lock(_transcribe_parakeet, label=label)
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
    info = SimpleNamespace(duration=duration, language=language or DEFAULT_LANGUAGE)
    if segments:
        log_progress(f'{label} — transcription 100%')
    return segments, info
