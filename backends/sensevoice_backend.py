from types import SimpleNamespace

from transcribe_common import log_progress

MODELS = (
    'FunAudioLLM/SenseVoiceSmall',
)

SUPPORTED_LANGUAGES = {'auto', 'zh', 'yue', 'en', 'ja', 'ko'}
# SUPPORTED_LANGUAGES: Languages SenseVoice was trained to recognize. Other languages are rejected
# because they can silently produce hallucinated text in one of the supported languages.
MODEL_HUB = 'hf'
# MODEL_HUB: Repository service used to resolve the model ID. "hf" selects Hugging Face.
VAD_MODEL = 'fsmn-vad'
# VAD_MODEL: Voice-activity model used to split speech from silence before recognition.
MAX_SINGLE_SEGMENT_TIME_MS = 30000
# MAX_SINGLE_SEGMENT_TIME_MS: Maximum VAD segment duration in milliseconds. Larger values retain more
# context but use more memory; smaller values split audio more often and can divide phrases.
DEFAULT_LANGUAGE = 'auto'
# DEFAULT_LANGUAGE: Language used when none is supplied. "auto" asks SenseVoice to detect it.
USE_ITN = True
# USE_ITN: Enable inverse text normalization. True formats spoken numbers and dates as written text;
# False preserves a more literal transcription.
BATCH_SIZE_S = 60
# BATCH_SIZE_S: Approximate seconds of audio processed per batch. Larger values improve throughput but
# use more memory; smaller values reduce memory use but increase batching overhead.


def load(model_name: str, device: str):
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
        hub=MODEL_HUB,
        vad_model=VAD_MODEL,
        vad_kwargs={'max_single_segment_time': MAX_SINGLE_SEGMENT_TIME_MS},
        device=device_str,
    )


def transcribe(model, path, language, label: str):
    if language and language not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f'sensevoice backend does not support language "{language}". '
            f'Supported languages: {sorted(SUPPORTED_LANGUAGES)}. '
            'Use --backend faster-whisper, --backend pho-whisper, or --backend viet-lyrics for Vietnamese.'
        )
    results = model.generate(
        input=str(path),
        cache={},
        language=language or DEFAULT_LANGUAGE,
        use_itn=USE_ITN,
        batch_size_s=BATCH_SIZE_S,
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
    info = SimpleNamespace(duration=duration, language=language or DEFAULT_LANGUAGE)
    if segments:
        log_progress(f'{label} — transcription 100%')
    return segments, info
