from types import SimpleNamespace

from transcribe_common import log_progress

MODELS = (
    'FunAudioLLM/SenseVoiceSmall',
)

# SenseVoice was only trained on Chinese, Cantonese, English, Japanese, and
# Korean. Requesting any other language (e.g. Vietnamese) silently produces
# hallucinated text in one of those trained languages instead of failing,
# so this is rejected explicitly rather than returning garbage output.
SUPPORTED_LANGUAGES = {'auto', 'zh', 'yue', 'en', 'ja', 'ko'}


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
        hub='hf',
        vad_model='fsmn-vad',
        vad_kwargs={'max_single_segment_time': 30000},
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
