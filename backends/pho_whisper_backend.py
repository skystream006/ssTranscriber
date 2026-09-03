from transcribe_common import load_transformers_asr_pipeline, run_transformers_pipeline_transcription

MODELS = (
    'vinai/PhoWhisper-small',
    'vinai/PhoWhisper-base',
    'vinai/PhoWhisper-large',
)

DEFAULT_LANGUAGE = 'vi'
# DEFAULT_LANGUAGE: Language recorded when none is supplied. PhoWhisper is specialized for Vietnamese.
CHUNK_LENGTH_S = 30
# CHUNK_LENGTH_S: Audio chunk duration in seconds. Larger values provide more context but use more
# memory; smaller values reduce memory use but create more boundaries and potential word splits.
STRIDE_LENGTH_S = 5
# STRIDE_LENGTH_S: Overlap in seconds between adjacent chunks. Larger values preserve more boundary
# context but repeat more work; smaller values are faster but can lose words near chunk edges.
LINE_PAUSE_THRESHOLD = 0.6
# LINE_PAUSE_THRESHOLD: Silence gap in seconds that starts a new lyric line. Larger values combine
# phrases across longer pauses; smaller values split lines more frequently.
MAX_WORDS_PER_LINE = 14
# MAX_WORDS_PER_LINE: Maximum words grouped into one lyric line. Larger values produce longer lines;
# smaller values produce shorter lines and more timestamp entries.
MAX_LINE_DURATION = 12.0
# MAX_LINE_DURATION: Maximum duration in seconds for one lyric line. Larger values permit longer
# lines; smaller values split phrases sooner for more frequent synchronized-lyrics updates.


def load(model_name: str, device: str):
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
        return load_transformers_asr_pipeline(model_name, device, 'PhoWhisper')


def transcribe(model, path, language, label: str):
    return run_transformers_pipeline_transcription(
        model,
        path,
        language,
        label,
        default_language=DEFAULT_LANGUAGE,
        chunk_length_s=CHUNK_LENGTH_S,
        stride_length_s=STRIDE_LENGTH_S,
        line_pause_threshold=LINE_PAUSE_THRESHOLD,
        max_words_per_line=MAX_WORDS_PER_LINE,
        max_line_duration=MAX_LINE_DURATION,
    )
