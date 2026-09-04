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

GENERATION_KWARGS = {
    'task': 'transcribe',
    # task: Preserve speech in its original language. Use "translate" to generate English text.
    'num_beams': 1,
    # num_beams: Candidate sequences retained during beam search. Larger values may improve accuracy
    # but increase decoding time and GPU memory; smaller values are faster. Must be a positive integer.
    'condition_on_prev_tokens': False,
    # condition_on_prev_tokens: Feed previous chunk text into the next chunk. True improves continuity
    # but can propagate repetition; False isolates chunks and reduces repeated or contaminated text.
    'repetition_penalty': 1.1,
    # repetition_penalty: Penalize previously generated tokens. Larger values suppress repetition more
    # aggressively but can remove repeated lyrics; values nearer 1.0 preserve intentional repetition.
    'no_repeat_ngram_size': 3,
    # no_repeat_ngram_size: Block repeated token groups of this size. Smaller positive values block
    # short patterns more aggressively; larger values only block longer phrases. Zero disables it.
    'temperature': 0.0,
    # temperature: Sampling randomness. Larger values produce more varied, less predictable text;
    # smaller values are more deterministic. Zero uses deterministic decoding.
    'no_speech_threshold': 0.3,
    # no_speech_threshold: Treat a chunk as silence above this probability when confidence is low.
    # Smaller values skip audio more aggressively; larger values retain more quiet audio. None disables it.
    'logprob_threshold': -5.0,
    # logprob_threshold: Retry or reject decoding below this average token log probability. Larger values
    # are stricter; smaller/more-negative values accept lower-confidence singing. None disables it.
    'compression_ratio_threshold': 2.4,
    # compression_ratio_threshold: Retry text above this repetition/compression ratio. Smaller values
    # reject repetition more aggressively; larger values tolerate more repetitive output. None disables it.
    'max_new_tokens': 440,
    # max_new_tokens: Maximum generated tokens per chunk. Larger values allow longer text but use more
    # time and memory; smaller values finish sooner but can truncate output. Keep below Whisper's limit.
}


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


def transcribe(model, path, language, label: str, lyrics_text=None):
    return run_transformers_pipeline_transcription(
        model,
        path,
        language,
        label,
        lyrics_text=lyrics_text,
        default_language=DEFAULT_LANGUAGE,
        chunk_length_s=CHUNK_LENGTH_S,
        stride_length_s=STRIDE_LENGTH_S,
        line_pause_threshold=LINE_PAUSE_THRESHOLD,
        max_words_per_line=MAX_WORDS_PER_LINE,
        max_line_duration=MAX_LINE_DURATION,
        generation_kwargs=GENERATION_KWARGS,
    )
