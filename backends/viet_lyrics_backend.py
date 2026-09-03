from transcribe_common import load_transformers_asr_pipeline, run_transformers_pipeline_transcription

MODELS = (
    'kelvinbksoh/whisper-small-vietnamese-lyrics-transcription',
    'kelvinbksoh/whisper-medium-vietnamese-lyrics-transcription',
    'kelvinbksoh/whisper-large-v2-vietnamese-lyrics-transcription',
)
CHUNK_LENGTH_S = 30


def load(model_name: str, device: str):
    # Standalone backend for kelvinbksoh's Vietnamese lyrics-transcription
    # checkpoints (fine-tuned Whisper), independent of the pho-whisper backend.
    return load_transformers_asr_pipeline(model_name, device, 'viet-lyrics')


def transcribe(model, path, language, label: str):
    return run_transformers_pipeline_transcription(
        model,
        path,
        language,
        label,
        default_language='vi',
        chunk_length_s=CHUNK_LENGTH_S,
    )
