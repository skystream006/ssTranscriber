from transcribe_common import load_transformers_asr_pipeline, run_transformers_pipeline_transcription

MODELS = (
    'vinai/PhoWhisper-small',
    'vinai/PhoWhisper-base',
    'vinai/PhoWhisper-large',
)


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
    return run_transformers_pipeline_transcription(model, path, language, label, default_language='vi')
