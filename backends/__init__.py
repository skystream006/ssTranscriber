"""Registry that maps a --backend name to its model loader/transcriber module."""
from . import faster_whisper_backend
from . import parakeet_backend
from . import pho_whisper_backend
from . import sensevoice_backend
from . import viet_lyrics_backend
from transcribe_common import log_progress

_BACKEND_MODULES = {
    'faster-whisper': faster_whisper_backend,
    'pho-whisper': pho_whisper_backend,
    'parakeet': parakeet_backend,
    'sensevoice': sensevoice_backend,
    'viet-lyrics': viet_lyrics_backend,
}

BACKEND_CHOICES = tuple(_BACKEND_MODULES)
AVAILABLE_MODELS = {name: module.MODELS for name, module in _BACKEND_MODULES.items()}


def load_model(model_name: str, backend: str, device: str):
    module = _BACKEND_MODULES.get(backend)
    if module is None:
        raise ValueError(f'Unsupported backend: {backend}')
    return module.load(model_name, device)


def transcribe_audio(model, path, language, label: str, backend: str):
    module = _BACKEND_MODULES.get(backend)
    if module is None:
        raise ValueError(f'Unsupported backend for transcription: {backend}')
    log_progress(f'{label} — transcription started')
    return module.transcribe(model, path, language, label)
