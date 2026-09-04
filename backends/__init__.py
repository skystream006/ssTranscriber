"""Registry that maps a --backend name to its model loader/transcriber module.

Backend modules are imported lazily (only once actually requested), so e.g.
running with --backend faster-whisper never imports nemo/funasr, and a
process that only needs backends.viet_lyrics_backend never imports
faster_whisper_backend. This matters beyond startup time: CTranslate2
(faster-whisper) and torch/transformers bundle different, ABI-incompatible
cuDNN builds, and having both loaded in the same process crashes on Windows
with "Could not load symbol cudnnGetLibConfig. Error code 127".
"""
import importlib
from pathlib import Path

from transcribe_common import log_progress

_MODULE_NAMES = {
    'faster-whisper': 'faster_whisper_backend',
    'pho-whisper': 'pho_whisper_backend',
    'parakeet': 'parakeet_backend',
    'sensevoice': 'sensevoice_backend',
    'viet-lyrics': 'viet_lyrics_backend',
}

BACKEND_CHOICES = tuple(_MODULE_NAMES)
_loaded_modules = {}


def _get_module(backend: str):
    module_name = _MODULE_NAMES.get(backend)
    if module_name is None:
        raise ValueError(f'Unsupported backend: {backend}')
    module = _loaded_modules.get(backend)
    if module is None:
        module = importlib.import_module(f'.{module_name}', __name__)
        _loaded_modules[backend] = module
    return module


def get_models(backend: str):
    return _get_module(backend).MODELS


def _json_value(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, set):
        return sorted(_json_value(item) for item in value)
    if isinstance(value, Path):
        return str(value)
    return repr(value)


def get_options(backend: str):
    module = _get_module(backend)
    return {
        name: _json_value(value)
        for name, value in vars(module).items()
        if name.isupper() and not name.startswith('_') and name != 'MODELS'
    }


def load_model(model_name: str, backend: str, device: str):
    return _get_module(backend).load(model_name, device)


def transcribe_audio(model, path, language, label: str, backend: str, lyrics_text=None):
    log_progress(f'{label} — transcription started')
    return _get_module(backend).transcribe(model, path, language, label, lyrics_text=lyrics_text)

