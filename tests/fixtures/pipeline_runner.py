"""Run the real pipeline with deterministic ASR/Demucs substitutes (no GPU/models)."""
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.modules['torch'] = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 0, is_available=lambda: False))

import backends
import transcribe_common

language = sys.argv[sys.argv.index('--language') + 1] if '--language' in sys.argv else None


def transcribe(*args, **kwargs):
    if language == 'fail':
        raise RuntimeError('Simulated recognition failure')
    segments = [] if language == 'empty' else [SimpleNamespace(start=0.5, end=3.0, text='A gentle melody')]
    return segments, SimpleNamespace(language='en')


def separate(path, device, output_root, use_mp3=False, mp3_bitrate=320):
    if language == 'no-stems':
        raise RuntimeError('Simulated separation failure')
    destination = output_root / 'htdemucs' / path.stem
    destination.mkdir(parents=True, exist_ok=True)
    suffix = '.mp3' if use_mp3 else '.wav'
    vocals = destination / f'vocals{suffix}'
    vocals.write_bytes(b'isolated vocals')
    (destination / f'no_vocals{suffix}').write_bytes(b'accompaniment')
    return vocals


original_embed = transcribe_common.write_lyrics_to_file


def embed(*args, **kwargs):
    if language == 'embed-fail':
        raise OSError('Simulated embedding failure')
    return original_embed(*args, **kwargs)


with patch.multiple(backends, resolve_options=Mock(return_value={}), apply_options=Mock(return_value={}),
                    get_options=Mock(return_value={}), get_models=Mock(return_value=[]),
                    load_model=Mock(return_value=object()), transcribe_audio=transcribe), \
        patch.object(transcribe_common, 'separate_vocals', separate), \
        patch.object(transcribe_common, 'write_lyrics_to_file', embed):
    runpy.run_path(str(ROOT / 'process_audio_folder.py'), run_name='__main__')