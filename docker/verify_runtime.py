"""Build-time native dependency/audio I/O check; no GPU or model downloads required."""
import ctypes
import os
import tempfile
from pathlib import Path

# Check dynamic linker discovery before importing Torch (which preloads CUDA libs).
# Loading these libraries does not require a GPU/driver during the image build.
if os.environ.get('NVIDIA_VISIBLE_DEVICES'):
    for library in ('libcublas.so.12', 'libcudnn.so.9'):
        ctypes.CDLL(library)

import ctranslate2
import demucs.separate
import librosa
import soundfile
import torch
import torchaudio
from demucs.audio import save_audio
from faster_whisper.audio import decode_audio
from transformers import pipeline


with tempfile.TemporaryDirectory() as directory:
    waveform = torch.zeros(2, 4410)
    for suffix in ('.wav', '.mp3'):
        path = Path(directory) / f'smoke{suffix}'
        save_audio(waveform, str(path), samplerate=44100)
        assert path.stat().st_size > 0
        assert decode_audio(str(path)).size > 0

print(f'Runtime imports and WAV/MP3 I/O OK: torch={torch.__version__}, '
      f'torchaudio={torchaudio.__version__}, ctranslate2={ctranslate2.__version__}')