import importlib.util
import os
from pathlib import Path


def _register_nvidia_dll_directories():
    if os.name != 'nt' or not hasattr(os, 'add_dll_directory'):
        return []

    handles = []
    for package_name in ('nvidia.cublas', 'nvidia.cudnn'):
        package_spec = importlib.util.find_spec(package_name)
        if package_spec is None or package_spec.submodule_search_locations is None:
            continue
        for package_dir in package_spec.submodule_search_locations:
            bin_dir = Path(package_dir) / 'bin'
            if bin_dir.is_dir():
                handles.append(os.add_dll_directory(str(bin_dir)))
    return handles


_NVIDIA_DLL_DIRECTORY_HANDLES = _register_nvidia_dll_directories()

from faster_whisper import WhisperModel

from transcribe_common import faster_whisper_device, log_progress

MODELS = (
    'tiny',
    'base',
    'small',
    'medium',
    'large-v2',
    'large-v3',
    'large-v3-turbo',
    'distil-large-v2',
    'distil-large-v3',
)


def load(model_name: str, device: str):
    backend_device, device_index, compute_type = faster_whisper_device(device)
    return WhisperModel(
        model_name,
        device=backend_device,
        device_index=device_index,
        compute_type=compute_type,
    )


def transcribe(model, path, language, label: str):
    segment_iterator, info = model.transcribe(
        str(path),
        language=language,
        beam_size=5,
        # beam_size: Number of candidate decodings retained at each step. Higher values
            # Larger numbers can improve accuracy but increase GPU memory use and
            # processing time; smaller numbers are faster but less exhaustive.
        repetition_penalty=1.1,
        # repetition_penalty: Discourages the model from repeating the same phrase in music or
            # quiet sections. Values near 1.0 are neutral; higher values suppress
            # Larger numbers suppress repetition more aggressively but may remove
            # intentional repeats; values near 1.0 preserve more repetitions.
        word_timestamps=True,
        # word_timestamps: Return segment timestamps suitable for SYLT synchronized lyrics.
        # temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        # temperature: Enable temperature fallback by passing a tuple.
        #   Prevent the model from getting stuck in a loop of non-speech tokens.
        no_speech_threshold=0.01,
        # no_speech_threshold: Lower than the default so quiet singing is less likely to be rejected
            # Larger values reject more quiet audio as silence; lower values capture
            # more quiet singing but can increase hallucinations.
        log_prob_threshold=-5.0,
        # log_prob_threshold: Accept segments with lower average log probability. This helps unclear
            # Larger values reject more low-confidence text; more negative values
            # accept less reliable singing or instrumental guesses.
        vad_filter=False,
        # vad_filter: Disabled because VAD is tuned for speech and can remove sung vocals.
        condition_on_previous_text=False,
        # condition_on_previous_text: Prevents text from one segment contaminating the next segment and
        #   reduces repeated promotional or hallucinated phrases.
    )
    segments = []
    last_reported = 0
    duration = max(float(info.duration), 0.001)
    for segment in segment_iterator:
        segments.append(segment)
        percent = min(100, int(float(segment.end) / duration * 100))
        report_at = percent // 10 * 10
        if report_at >= last_reported + 10:
            last_reported = report_at
            log_progress(f'{label} — transcription {report_at}%')
    if segments and last_reported < 100:
        log_progress(f'{label} — transcription 100%')
    return segments, info
