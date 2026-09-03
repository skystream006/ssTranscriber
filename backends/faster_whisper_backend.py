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

MODEL_KWARGS = {
    'cpu_threads': 0,
    # cpu_threads: Number of CPU threads used per worker. Larger values can speed CPU inference but use
    # more CPU resources; smaller values reduce contention. Zero chooses automatically.
    'num_workers': 5,
    # num_workers: Number of concurrent transcription workers sharing this model instance. Larger values
    # improve parallel throughput but consume more memory; smaller values use fewer resources.
    'download_root': None,
    # download_root: Directory used for model files. None uses the default Hugging Face cache.
    'local_files_only': False,
    # local_files_only: Load only cached/local model files and disable network downloads when True.
    'files': None,
    # files: Preloaded model-file mapping for advanced loaders. None reads files normally.
    'revision': None,
    # revision: Hugging Face model branch, tag, or commit. None uses the repository default.
    'use_auth_token': None,
    # use_auth_token: Hugging Face token for private/gated models. None uses configured credentials.
}

TRANSCRIBE_KWARGS = {
    'task': 'transcribe',
    # task: Decode speech in its original language. Use "translate" for English translation.
    'log_progress': False,
    # log_progress: Enable Faster-Whisper's progress output in addition to this app's progress log.
    'beam_size': 1,
    # beam_size: Number of candidate sequences retained during beam search. Higher values can improve
    # accuracy but increase decoding time and memory use; this must be a positive integer.
    'best_of': 5,
    # best_of: Number of sampled candidates considered when temperature is greater than zero. Larger
    # values may find better text but decode more slowly; smaller values are faster and narrower.
    'patience': 1.0,
    # patience: Beam-search patience factor. Larger values explore more candidates and run longer;
    # smaller values stop sooner and can miss alternatives. Values below 1.0 are more aggressive.
    'length_penalty': 1.0,
    # length_penalty: Score penalty based on output length. Larger values favor longer sequences;
    # smaller values favor shorter sequences. A value of 1.0 applies standard length normalization.
    'repetition_penalty': 1.1,
    # repetition_penalty: Penalize generated tokens. Higher values reduce repetition but can remove
    # intentional repeated lyrics; 1.0 disables the penalty.
    'no_repeat_ngram_size': 0,
    # no_repeat_ngram_size: Block repeated token groups of this size. Smaller positive values block
    # short patterns more aggressively; larger values only block longer phrases. Zero disables it.
    'temperature': (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    # temperature: Decoding temperatures tried in order when quality thresholds trigger fallback.
    # Zero uses deterministic beam search; higher values introduce more randomness.
    'compression_ratio_threshold': 2.4,
    # compression_ratio_threshold: Retry when gzip compression exceeds this value. Smaller values reject
    # repetition more aggressively; larger values tolerate more repetitive text. None disables it.
    'log_prob_threshold': -5.0,
    # log_prob_threshold: Retry below this average token log probability. Larger values are stricter and
    # reject more uncertain text; smaller/more-negative values accept lower confidence. None disables it.
    'no_speech_threshold': 0.01,
    # no_speech_threshold: Treat a window as silence above this probability when log probability is low.
    # Smaller values skip audio more aggressively; larger values retain more quiet audio. None disables it.
    'condition_on_previous_text': False,
    # condition_on_previous_text: Feed previous text into the next window. False reduces repetition.
    'prompt_reset_on_temperature': 0.5,
    # prompt_reset_on_temperature: Reset previous text at or above this temperature. Smaller values
    # reset context more often; larger values preserve context through more fallback attempts.
    'initial_prompt': None,
    # initial_prompt: Text or token IDs supplied as context at the beginning of the recording.
    'prefix': None,
    # prefix: Text forced at the beginning of the first decoding window.
    'suppress_blank': True,
    # suppress_blank: Suppress blank output at the start of each decoding window.
    'suppress_tokens': [-1],
    # suppress_tokens: Token IDs to suppress. [-1] expands to the default non-speech token list.
    'without_timestamps': False,
    # without_timestamps: Decode without timestamp tokens. Must be False for synchronized lyrics.
    'max_initial_timestamp': 1.0,
    # max_initial_timestamp: Latest allowed first-token timestamp. Larger values permit
    # longer leading silence; smaller values force transcription to begin nearer the chunk start.
    'word_timestamps': True,
    # word_timestamps: Compute individual word times for transcript and SYLT generation.
    'prepend_punctuations': '"\'“¿([{-',
    # prepend_punctuations: Punctuation attached to the following word for word timestamps.
    'append_punctuations': '"\'.。,，!！?？:：”)]}、',
    # append_punctuations: Punctuation attached to the preceding word for word timestamps.
    'multilingual': False,
    # multilingual: Allow language detection and decoding to change between audio segments.
    'vad_filter': False,
    # vad_filter: Filter non-speech regions with Silero VAD before transcription.
    'vad_parameters': None,
    # vad_parameters: Silero VAD settings. A larger threshold removes more quiet audio; a smaller
    # threshold retains more. Larger minimum silence merges pauses; larger padding retains context.
    'max_new_tokens': None,
    # max_new_tokens: Maximum new tokens per chunk. Larger values allow longer text but use more
    # time and memory; smaller values finish sooner but can truncate output. None uses the default.
    'chunk_length': None,
    # chunk_length: Audio chunk duration in seconds. Larger chunks provide context but use more memory;
    # smaller chunks use less memory but introduce more boundaries. None uses the default.
    'clip_timestamps': '0',
    # clip_timestamps: Comma-separated clip boundaries in seconds; "0" processes the full recording.
    'hallucination_silence_threshold': None,
    # hallucination_silence_threshold: Skip long silence after hallucinations. Smaller values skip
    # silence more aggressively; larger values preserve longer pauses. None disables it.
    'hotwords': None,
    # hotwords: Words or phrases to bias during decoding. Support depends on the selected model.
    'language_detection_threshold': 0.5,
    # language_detection_threshold: Minimum accepted language confidence. Larger values demand more
    # certainty and may inspect more audio; smaller values accept a language sooner. None accepts the best.
    'language_detection_segments': 1,
    # language_detection_segments: Initial segments used for detection. Larger values can improve
    # detection on ambiguous audio but take longer; smaller values are faster with less evidence.
}


def load(model_name: str, device: str):
    backend_device, device_index, compute_type = faster_whisper_device(device)
    return WhisperModel(
        model_name,
        device=backend_device,
        device_index=device_index,
        compute_type=compute_type,
        **MODEL_KWARGS,
    )


def transcribe(model, path, language, label: str):
    segment_iterator, info = model.transcribe(
        str(path),
        language=language,
        **TRANSCRIBE_KWARGS,
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
