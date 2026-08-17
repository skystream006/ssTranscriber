import argparse
import sys
from datetime import datetime
from pathlib import Path

import backends
from transcribe_common import (
    LOG_PATH,
    ROOT,
    TEMP_DIR,
    TRANSCRIPTS_DIR,
    available_devices,
    build_sylt_entries,
    clear_transcripts,
    filter_transcript_segments,
    format_sylt_as_lrc,
    id3_language,
    iter_audio_files,
    log_progress,
    print_available_models,
    print_devices,
    progress_heartbeat,
    prompt_for_device,
    resolve_device,
    separate_vocals,
    write_lyrics_to_file,
)


def main():
    start_time = datetime.now()
    parser = argparse.ArgumentParser(description='Transcribe audio files and embed USLT/SYLT lyrics metadata.')
    devices = available_devices()

    parser.add_argument(
        '--device',
        default=None,
        help='Processing device: auto, cpu, or cuda:N. Omit to choose interactively; falls back to auto when non-interactive.',
    )
    parser.add_argument(
        '--language',
        default=None,
        help='Force a transcription language (ISO 639-1, e.g. vi, zh, ja). Auto-detected when omitted.',
    )
    parser.add_argument(
        '--backend',
        choices=backends.BACKEND_CHOICES,
        default='faster-whisper',
        help=(
            'Choose the transcription backend: faster-whisper (Whisper models), '
            'pho-whisper (PhoWhisper), parakeet (NVIDIA NeMo Parakeet/Canary), '
            'sensevoice (FunASR SenseVoice), or viet-lyrics (kelvinbksoh Vietnamese '
            'lyrics-transcription Whisper fine-tunes).'
        ),
    )
    parser.add_argument(
        '--model',
        default='large-v3',
        help=(
            'Model name for the selected backend, e.g. large-v3 (faster-whisper), '
            'vinai/PhoWhisper-base (pho-whisper), nvidia/parakeet-tdt-0.6b-v2 (parakeet), '
            'FunAudioLLM/SenseVoiceSmall (sensevoice), or '
            'kelvinbksoh/whisper-large-v2-vietnamese-lyrics-transcription (viet-lyrics).'
        ),
    )
    parser.add_argument(
        '--no-vocal-separation',
        action='store_true',
        help='Transcribe original audio without first isolating vocals with Demucs.',
    )
    parser.add_argument(
        '--demucs-mp3',
        action='store_true',
        help='Have Demucs write the separated vocals stem as MP3 instead of WAV.',
    )
    parser.add_argument(
        '--demucs-mp3-bitrate',
        type=int,
        default=320,
        help='MP3 bitrate (kbps) used when --demucs-mp3 is set. Default 320.',
    )
    parser.add_argument(
        '--file',
        type=Path,
        default=None,
        help='Process one audio file under input/ (relative path) instead of the full folder.',
    )
    parser.add_argument(
        '--keep-promotions',
        action='store_true',
        help='Keep promotional phrases such as "subscribe" and "đăng ký" instead of removing them.',
    )
    parser.add_argument(
        '--opening-threshold',
        type=float,
        default=20.0,
        help=(
            'Seconds. If the first usable vocal segment from Demucs-separated audio starts '
            'later than this, retry transcription on the original (non-separated) audio to '
            'recover a possibly clipped opening. Default 1.0 forces transcription to expect '
            'the opening at essentially 0:01 instead of tolerating a longer detected gap.'
        ),
    )
    parser.add_argument(
        '--fallback-viet-lyrics',
        action='store_true',
        help=(
            'When the --opening-threshold retry triggers, transcribe the retry pass with the '
            'viet-lyrics backend instead of re-running the primary --backend/--model.'
        ),
    )
    parser.add_argument(
        '--fallback-viet-lyrics-model',
        default='kelvinbksoh/whisper-large-v2-vietnamese-lyrics-transcription',
        help='Model used for --fallback-viet-lyrics retries.',
    )
    args = parser.parse_args()

    LOG_PATH.write_text('', encoding='utf-8')
    log_progress('Processing started')

    print_devices(devices)
    if args.device is None:
        if sys.stdin is not None and sys.stdin.isatty():
            device = prompt_for_device(devices)
        else:
            device = resolve_device('auto')
    else:
        device = resolve_device(args.device)
    log_progress(f'Using device: {device}')
    print_available_models(args.backend, args.model, backends.AVAILABLE_MODELS)

    results = []
    log_progress(f'Scanning for audio files under {ROOT}')
    files = list(iter_audio_files(ROOT))
    if args.file is not None:
        requested_path = (ROOT / args.file).resolve()
        if ROOT not in requested_path.parents or requested_path not in files:
            parser.error('--file must name a supported audio file under input/.')
        files = [requested_path]
    log_progress(f'Found {len(files)} files to process')
    clear_transcripts()

    if not files:
        end_time = datetime.now()
        summary = f'[{start_time.strftime("%Y-%m-%d %H:%M:%S")}] Processed root: {ROOT}\nFiles found: 0\nSuccessfully updated: 0\nFailed: 0\nSkipped: 0\nNo supported sound files were found.\n[{end_time.strftime("%Y-%m-%d %H:%M:%S")}] Duration: {(end_time - start_time).total_seconds():.2f}s'
        with LOG_PATH.open('a', encoding='utf-8') as log:
            log.write(summary + '\n')
        print(summary)
        return

    log_progress(
        f'Loading {args.backend} model "{args.model}" '
        f'(first run may download several GB)'
    )
    try:
        with progress_heartbeat(f'Loading {args.backend} model "{args.model}"'):
            model = backends.load_model(args.model, args.backend, device)
    except Exception as load_error:
        log_progress(f'Failed to load {args.backend} model "{args.model}": {load_error}')
        raise
    log_progress(f'Model loaded; starting transcription with {args.backend} model "{args.model}"')
    fallback_model = None
    for idx, path in enumerate(files, 1):
        file_start = datetime.now()
        log_progress(f'[{idx}/{len(files)}] {path.name} — processing started using model "{args.model}"')
        try:
            transcription_path = path
            used_original_fallback = False
            if not args.no_vocal_separation:
                try:
                    transcription_path = separate_vocals(
                        path,
                        device,
                        TEMP_DIR,
                        use_mp3=args.demucs_mp3,
                        mp3_bitrate=args.demucs_mp3_bitrate,
                    )
                except Exception as separation_error:
                    log_progress(
                        f'{path.name} — WARNING: {separation_error}; '
                        'falling back to original audio'
                    )

            segments, info = backends.transcribe_audio(
                model, transcription_path, args.language, path.name, args.backend
            )
            candidate_segments, _, _ = filter_transcript_segments(segments, allow_promotions=args.keep_promotions)
            if (
                transcription_path != path
                and (
                    not candidate_segments
                    or float(candidate_segments[0].start) > args.opening_threshold
                )
            ):
                used_original_fallback = True
                if candidate_segments:
                    fallback_message = (
                        f'{path.name} — first usable vocal segment starts at '
                        f'{candidate_segments[0].start:.1f}s; retrying transcription '
                        'to recover the opening'
                    )
                else:
                    fallback_message = (
                        f'{path.name} — no usable vocal segments; retrying transcription'
                    )
                retry_backend = args.backend
                retry_model_name = args.model
                retry_model = model
                retry_audio_path = path
                if args.fallback_viet_lyrics:
                    retry_backend = 'viet-lyrics'
                    retry_model_name = args.fallback_viet_lyrics_model
                    retry_audio_path = transcription_path
                    if fallback_model is None:
                        log_progress(
                            f'Loading fallback viet-lyrics model "{retry_model_name}" '
                            '(first run may download several GB)'
                        )
                        with progress_heartbeat(f'Loading fallback viet-lyrics model "{retry_model_name}"'):
                            fallback_model = backends.load_model(retry_model_name, retry_backend, device)
                    retry_model = fallback_model
                    fallback_message += (
                        f' on the separated vocals using viet-lyrics model "{retry_model_name}"'
                    )
                else:
                    fallback_message += ' on the original audio'
                log_progress(fallback_message)
                try:
                    segments, info = backends.transcribe_audio(
                        retry_model, retry_audio_path, args.language, path.name, retry_backend
                    )
                except Exception as fallback_error:
                    # The retry audio can occasionally fail to decode (e.g. an
                    # unusual WAV encoding faster-whisper's loader can't parse)
                    # even though the first pass read it fine. Keep the transcript
                    # already produced instead of losing it entirely.
                    used_original_fallback = False
                    log_progress(
                        f'{path.name} — WARNING: could not retry on {retry_audio_path.name} '
                        f'({fallback_error}); keeping earlier transcript'
                    )
            segments, removed_promotions, removed_duplicates = filter_transcript_segments(
                segments, allow_promotions=args.keep_promotions
            )
            if removed_promotions or removed_duplicates:
                log_progress(
                    f'{path.name} — removed {removed_promotions} promotional and '
                    f'{removed_duplicates} duplicate segment(s)'
                )
            transcript = '\n'.join(
                segment.text.strip() for segment in segments if segment.text.strip()
            ).strip()
            if not transcript:
                status = 'Failed: no transcript produced'
                results.append((path.name, status))
                log_progress(f'[{idx}/{len(files)}] {path.name} — {status}')
                continue

            language = id3_language(args.language or info.language)
            log_progress(f'{path.name} — writing USLT/SYLT metadata and transcript')
            write_lyrics_to_file(path, transcript, segments, language)

            # Write transcript as LRC-style synced lyrics, matching the mp3's SYLT frame exactly.
            sylt_entries = build_sylt_entries(segments)
            transcript_output = format_sylt_as_lrc(sylt_entries) if sylt_entries else transcript
            relative_path = path.relative_to(ROOT).with_suffix('.txt')
            transcript_path = TRANSCRIPTS_DIR / relative_path
            transcript_path.parent.mkdir(parents=True, exist_ok=True)
            transcript_path.write_text(transcript_output, encoding='utf-8')

            if used_original_fallback:
                suffix = '_fallback' if args.fallback_viet_lyrics else '_original'
                original_relative = path.relative_to(ROOT).with_name(f'{path.stem}{suffix}.txt')
                original_transcript_path = TRANSCRIPTS_DIR / original_relative
                original_transcript_path.parent.mkdir(parents=True, exist_ok=True)
                original_transcript_path.write_text(transcript_output, encoding='utf-8')
                log_progress(f'{path.name} — wrote retry-pass fallback transcript: {original_transcript_path.name}')

            status = f'Success (USLT + SYLT embedded, lang={language})'
            results.append((path.name, status))
            elapsed = (datetime.now() - file_start).total_seconds()
            log_progress(f'[{idx}/{len(files)}] {path.name} — {status} ({elapsed:.1f}s)')
        except Exception as exc:
            status = f'Failed: {exc}'
            results.append((path.name, status))
            elapsed = (datetime.now() - file_start).total_seconds()
            log_progress(f'[{idx}/{len(files)}] {path.name} — {status} ({elapsed:.1f}s)')

    end_time = datetime.now()
    log_progress('Generating summary')
    summary_lines = [
        f'[{start_time.strftime("%Y-%m-%d %H:%M:%S")}] Processed root: {ROOT}',
        f'Files found: {len(files)}',
        f'Successfully updated: {sum(1 for _, status in results if status.startswith("Success"))}',
        f'Failed: {sum(1 for _, status in results if status.startswith("Failed"))}',
        f'Skipped: {sum(1 for _, status in results if status.startswith("Skipped"))}',
        f'Duration: {(end_time - start_time).total_seconds():.2f}s',
        f'[{end_time.strftime("%Y-%m-%d %H:%M:%S")}] Completed',
        '',
        'Per file:'
    ]
    for name, status in results:
        summary_lines.append(f'- {name} — {status}')

    summary = '\n'.join(summary_lines) + '\n'
    with LOG_PATH.open('a', encoding='utf-8') as log:
        log.write('\n' + summary)
    print(summary)


if __name__ == '__main__':
    main()
