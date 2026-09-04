import argparse
import json
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import backends
from transcribe_common import (
    LOG_PATH,
    REPO_ROOT,
    ROOT,
    TEMP_DIR,
    TRANSCRIPTS_DIR,
    apply_lyrics_mode,
    archive_demucs_results,
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
    write_transcript_file,
)


def _start_viet_lyrics_worker(model_name: str, device: str):
    # Runs in its own process; see viet_lyrics_worker.py for why this must
    # not share a process with the primary faster-whisper (CTranslate2) model.
    return subprocess.Popen(
        [sys.executable, str(REPO_ROOT / 'viet_lyrics_worker.py'), '--model', model_name, '--device', device],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace',
        bufsize=1,
    )


def _transcribe_with_viet_lyrics_worker(worker, audio_path: Path, language, label: str, lyrics_text=None):
    result_path = TEMP_DIR / f'viet_lyrics_result_{uuid.uuid4().hex}.json'
    request = {
        'audio_path': str(audio_path),
        'language': language,
        'label': label,
        'lyrics_text': lyrics_text,
        'result_path': str(result_path),
    }
    worker.stdin.write(json.dumps(request) + '\n')
    worker.stdin.flush()
    while True:
        line = worker.stdout.readline()
        if line == '':
            raise RuntimeError('viet-lyrics fallback worker exited unexpectedly')
        line = line.rstrip('\n')
        if line.startswith('__ASR_RESULT__:'):
            status = line[len('__ASR_RESULT__:'):]
            break
        print(line, flush=True)  # pass through the worker's own log output
    try:
        if status.startswith('ERROR:'):
            raise RuntimeError(status[len('ERROR:'):])
        payload = json.loads(result_path.read_text(encoding='utf-8'))
    finally:
        result_path.unlink(missing_ok=True)
    segments = [
        SimpleNamespace(start=float(item['start']), end=float(item['end']), text=item['text'])
        for item in payload['segments']
    ]
    info = SimpleNamespace(language=payload.get('language'))
    return segments, info


def _load_lyrics_for_audio(audio_path: Path):
    lyrics_path = ROOT / 'lyrics' / f'{audio_path.stem}.txt'
    if not lyrics_path.is_file():
        return None
    lyrics_text = lyrics_path.read_text(encoding='utf-8-sig').strip()
    if not lyrics_text:
        return None
    log_progress(f'{audio_path.name} — using lyric prompt: {lyrics_path.name}')
    return lyrics_text


def _write_model_options(pass_number, pass_name, backend, model_name, device, language):
    manifest_path = TRANSCRIPTS_DIR / f'__{pass_number}{pass_name}_model_options.json'
    if manifest_path.exists():
        return manifest_path
    payload = {
        'pass_number': pass_number,
        'pass_name': pass_name,
        'backend': backend,
        'model': model_name,
        'device': device,
        'language': language,
        'options': backends.get_options(backend),
    }
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    log_progress(f'Wrote model options: {manifest_path.name}')
    return manifest_path


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
        '--no-lyric-prompt',
        action='store_true',
        help='Ignore matching files under input/lyrics; this overrides --lyrics-mode.',
    )
    parser.add_argument(
        '--lyrics-mode',
        choices=('prompt', 'align', 'correct'),
        default='prompt',
        help=(
            'How matching lyrics assist transcription: prompt biases decoding, align maps '
            'authoritative lyric lines to ASR timing, and correct replaces ASR text while '
            'preserving its segment timing. Ignored with --no-lyric-prompt.'
        ),
    )
    parser.add_argument(
        '--save-previous-results',
        action='store_true',
        help=(
            'Preserve existing output/transcripts results by renaming the folder to '
            'transcripts_01, transcripts_02, and so on before processing.'
        ),
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
            'When the --opening-threshold retry triggers, run a 3rd transcription pass with the '
            'viet-lyrics backend (on the separated vocals) after the primary --backend/--model '
            'retry on the original audio, instead of stopping at that one retry.'
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
    print_available_models(args.backend, args.model, backends.get_models(args.backend))

    results = []
    log_progress(f'Scanning for audio files under {ROOT}')
    files = list(iter_audio_files(ROOT))
    if args.file is not None:
        requested_path = (ROOT / args.file).resolve()
        if ROOT not in requested_path.parents or requested_path not in files:
            parser.error('--file must name a supported audio file under input/.')
        files = [requested_path]
    log_progress(f'Found {len(files)} files to process')
    clear_transcripts(save_previous=args.save_previous_results)
    if args.save_previous_results:
        archive_demucs_results()

    if not files:
        end_time = datetime.now()
        summary = f'[{start_time.strftime("%Y-%m-%d %H:%M:%S")}] Processed root: {ROOT}\nFiles found: 0\nSuccessfully updated: 0\nFailed: 0\nSkipped: 0\nNo supported sound files were found.\n[{end_time.strftime("%Y-%m-%d %H:%M:%S")}] Duration: {(end_time - start_time).total_seconds():.2f}s'
        with LOG_PATH.open('a', encoding='utf-8') as log:
            log.write(summary + '\n')
        print(summary)
        return

    _write_model_options(
        1,
        'initial',
        args.backend,
        args.model,
        device,
        args.language,
    )

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
    fallback_worker = None
    for idx, path in enumerate(files, 1):
        file_start = datetime.now()
        log_progress(f'[{idx}/{len(files)}] {path.name} — processing started using model "{args.model}"')
        try:
            transcription_path = path
            used_original_fallback = False
            lyrics_text = None if args.no_lyric_prompt else _load_lyrics_for_audio(path)
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
                model,
                transcription_path,
                args.language,
                path.name,
                args.backend,
                lyrics_text=lyrics_text,
            )
            segments = apply_lyrics_mode(segments, lyrics_text, args.lyrics_mode)
            candidate_segments, _, _ = filter_transcript_segments(segments, allow_promotions=args.keep_promotions)

            # Persist the first-pass transcript immediately so a slow/crashing retry
            # (e.g. the viet-lyrics fallback worker) never loses it.
            relative_path = path.relative_to(ROOT).with_suffix('.txt')
            transcript_path = TRANSCRIPTS_DIR / relative_path
            transcript_path.parent.mkdir(parents=True, exist_ok=True)
            initial_transcript = '\n'.join(
                segment.text.strip() for segment in candidate_segments if segment.text.strip()
            ).strip()
            if initial_transcript:
                initial_sylt_entries = build_sylt_entries(candidate_segments)
                initial_transcript_output = (
                    format_sylt_as_lrc(initial_sylt_entries) if initial_sylt_entries else initial_transcript
                )
                initial_language = id3_language(args.language or info.language)
                write_transcript_file(transcript_path, initial_language, initial_transcript_output)
                # Keep the first pass's own result in a dedicated sibling file too, so a
                # later retry overwriting the main transcript never loses it entirely.
                initial_relative = path.relative_to(ROOT).with_name(f'{path.stem}_1initial.txttxt')
                initial_transcript_path = TRANSCRIPTS_DIR / initial_relative
                write_transcript_file(initial_transcript_path, initial_language, initial_transcript_output)
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
                log_progress(f'{fallback_message} on the original audio')
                try:
                    _write_model_options(
                        2,
                        'original',
                        args.backend,
                        args.model,
                        device,
                        args.language,
                    )
                    segments, info = backends.transcribe_audio(
                        model,
                        path,
                        args.language,
                        path.name,
                        args.backend,
                        lyrics_text=lyrics_text,
                    )
                    segments = apply_lyrics_mode(segments, lyrics_text, args.lyrics_mode)
                    # Persist this retry immediately so a crashing viet-lyrics fallback
                    # pass below never loses it, and keep it in a dedicated sibling file
                    # so it isn't lost once the fallback pass overwrites the main transcript.
                    retry_candidate_segments, _, _ = filter_transcript_segments(
                        segments, allow_promotions=args.keep_promotions
                    )
                    retry_sylt_entries = build_sylt_entries(retry_candidate_segments)
                    retry_transcript = '\n'.join(
                        segment.text.strip() for segment in retry_candidate_segments if segment.text.strip()
                    ).strip()
                    if retry_transcript:
                        retry_transcript_output = (
                            format_sylt_as_lrc(retry_sylt_entries) if retry_sylt_entries else retry_transcript
                        )
                        retry_language = id3_language(args.language or info.language)
                        write_transcript_file(transcript_path, retry_language, retry_transcript_output)
                        original_relative = path.relative_to(ROOT).with_name(f'{path.stem}_2original.txttxt')
                        original_transcript_path = TRANSCRIPTS_DIR / original_relative
                        write_transcript_file(original_transcript_path, retry_language, retry_transcript_output)
                except Exception as original_retry_error:
                    # The original file can occasionally fail to decode (e.g. an
                    # unusual WAV encoding faster-whisper's loader can't parse)
                    # even though Demucs read it fine. Keep the transcript already
                    # produced instead of losing it entirely.
                    used_original_fallback = False
                    log_progress(
                        f'{path.name} — WARNING: could not retry on {path.name} '
                        f'({original_retry_error}); keeping earlier transcript'
                    )

                if args.fallback_viet_lyrics:
                    # Runs after the original-audio retry above, so a fallback run is
                    # always the third pass (initial + original retry + viet-lyrics).
                    if fallback_worker is None:
                        _write_model_options(
                            3,
                            'fallback',
                            'viet-lyrics',
                            args.fallback_viet_lyrics_model,
                            device,
                            args.language,
                        )
                        log_progress(
                            f'Starting isolated viet-lyrics worker process for model '
                            f'"{args.fallback_viet_lyrics_model}" (first run may download several GB; '
                            'runs in its own process to avoid a cuDNN conflict with the primary '
                            'faster-whisper model)'
                        )
                        with progress_heartbeat(
                            f'Loading fallback viet-lyrics model "{args.fallback_viet_lyrics_model}"'
                        ):
                            fallback_worker = _start_viet_lyrics_worker(args.fallback_viet_lyrics_model, device)
                    log_progress(
                        f'{path.name} — retrying transcription on the separated vocals using '
                        f'viet-lyrics model "{args.fallback_viet_lyrics_model}" (isolated process)'
                    )
                    try:
                        segments, info = _transcribe_with_viet_lyrics_worker(
                            fallback_worker,
                            transcription_path,
                            args.language,
                            path.name,
                            lyrics_text=lyrics_text,
                        )
                        segments = apply_lyrics_mode(segments, lyrics_text, args.lyrics_mode)
                        # Persist this pass immediately too, applying the same
                        # --keep-promotions filtering as the other two passes.
                        fallback_candidate_segments, _, _ = filter_transcript_segments(
                            segments, allow_promotions=args.keep_promotions
                        )
                        fallback_sylt_entries = build_sylt_entries(fallback_candidate_segments)
                        fallback_pass_transcript = '\n'.join(
                            segment.text.strip() for segment in fallback_candidate_segments if segment.text.strip()
                        ).strip()
                        if fallback_pass_transcript:
                            write_transcript_file(
                                transcript_path,
                                id3_language(args.language or info.language),
                                format_sylt_as_lrc(fallback_sylt_entries)
                                if fallback_sylt_entries
                                else fallback_pass_transcript,
                            )
                    except Exception as fallback_error:
                        log_progress(
                            f'{path.name} — WARNING: viet-lyrics fallback failed on '
                            f'{transcription_path.name} ({fallback_error}); keeping earlier transcript'
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
            log_progress(f'{path.name} — writing transcript')

            # Write transcript as LRC-style synced lyrics, matching what write_lyrics_to_file
            # would embed as the mp3's SYLT frame. USLT/SYLT embedding itself is a separate,
            # manually-run step; see embed_lyrics.py.
            sylt_entries = build_sylt_entries(segments)
            transcript_output = format_sylt_as_lrc(sylt_entries) if sylt_entries else transcript
            write_transcript_file(transcript_path, language, transcript_output)

            if used_original_fallback and args.fallback_viet_lyrics:
                # The original-audio retry's own result was already saved as
                # "_2original.txttxt" above; this is the viet-lyrics fallback's result.
                fallback_relative = path.relative_to(ROOT).with_name(f'{path.stem}_3fallback.txttxt')
                fallback_transcript_path = TRANSCRIPTS_DIR / fallback_relative
                write_transcript_file(fallback_transcript_path, language, transcript_output)
                log_progress(f'{path.name} — wrote viet-lyrics fallback transcript: {fallback_transcript_path.name}')

            status = f'Success (transcript written, lang={language})'
            results.append((path.name, status))
            elapsed = (datetime.now() - file_start).total_seconds()
            log_progress(f'[{idx}/{len(files)}] {path.name} — {status} ({elapsed:.1f}s)')
        except Exception as exc:
            status = f'Failed: {exc}'
            results.append((path.name, status))
            elapsed = (datetime.now() - file_start).total_seconds()
            log_progress(f'[{idx}/{len(files)}] {path.name} — {status} ({elapsed:.1f}s)')

    if fallback_worker is not None:
        try:
            fallback_worker.stdin.close()
            fallback_worker.wait(timeout=10)
        except Exception:
            fallback_worker.kill()

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
