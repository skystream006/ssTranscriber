"""Embeds USLT/SYLT lyrics metadata into input/ audio files from output/transcripts/*.txt.

Run this manually after process_audio_folder.py has finished producing transcripts, to
(re-)embed lyrics into the audio files without re-running transcription. For each audio
file under input/, it looks for a transcript file with the same relative path/filename
(stem) under output/transcripts/ and embeds that transcript's USLT (plain) and SYLT
(synced) tags into the audio file.

The "_initial.txt"/"_original.txt"/"_fallback.txt" pass-specific sibling files
written by process_audio_folder.py never match an audio filename, so they are naturally
skipped.
"""
import argparse
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from transcribe_common import (
    REPO_ROOT,
    ROOT,
    TRANSCRIPTS_DIR,
    colorize,
    id3_language,
    iter_audio_files,
    read_transcript_file,
    write_lyrics_to_file,
)

LOG_PATH = REPO_ROOT / 'output' / 'embed_lyrics_log.txt'


def timestamp():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def _status_color(status: str):
    if status.startswith('Success'):
        return 'green'
    if status.startswith('Failed'):
        return 'red'
    if status.startswith('Skipped'):
        return 'yellow'
    return None


def log_line(message: str, color: str = None):
    line = f'[{timestamp()}] {message}'
    print(colorize(line, color, bold=True) if color else line, flush=True)
    with LOG_PATH.open('a', encoding='utf-8') as log:
        log.write(line + '\n')


def main():
    parser = argparse.ArgumentParser(
        description='Embed USLT/SYLT lyrics metadata from output/transcripts/ into input/ audio files.'
    )
    parser.add_argument(
        '--language',
        default=None,
        help=(
            'ISO 639-1 language code to tag (e.g. vi, en). Overrides the language recorded '
            'in each transcript file; defaults to that recorded language, or "und" if absent.'
        ),
    )
    parser.add_argument(
        '--file',
        type=Path,
        default=None,
        help='Only process one audio file under input/ (relative path) instead of the full folder.',
    )
    args = parser.parse_args()

    LOG_PATH.write_text('', encoding='utf-8')
    started = datetime.now()

    log_line('Lyrics embedding started', 'cyan')
    files = list(iter_audio_files(ROOT))
    if args.file is not None:
        requested_path = (ROOT / args.file).resolve()
        if ROOT not in requested_path.parents or requested_path not in files:
            parser.error('--file must name a supported audio file under input/.')
        files = [requested_path]
    log_line(f'Found {len(files)} audio file(s) to check under {ROOT}', 'cyan')

    results = []
    for index, path in enumerate(files, 1):
        relative_path = path.relative_to(ROOT).with_suffix('.txt')
        transcript_path = TRANSCRIPTS_DIR / relative_path
        if not transcript_path.is_file():
            status = f'Skipped: no matching transcript ({transcript_path.name})'
            results.append((path, status))
            log_line(f'[{index}/{len(files)}] {path.name} — {status}', _status_color(status))
            continue
        log_line(f'[{index}/{len(files)}] {path.name} — found matching transcript {transcript_path.name}')

        recorded_language, sylt_entries, transcript = read_transcript_file(transcript_path)
        if not transcript:
            status = 'Skipped: transcript file is empty'
            results.append((path, status))
            log_line(f'[{index}/{len(files)}] {path.name} — {status}', _status_color(status))
            continue

        language = id3_language(args.language or recorded_language)
        segments = [
            SimpleNamespace(start=start_ms / 1000.0, end=start_ms / 1000.0, text=text)
            for text, start_ms in sylt_entries
        ]
        try:
            write_lyrics_to_file(path, transcript, segments, language)
            status = f'Success (USLT + SYLT embedded, lang={language})'
        except Exception as exc:
            status = f'Failed: {exc}'
        results.append((path, status))
        log_line(f'[{index}/{len(files)}] {path.name} — {status}', _status_color(status))

    finished = datetime.now()
    embedded_count = sum(1 for _, status in results if status.startswith('Success'))
    skipped_count = sum(1 for _, status in results if status.startswith('Skipped'))
    failed_count = sum(1 for _, status in results if status.startswith('Failed'))

    log_line(
        f'Embedded: {embedded_count} | Skipped: {skipped_count} | Failed: {failed_count} '
        f'({(finished - started).total_seconds():.2f}s)',
        'red' if failed_count else 'green',
    )
    if failed_count:
        for path, status in results:
            if status.startswith('Failed'):
                log_line(f'  {path.relative_to(ROOT)} — {status}', 'red')

    summary = [
        f'[{started.strftime("%Y-%m-%d %H:%M:%S")}] Embed lyrics from {TRANSCRIPTS_DIR}',
        f'Processed root: {ROOT}',
        f'Files checked: {len(files)}',
        f'Embedded: {embedded_count}',
        f'Skipped: {skipped_count}',
        f'Failed: {failed_count}',
        f'Duration: {(finished - started).total_seconds():.2f}s',
        f'[{finished.strftime("%Y-%m-%d %H:%M:%S")}] Completed',
        '',
        'Per file:',
    ]
    summary.extend(f'- {path.relative_to(ROOT)} — {status}' for path, status in results)
    with LOG_PATH.open('a', encoding='utf-8') as log:
        log.write('\n' + '\n'.join(summary) + '\n')
    print(f'[{timestamp()}] Log written to {LOG_PATH}')


if __name__ == '__main__':
    main()

