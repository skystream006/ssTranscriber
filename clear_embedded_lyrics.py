import argparse
import sys
from datetime import datetime
from pathlib import Path

from mutagen import File, MutagenError

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass

REPO_ROOT = Path(__file__).resolve().parent
INPUT_ROOT = REPO_ROOT / 'input'
LOG_PATH = REPO_ROOT / 'output' / 'clear_metadata_log.txt'
SUPPORTED = {'.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg', '.opus', '.wma'}


def timestamp():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def iter_audio_files(root: Path):
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            yield path


def clear_metadata(path: Path, dry_run: bool):
    try:
        audio = File(path)
    except MutagenError as exc:
        return f'Failed: could not read metadata ({exc})'

    if audio is None:
        return 'Skipped: unsupported metadata format'
    if not audio.tags:
        return 'Skipped: no metadata'
    tag_count = len(audio.tags)

    if dry_run:
        return f'Would remove all metadata ({tag_count} tag item(s))'

    try:
        audio.delete()
        return f'Removed all metadata ({tag_count} tag item(s))'
    except (MutagenError, OSError) as exc:
        return f'Failed: could not remove metadata ({exc})'


def main():
    parser = argparse.ArgumentParser(
        description='Remove all embedded metadata from audio files while preserving audio data.'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Report metadata that would be removed without changing files.',
    )
    args = parser.parse_args()

    INPUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    files = list(iter_audio_files(INPUT_ROOT))
    started = datetime.now()
    results = []

    print(f'[{timestamp()}] Scanning {INPUT_ROOT}')
    print(f'[{timestamp()}] Found {len(files)} supported audio file(s)')
    for index, path in enumerate(files, 1):
        status = clear_metadata(path, args.dry_run)
        results.append((path, status))
        print(f'[{timestamp()}] [{index}/{len(files)}] {path.name} — {status}', flush=True)

    removed = sum(1 for _, status in results if status.startswith('Removed all metadata'))
    would_remove = sum(1 for _, status in results if status.startswith('Would remove all metadata'))
    failed = sum(1 for _, status in results if status.startswith('Failed:'))
    skipped = sum(1 for _, status in results if status.startswith('Skipped:'))
    finished = datetime.now()

    summary = [
        f'[{started.strftime("%Y-%m-%d %H:%M:%S")}] Clear all embedded metadata',
        f'Processed root: {INPUT_ROOT}',
        f'Files found: {len(files)}',
        f'Removed: {removed}',
        f'Would remove: {would_remove}',
        f'Failed: {failed}',
        f'Skipped: {skipped}',
        f'Duration: {(finished - started).total_seconds():.2f}s',
        f'[{finished.strftime("%Y-%m-%d %H:%M:%S")}] Completed',
        '',
        'Per file:',
    ]
    summary.extend(f'- {path.relative_to(INPUT_ROOT)} — {status}' for path, status in results)
    LOG_PATH.write_text('\n'.join(summary) + '\n', encoding='utf-8')
    print(f'[{timestamp()}] Log written to {LOG_PATH}')


if __name__ == '__main__':
    main()
