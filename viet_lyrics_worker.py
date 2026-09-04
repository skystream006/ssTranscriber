"""Standalone worker process for the --fallback-viet-lyrics retry pass.

Runs the viet-lyrics (transformers/torch) backend in its own OS process so
its bundled cuDNN never shares an address space with the primary
faster-whisper (CTranslate2) model. Loading both in the same process
crashes on Windows with "Could not load symbol cudnnGetLibConfig. Error
code 127", because the two runtimes bundle different, ABI-incompatible
cuDNN builds that collide once both are loaded into that process.

Only backends.viet_lyrics_backend is imported here (never the `backends`
package's other modules), so faster_whisper/ctranslate2 never loads in this
process either.

Protocol: after startup (which loads the model once), reads one JSON object
per line from stdin: {"audio_path": str, "language": str|null, "label": str,
"result_path": str}. For each request it writes {"language": str, "segments":
[{"text", "start", "end"}, ...]} to result_path, then prints exactly one line
to stdout: "__ASR_RESULT__:OK" or "__ASR_RESULT__:ERROR:<message>". Any other
stdout output is normal logging and should be passed through by the caller.
"""
import argparse
import json
import sys
from pathlib import Path

from backends import viet_lyrics_backend

RESULT_PREFIX = '__ASR_RESULT__:'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', required=True)
    args = parser.parse_args()

    try:
        model = viet_lyrics_backend.load(args.model, args.device)
    except Exception as exc:
        print(f'{RESULT_PREFIX}ERROR:failed to load model: {exc}', flush=True)
        sys.exit(1)

    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        request = json.loads(raw_line)
        result_path = Path(request['result_path'])
        try:
            segments, info = viet_lyrics_backend.transcribe(
                model,
                Path(request['audio_path']),
                request.get('language'),
                request.get('label') or Path(request['audio_path']).name,
                lyrics_text=request.get('lyrics_text'),
            )
            payload = {
                'language': info.language,
                'segments': [
                    {'text': seg.text, 'start': seg.start, 'end': seg.end}
                    for seg in segments
                ],
            }
            result_path.write_text(json.dumps(payload), encoding='utf-8')
            print(f'{RESULT_PREFIX}OK', flush=True)
        except Exception as exc:
            print(f'{RESULT_PREFIX}ERROR:{exc}', flush=True)


if __name__ == '__main__':
    main()
