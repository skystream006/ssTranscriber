---
agent: agent
description: "Transcribe every sound file in a folder tree, then embed USLT and SYLT lyrics into each file as a separate step."
---

<!--
Prompt text to paste in Copilot Agent mode:

Use the instructions in .github/prompts/transcribe-sound-files.prompt.md and process the 'input' folder. Recursively scan all subfolders, transcribe every supported sound file (writing transcripts to output/transcripts/), then run the embed step to embed both USLT and SYLT lyrics metadata into each file, and provide a per-file success/failure summary and totals.
-->

You are working with a folder of sound files. Your job is to recursively scan the provided folder and
all subfolders, transcribe each supported audio file (via `process_audio_folder.py`), and then embed
the transcription into the file as ID3 lyrics metadata using both USLT and SYLT frames (via the
separate `embed_lyrics.py` step).

Requirements:
- Search the provided root folder recursively, including all nested subfolders.
- Process every supported sound file found under that root.
- Skip unsupported files, hidden/system files, or non-audio files.
- For each audio file, generate and embed:
  - USLT (Unsynchronized Lyrics): plain-text lyrics block suitable for a full transcription or lyrics summary.
  - SYLT (Synchronized Lyrics): timestamped, scrolling lyrics with start/end times for each phrase or segment.
- Preserve the original audio data and do not destroy the file.
- If a file already contains lyrics metadata, replace or overwrite the relevant USLT/SYLT tags with the newly generated transcription.
- If a file cannot be transcribed, record the reason and continue with the rest.

Supported audio file types:
- .mp3, .wav, .flac, .m4a, .aac, .ogg, .opus, .wma

Workflow:
1. Confirm the target root folder path.
2. Enumerate all supported audio files under that folder, including subfolders.
3. Run `process_audio_folder.py` to transcribe every file:
   - It isolates the vocal stem with Demucs, transcribes the spoken content, splits the transcript
     into meaningful lyric segments, and writes each result as a synced `[mm:ss.xx]text` transcript
     under `output/transcripts/`, matching the audio file's relative path/filename (stem).
   - This step does not modify the audio files themselves.
4. Run `embed_lyrics.py` to embed the transcripts into the audio files:
   - It matches each audio file under `input/` to a transcript with the same filename (stem) under
     `output/transcripts/`.
   - It creates a USLT text block containing the full transcription text.
   - It creates a SYLT track with timestamped segments, using the timestamps recorded in the transcript.
   - It writes both lyric frames back into the audio file metadata and validates that they were written.
5. After processing all files, provide a concise summary:
   - Number of files found
   - Number successfully transcribed and embedded
   - Number skipped or failed
   - List any files that failed and the reason

Important implementation constraints:
- Use Demucs vocal separation followed by faster-whisper `large-v3`, beam search, and word timestamps for accurate local song transcription.
- Keep timestamps aligned with the actual audio timing.
- Do not add random or synthetic timestamps; use the transcript timing data.
- Maintain the file format integrity of each audio file.
- Write clear, readable metadata; avoid malformed tags.
- Use durable error handling and continue processing even if one file fails.

Output expectations:
- Begin by stating the root folder reviewed.
- Show a per-file result for each audio file processed.
- Report success/failure status clearly.
- If the folder is empty or contains no supported sound files, say so explicitly.

Example result format:
- Processed root: /path/to/folder
- Files found: 12
- Successfully updated: 10
- Failed: 2
- Skipped: 0

Per file:
- file1.mp3 — Success (USLT + SYLT embedded)
- file2.wav — Failed: transcription unavailable

Do not stop after the first file. The agent must continue through the entire folder tree and act on each eligible sound file in sequence.
