import tempfile
import unittest
from pathlib import Path

from transcribe_common import run_transformers_pipeline_transcription


class FakePipeline:
    def __init__(self, text):
        self.text = text

    def __call__(self, path, **kwargs):
        return {
            'chunks': [
                {
                    'text': self.text,
                    'timestamp': (0.0, 6.0),
                }
            ]
        }


class TranscriptSegmentationTests(unittest.TestCase):
    def transcribe(self, text, split_on_capitalized_phrases):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'audio.wav'
            path.write_bytes(b'audio')
            segments, _ = run_transformers_pipeline_transcription(
                FakePipeline(text),
                path,
                'vi',
                'test',
                max_words_per_line=100,
                max_line_duration=60.0,
                split_on_capitalized_phrases=split_on_capitalized_phrases,
            )
        return [segment.text for segment in segments]

    def test_capitalized_phrases_start_new_viet_lyrics_lines(self):
        segments = self.transcribe(
            'Có lẽ cả hai từng nghĩ tình yêu chẳng khó Chỉ cần ta có niềm tin '
            'Rồi bỗng một ngày không nắng sầuTình yêu là thế',
            split_on_capitalized_phrases=True,
        )

        self.assertEqual(
            segments,
            [
                'Có lẽ cả hai từng nghĩ tình yêu chẳng khó',
                'Chỉ cần ta có niềm tin',
                'Rồi bỗng một ngày không nắng sầu',
                'Tình yêu là thế',
            ],
        )

    def test_isolated_uppercase_typo_does_not_start_a_new_line(self):
        text = 'Rồi khi tỉnh giấC sự thật đau đến phũ phàng'

        self.assertEqual(
            self.transcribe(text, split_on_capitalized_phrases=True),
            [text],
        )

    def test_capitalized_phrase_splitting_is_backend_specific(self):
        text = 'Có lẽ cả hai từng nghĩ tình yêu chẳng khó Chỉ cần ta có niềm tin'

        self.assertEqual(
            self.transcribe(text, split_on_capitalized_phrases=False),
            [text],
        )


if __name__ == '__main__':
    unittest.main()
