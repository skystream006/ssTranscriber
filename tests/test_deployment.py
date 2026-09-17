"""Container path/binding regressions, also run in the local test suite."""
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

import web_api


class DeploymentTests(unittest.TestCase):
    def test_default_api_paths_stay_repository_relative_without_loading_torch(self):
        env = dict(os.environ)
        env.pop('SSTRANSCRIBER_WORK_DIR', None)
        result = subprocess.run(
            [sys.executable, '-c',
             'import sys, web_api; '
             'assert web_api.WORK_DIR == web_api.REPO_ROOT; '
             'assert web_api.INPUT_DIR == web_api.REPO_ROOT / "input"; '
             'assert web_api.ENDPOINT_CONFIG_PATH == web_api.REPO_ROOT / ".endpoint-config.json"; '
             'assert "torch" not in sys.modules'],
            cwd=web_api.REPO_ROOT, env=env, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_custom_root_is_shared_by_api_pipeline_and_utilities(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / 'data'
            code = '''
import json, sys
from types import SimpleNamespace
import web_api
assert 'torch' not in sys.modules
sys.modules['torch'] = SimpleNamespace()
import transcribe_common, embed_lyrics, clear_embedded_lyrics
assert web_api.INPUT_DIR == transcribe_common.ROOT == clear_embedded_lyrics.INPUT_ROOT
assert web_api.OUTPUT_DIR == transcribe_common.TRANSCRIPTS_DIR.parent
assert web_api.TEMP_DIR == transcribe_common.TEMP_DIR
assert embed_lyrics.LOG_PATH.parent == clear_embedded_lyrics.LOG_PATH.parent == web_api.OUTPUT_DIR
assert web_api.WEB_DIST == web_api.REPO_ROOT / 'webui' / 'dist'
print(json.dumps({'root': str(web_api.WORK_DIR), 'settings': str(web_api.ENDPOINT_CONFIG_PATH)}))
'''
            result = subprocess.run(
                [sys.executable, '-c', code], cwd=web_api.REPO_ROOT,
                env={**os.environ, 'SSTRANSCRIBER_WORK_DIR': str(root)},
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout), {
                'root': str(root), 'settings': str(root / '.endpoint-config.json'),
            })

    def test_readiness_does_not_query_gpu(self):
        with patch.object(web_api, 'gpu_health', side_effect=AssertionError('GPU queried')):
            client = TestClient(web_api.app)
            self.addCleanup(client.close)
            response = client.get('/api/ready')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {'status': 'ok'})

    def test_lifespan_initializes_empty_data_mount_and_persists_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'new-data'
            with patch.multiple(web_api, INPUT_DIR=root / 'input', OUTPUT_DIR=root / 'output',
                                SONGS_DIR=root / 'output' / 'songs', TEMP_DIR=root / 'temp',
                                ENDPOINT_CONFIG_PATH=root / '.endpoint-config.json'):
                with TestClient(web_api.app) as client:
                    for path in ('input', 'output/songs', 'temp'):
                        self.assertTrue((root / path).is_dir())
                    settings = client.get('/api/endpoint-config').json()
                    settings['language'] = 'vi'
                    self.assertEqual(client.put('/api/endpoint-config', json=settings).status_code, 200)
                with TestClient(web_api.app) as client:
                    self.assertEqual(client.get('/api/endpoint-config').json()['language'], 'vi')

    def test_library_and_upload_jobs_pass_the_correct_root_to_subprocesses(self):
        async def exercise(root):
            for work_dir in (None, root / 'isolated-upload'):
                with self.subTest(work_dir=work_dir):
                    process = SimpleNamespace(
                        stdout=SimpleNamespace(readline=AsyncMock(return_value=b'')),
                        wait=AsyncMock(return_value=0), returncode=0,
                    )
                    with patch.object(web_api, 'WORK_DIR', root), \
                            patch.object(web_api, 'job_lock', asyncio.Lock()), \
                            patch.object(web_api.asyncio, 'create_subprocess_exec',
                                         AsyncMock(return_value=process)) as spawn:
                        job = web_api.Job(web_api.JobRequest(), work_dir=work_dir, upload_name='song.mp3')
                        await web_api.run_job(job)
                        self.assertEqual(spawn.call_args.kwargs['env']['SSTRANSCRIBER_WORK_DIR'],
                                         str(work_dir if work_dir is not None else root))
                        self.assertEqual(spawn.call_args.kwargs['cwd'], web_api.REPO_ROOT)
                        self.assertEqual(job.return_code, 0)

        with tempfile.TemporaryDirectory() as directory:
            asyncio.run(exercise(Path(directory)))

    def test_host_flag_is_available_for_local_and_container_startup(self):
        result = subprocess.run(
            [sys.executable, str(web_api.REPO_ROOT / 'web_api.py'), '--help'],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--host', result.stdout)
        self.assertIn('--reload', result.stdout)


if __name__ == '__main__':
    unittest.main()