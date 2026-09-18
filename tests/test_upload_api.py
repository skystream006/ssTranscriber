"""HTTP + real pipeline/ID3 integration tests with only inference/separation stubbed.

Run: python -m unittest discover -s tests -v (requires httpx).
"""
import asyncio
import io
import json
import sys
import tempfile
import threading
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient
from mutagen.id3 import ID3

import web_api


class UploadAPITests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.seen_jobs = []
        original_command = web_api.build_command
        original_run_job = web_api.run_job

        def command(request):
            result = original_command(request)
            result[2] = str(Path(__file__).parent / 'fixtures' / 'pipeline_runner.py')
            return result

        async def run_job(job):
            self.seen_jobs.append(job)
            await original_run_job(job)

        for patcher in (
            patch.object(web_api, 'ENDPOINT_CONFIG_PATH', self.root / 'config.json'),
            patch.object(web_api, 'build_command', command),
            patch.object(web_api, 'run_job', run_job),
            patch.object(web_api, 'job_lock', asyncio.Lock()),
            patch.object(web_api, 'endpoint_jobs', {}),
            patch.object(web_api, 'jobs', {}),
            patch.object(web_api, 'TEMP_DIR', self.root),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        # Do not run the production lifespan's generated-file cleanup in tests.
        self.client = TestClient(web_api.app)
        self.addCleanup(self.client.close)

    def settings(self, **updates):
        settings = self.client.get('/api/endpoint-config').json()
        settings.update(updates)
        response = self.client.put('/api/endpoint-config', json=settings)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def upload(self, filename='song.mp3', content=b'original audio', **data):
        return self.client.post('/api/transcribe', files={'file': (filename, content, 'audio/mpeg')}, data=data)

    def assert_clean(self):
        self.assertFalse(list(self.root.glob('ss-transcriber-api-*')))

    def test_defaults_are_persistent_and_separate_from_job_settings(self):
        default = self.client.get('/api/endpoint-config').json()
        for excluded in ('file', 'use_lyrics', 'lyrics_mode', 'save_previous_results',
                         'language', 'demucs_mp3', 'demucs_mp3_bitrate', 'copy_no_vocals'):
            self.assertNotIn(excluded, default)
        saved = self.settings(vocal_separation=False, opening_threshold=7.5)
        self.assertEqual(saved, json.loads((self.root / 'config.json').read_text()))
        self.assertEqual(saved, self.client.get('/api/endpoint-config').json())
        self.assertTrue(web_api.JobRequest().vocal_separation)

    def test_invalid_defaults_are_rejected_without_overwriting(self):
        saved = self.settings()
        for update in (
            {'copy_no_vocals': True}, {'backend': 'unknown'}, {'opening_threshold': -1},
            {'language': 'en'}, {'demucs_mp3': True}, {'demucs_mp3_bitrate': 192},
            {'backend_options': {'NOT_AN_OPTION': True}},
            {'fallback_viet_lyrics_options': {'NOT_AN_OPTION': True}},
            {'save_previous_results': True}, {'lyrics_mode': 'prompt'},
        ):
            with self.subTest(update=update):
                response = self.client.put('/api/endpoint-config', json={**saved, **update})
                self.assertEqual(response.status_code, 422)
                self.assertEqual(saved, self.client.get('/api/endpoint-config').json())

    def test_legacy_defaults_load_without_request_specific_options(self):
        saved = self.settings(vocal_separation=False, opening_threshold=7.5)
        legacy = {**saved, 'language': 'vi', 'demucs_mp3': True,
                  'demucs_mp3_bitrate': 192, 'copy_no_vocals': True}
        (self.root / 'config.json').write_text(json.dumps(legacy))
        self.assertEqual(self.client.get('/api/endpoint-config').json(), saved)
        response = self.upload()
        self.assertEqual(response.status_code, 200, response.text)
        request = self.seen_jobs[-1].request
        self.assertIsNone(request.language)
        self.assertFalse(request.copy_no_vocals)
        self.assertFalse(request.demucs_mp3)
        self.assertFalse(request.vocal_separation)
        self.assertEqual(self.settings(), saved)
        self.assertEqual(json.loads((self.root / 'config.json').read_text()), saved)
        self.assert_clean()

    def test_invalid_saved_defaults_still_report_load_errors(self):
        for invalid in ({'unknown': True}, {'opening_threshold': -1}, [], 'invalid'):
            with self.subTest(invalid=invalid):
                (self.root / 'config.json').write_text(json.dumps(invalid))
                self.assertEqual(self.client.get('/api/endpoint-config').status_code, 500)

    def test_local_job_no_vocals_validation_is_unchanged(self):
        for settings in ({'copy_no_vocals': True},
                         {'copy_no_vocals': True, 'demucs_mp3': True, 'vocal_separation': False}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                web_api.JobRequest(**settings)
        request = web_api.JobRequest(copy_no_vocals=True, demucs_mp3=True, language='vi')
        self.assertTrue(request.copy_no_vocals)
        self.assertEqual(request.language, 'vi')

    def test_known_lyrics_automatically_align_and_embed(self):
        response = self.upload(filename='Có Tất Cả.mp3', lyrics='Một khúc hát\nÊm đềm')
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.headers['content-type'], 'application/octet-stream')
        self.assertIn('filename*=utf-8', response.headers['content-disposition'])
        tags = ID3(io.BytesIO(response.content))
        self.assertEqual(tags.getall('USLT')[0].text, 'Một khúc hát\nÊm đềm')
        self.assertEqual([text for text, _ in tags.getall('SYLT')[0].text], ['Một khúc hát', 'Êm đềm'])
        request = self.seen_jobs[0].request
        self.assertTrue(request.use_lyrics)
        self.assertEqual(request.lyrics_mode, 'align')
        self.assertFalse(request.save_previous_results)
        self.assertIsNone(request.file)
        listed = self.client.get('/api/endpoint-jobs').json()
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0]['filename'], 'Có Tất Cả.mp3')
        self.assertEqual(listed[0]['status'], 'completed')
        self.assertNotIn('logs', listed[0])
        self.assertNotIn('work_dir', listed[0])
        self.assertEqual(response.headers['x-job-id'], listed[0]['id'])
        detail = self.client.get(f"/api/endpoint-jobs/{listed[0]['id']}").json()
        self.assertTrue(detail['logs'])
        self.assertIsNotNone(detail['finished_at'])
        self.assertEqual(self.client.get('/api/jobs').json(), [])
        self.assert_clean()

    def test_absent_and_blank_lyrics_disable_known_lyrics(self):
        for data in ({}, {'lyrics': ' \n\ufeff '}):
            with self.subTest(data=data):
                response = self.upload(**data)
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(ID3(io.BytesIO(response.content)).getall('USLT')[0].text, 'A gentle melody')
                self.assertFalse(self.seen_jobs[-1].request.use_lyrics)
                self.assert_clean()

    def test_explicit_lyrics_modes(self):
        for mode in ('prompt', 'align', 'correct'):
            with self.subTest(mode=mode):
                response = self.upload(lyrics='Known words', lyrics_mode=mode)
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(self.seen_jobs[-1].request.lyrics_mode, mode)
                self.assert_clean()

    def test_language_override_is_job_specific(self):
        saved = self.settings()
        response = self.upload(language='EN')
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.seen_jobs[-1].request.language, 'en')
        tags = ID3(io.BytesIO(response.content))
        self.assertEqual(tags.getall('USLT')[0].lang, 'eng')
        detail = self.client.get(f'/api/endpoint-jobs/{self.seen_jobs[-1].id}').json()
        self.assertEqual(detail['request']['language'], 'en')
        self.assertEqual(self.client.get('/api/endpoint-config').json(), saved)
        self.assert_clean()

    def test_omitted_or_empty_language_uses_auto_detection(self):
        for data in ({}, {'language': ''}):
            with self.subTest(data=data):
                response = self.upload(**data)
                self.assertEqual(response.status_code, 200, response.text)
                self.assertIsNone(self.seen_jobs[-1].request.language)
                self.assert_clean()

    def test_invalid_language_format_is_rejected_before_processing(self):
        for language in ('English', 'en-US', 'e', '12', ' en ', 'auto'):
            with self.subTest(language=language):
                self.assertEqual(self.upload(language=language).status_code, 422)
                self.assert_clean()
        self.assertEqual(self.seen_jobs, [])

    def test_zip_contains_both_embedded_songs(self):
        response = self.upload(filename='My melody.mp3', lyrics='A gentle melody', NoVocals='true')
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.headers['content-type'], 'application/zip')
        with zipfile.ZipFile(io.BytesIO(response.content)) as bundle:
            self.assertEqual(bundle.namelist(), ['My melody.mp3', '[NoVocals] My melody.mp3'])
            for name in bundle.namelist():
                tags = ID3(io.BytesIO(bundle.read(name)))
                self.assertEqual(tags.getall('USLT')[0].text, 'A gentle melody')
                self.assertTrue(tags.getall('SYLT'))
        self.assert_clean()

    def test_no_vocals_is_job_specific_and_enables_prerequisites(self):
        for separate in (False, True):
            saved = self.settings(vocal_separation=separate)
            for data in ({'NoVocals': 'true'}, {'NoVocals': 'false'}, {}):
                with self.subTest(separate=separate, data=data):
                    enabled = data.get('NoVocals') == 'true'
                    response = self.upload(**data)
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertEqual(response.headers['content-type'],
                                     'application/zip' if enabled else 'application/octet-stream')
                    request = self.seen_jobs[-1].request
                    self.assertEqual(request.copy_no_vocals, enabled)
                    self.assertEqual(request.demucs_mp3, enabled)
                    self.assertEqual(request.vocal_separation, enabled or separate)
                    self.assertEqual(request.demucs_mp3_bitrate, 320)
                    detail = self.client.get(f'/api/endpoint-jobs/{self.seen_jobs[-1].id}').json()
                    self.assertEqual(detail['request']['copy_no_vocals'], enabled)
                    self.assertEqual(self.client.get('/api/endpoint-config').json(), saved)
                    self.assert_clean()

    def test_invalid_no_vocals_is_rejected_before_processing(self):
        for value in ('invalid', '2'):
            with self.subTest(value=value):
                self.assertEqual(self.upload(NoVocals=value).status_code, 422)
                self.assert_clean()
        self.assertEqual(self.seen_jobs, [])

    def test_viet_lyrics_fallback_override_is_job_specific(self):
        for default in (False, True):
            saved = self.settings(fallback_viet_lyrics=default)
            for data in ({'VietLyricsFallback': 'true'}, {'VietLyricsFallback': 'false'}, {}):
                with self.subTest(default=default, data=data):
                    enabled = data['VietLyricsFallback'] == 'true' if data else default
                    response = self.upload(**data)
                    self.assertEqual(response.status_code, 200, response.text)
                    request = self.seen_jobs[-1].request
                    self.assertEqual(request.fallback_viet_lyrics, enabled)
                    self.assertEqual('--fallback-viet-lyrics' in web_api.build_command(request), enabled)
                    self.assertEqual(request.fallback_viet_lyrics_model, saved['fallback_viet_lyrics_model'])
                    self.assertEqual(request.fallback_viet_lyrics_options, saved['fallback_viet_lyrics_options'])
                    detail = self.client.get(f'/api/endpoint-jobs/{self.seen_jobs[-1].id}').json()
                    self.assertEqual(detail['request']['fallback_viet_lyrics'], enabled)
                    self.assertEqual(self.client.get('/api/endpoint-config').json(), saved)
                    self.assert_clean()

    def test_viet_lyrics_fallback_combines_with_no_vocals_and_language(self):
        response = self.upload(VietLyricsFallback='true', NoVocals='true', language='EN')
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.headers['content-type'], 'application/zip')
        request = self.seen_jobs[-1].request
        self.assertTrue(request.fallback_viet_lyrics)
        self.assertTrue(request.copy_no_vocals)
        self.assertEqual(request.language, 'en')
        self.assert_clean()

    def test_invalid_viet_lyrics_fallback_is_rejected_before_processing(self):
        for value in ('invalid', '2'):
            with self.subTest(value=value):
                self.assertEqual(self.upload(VietLyricsFallback=value).status_code, 422)
                self.assert_clean()
        self.assertEqual(self.seen_jobs, [])

    def test_missing_requested_accompaniment_is_an_error(self):
        with patch.dict('os.environ', {'SSTRANSCRIBER_TEST_FAILURE': 'no-stems'}):
            response = self.upload(NoVocals='true')
        self.assertEqual(response.status_code, 500)
        self.assertIn('no-vocals', response.json()['detail'])
        detail = self.client.get(f'/api/endpoint-jobs/{self.seen_jobs[-1].id}').json()
        self.assertEqual(detail['status'], 'failed')
        self.assertIn('no-vocals', detail['logs'][-1])
        self.assert_clean()

    def test_pipeline_failures_never_return_unprocessed_audio(self):
        for failure in ('fail', 'empty', 'embed-fail'):
            with self.subTest(failure=failure), \
                    patch.dict('os.environ', {'SSTRANSCRIBER_TEST_FAILURE': failure}):
                response = self.upload()
                self.assertEqual(response.status_code, 500)
                self.assertEqual(self.seen_jobs[-1].status, 'failed')
                self.assertEqual(self.seen_jobs[-1].return_code, 1)
                self.assert_clean()

    def test_validation_and_empty_upload_cleanup(self):
        self.assertEqual(self.client.post('/api/transcribe').status_code, 422)
        for filename, content, data in (
            ('song.txt', b'not audio', {}), ('song.mp3', b'', {}),
            ('song.mp3', b'audio', {'lyrics_mode': 'invalid'}), ('bad:name.mp3', b'audio', {}),
        ):
            with self.subTest(filename=filename, data=data):
                self.assertEqual(self.upload(filename=filename, content=content, **data).status_code, 422)
                self.assert_clean()
        self.assertEqual(self.seen_jobs, [])

    def test_paths_and_reserved_stem_names_are_isolated(self):
        for filename in ('../../escape.mp3', 'C:\\fakepath\\vocals.mp3', 'no_vocals.mp3'):
            with self.subTest(filename=filename):
                response = self.upload(filename=filename)
                self.assertEqual(response.status_code, 200, response.text)
                job = self.seen_jobs[-1]
                self.assertEqual(job.upload_name, 'song.mp3')
                self.assertNotEqual(job.work_dir, web_api.REPO_ROOT)
                self.assertNotIn('../', response.headers['content-disposition'])
                self.assert_clean()

    def test_openapi_documents_multipart_and_downloads(self):
        schema = self.client.get('/openapi.json').json()
        operation = schema['paths']['/api/transcribe']['post']
        self.assertIn('multipart/form-data', operation['requestBody']['content'])
        self.assertIn('application/zip', operation['responses']['200']['content'])
        body_ref = operation['requestBody']['content']['multipart/form-data']['schema']['$ref']
        body = schema['components']['schemas'][body_ref.rsplit('/', 1)[-1]]
        self.assertEqual(body['properties']['NoVocals']['type'], 'boolean')
        self.assertIs(body['properties']['NoVocals']['default'], False)
        self.assertNotIn('NoVocals', body['required'])
        self.assertIn({'type': 'boolean'}, body['properties']['VietLyricsFallback']['anyOf'])
        self.assertNotIn('VietLyricsFallback', body['required'])

    def test_concurrent_uploads_queue_with_independent_settings_snapshots(self):
        self.settings(vocal_separation=False, opening_threshold=7.5)

        async def exercise():
            queued = asyncio.Event()
            original_run = web_api.run_job
            count = 0

            async def observe_queue(job):
                nonlocal count
                count += 1
                if count == 2:
                    queued.set()
                await original_run(job)

            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web_api.app), base_url='http://test') as client:
                await web_api.job_lock.acquire()
                with patch.object(web_api, 'run_job', observe_queue):
                    requests = [asyncio.create_task(client.post('/api/transcribe',
                                files={'file': ('same.mp3', b'audio', 'audio/mpeg')},
                                data={'lyrics': lyric, 'language': 'en'})) for lyric in ('First melody', 'Second melody')]
                    try:
                        await asyncio.wait_for(queued.wait(), timeout=10)
                        self.assertTrue(all(job.status == 'queued' and job.process is None for job in self.seen_jobs))
                        listed = (await client.get('/api/endpoint-jobs')).json()
                        self.assertEqual(len(listed), 2)
                        self.assertTrue(all(job['status'] == 'queued' and job['filename'] == 'same.mp3' for job in listed))
                        self.assertEqual((await client.get('/api/jobs')).json(), [])
                        settings = web_api.EndpointSettings(vocal_separation=True, opening_threshold=20)
                        saved = await client.put('/api/endpoint-config', json=settings.model_dump())
                        self.assertEqual(saved.status_code, 200)
                    finally:
                        web_api.job_lock.release()
                    responses = await asyncio.gather(*requests)
                for response, lyric in zip(responses, ('First melody', 'Second melody')):
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertEqual(ID3(io.BytesIO(response.content)).getall('USLT')[0].text, lyric)
                self.assertTrue(all(job.request.language == 'en' for job in self.seen_jobs))
                self.assertTrue(all(not job.request.vocal_separation and job.request.opening_threshold == 7.5
                                    for job in self.seen_jobs))
                self.assertNotEqual(self.seen_jobs[0].work_dir, self.seen_jobs[1].work_dir)
                self.assertLessEqual(self.seen_jobs[0].started_at, self.seen_jobs[1].started_at)

        asyncio.run(exercise())
        self.assert_clean()

    def test_running_upload_can_be_monitored_before_response_is_ready(self):
        async def exercise():
            started = asyncio.Event()
            release = asyncio.Event()

            async def hold_running(job):
                job.status = 'running'
                job.started_at = web_api.utc_now()
                job.logs.append('transcription 35%')
                started.set()
                await release.wait()
                job.status = 'failed'
                job.return_code = 1

            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web_api.app), base_url='http://test') as client:
                with patch.object(web_api, 'run_job', hold_running):
                    request = asyncio.create_task(client.post('/api/transcribe', files={'file': ('live.mp3', b'audio')}))
                    try:
                        await asyncio.wait_for(started.wait(), timeout=10)
                        self.assertFalse(request.done())
                        listed = (await client.get('/api/endpoint-jobs')).json()
                        self.assertEqual(listed[0]['status'], 'running')
                        self.assertEqual(listed[0]['filename'], 'live.mp3')
                        detail = (await client.get(f"/api/endpoint-jobs/{listed[0]['id']}")).json()
                        self.assertEqual(detail['logs'], ['transcription 35%'])
                        self.assertIsNone(detail['finished_at'])
                    finally:
                        release.set()
                        response = await request
                    self.assertEqual(response.status_code, 500)
                    self.assertEqual((await client.get('/api/endpoint-jobs')).json()[0]['status'], 'failed')

        asyncio.run(exercise())
        self.assert_clean()

    def test_endpoint_history_is_bounded_without_dropping_active_jobs(self):
        with patch.object(web_api, 'ENDPOINT_JOB_HISTORY_LIMIT', 2):
            active = web_api.Job(web_api.JobRequest())
            web_api.endpoint_jobs[active.id] = active
            for _ in range(3):
                job = web_api.Job(web_api.JobRequest())
                job.status = 'completed'
                web_api.endpoint_jobs[job.id] = job
            oldest_finished = list(web_api.endpoint_jobs)[1]
            web_api.prune_endpoint_jobs()
            self.assertIn(active.id, web_api.endpoint_jobs)
            self.assertNotIn(oldest_finished, web_api.endpoint_jobs)
            self.assertEqual(len(web_api.endpoint_jobs), 3)
            self.assertEqual(self.client.get('/api/endpoint-jobs/missing').status_code, 404)
            local = web_api.Job(web_api.JobRequest())
            web_api.jobs[local.id] = local
            self.assertEqual(self.client.get(f'/api/endpoint-jobs/{local.id}').status_code, 404)

    def test_cancelled_queued_upload_is_marked_cancelled(self):
        self.assertEqual(self.client.delete('/api/endpoint-jobs/missing').status_code, 404)
        local = web_api.Job(web_api.JobRequest())
        web_api.jobs[local.id] = local
        self.assertEqual(self.client.delete(f'/api/endpoint-jobs/{local.id}').status_code, 404)

        async def exercise():
            queued = asyncio.Event()
            original_run = web_api.run_job

            async def observe_queue(job):
                queued.set()
                await original_run(job)

            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web_api.app), base_url='http://test') as client:
                async with web_api.job_lock:
                    with patch.object(web_api, 'run_job', observe_queue):
                        request = asyncio.create_task(client.post('/api/transcribe', files={'file': ('cancel.mp3', b'audio')}))
                        try:
                            await asyncio.wait_for(queued.wait(), timeout=10)
                            job = next(iter(web_api.endpoint_jobs.values()))
                            response = await client.delete(f'/api/endpoint-jobs/{job.id}')
                            self.assertEqual(response.status_code, 202)
                            self.assertTrue(response.json()['cancel_requested'])
                            response = await asyncio.wait_for(request, timeout=10)
                            self.assertEqual(response.status_code, 409)
                            self.assertEqual(response.json()['detail'], 'Upload job cancelled')
                            self.assertIsNone(job.started_at)
                            self.assertIsNone(job.process)
                            self.assertEqual(job.status, 'cancelled')
                            self.assertIsNotNone(job.finished_at)
                            self.assertEqual((await client.delete(f'/api/endpoint-jobs/{job.id}')).status_code, 409)
                        finally:
                            request.cancel()
                            await asyncio.gather(request, return_exceptions=True)

        asyncio.run(exercise())
        self.assert_clean()

    def test_cancel_running_endpoint_job_terminates_process_and_releases_queue(self):
        async def exercise():
            started = asyncio.Event()
            processes = []
            original_spawn = asyncio.create_subprocess_exec

            async def spawn(*args, **kwargs):
                process = await original_spawn(
                    sys.executable, '-u', '-c', 'import threading; threading.Event().wait()', **kwargs,
                )
                processes.append(process)
                started.set()
                return process

            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web_api.app), base_url='http://test') as client:
                with patch.object(web_api.asyncio, 'create_subprocess_exec', spawn):
                    request = asyncio.create_task(client.post('/api/transcribe', files={'file': ('running.mp3', b'audio')}))
                    try:
                        await asyncio.wait_for(started.wait(), timeout=10)
                        job = next(iter(web_api.endpoint_jobs.values()))
                        self.assertEqual(job.status, 'running')
                        self.assertEqual((await client.delete(f'/api/endpoint-jobs/{job.id}')).status_code, 202)
                        response = await asyncio.wait_for(request, timeout=10)
                        self.assertEqual(response.status_code, 409)
                        self.assertEqual(job.status, 'cancelled')
                        self.assertIsNotNone(job.finished_at)
                        self.assertIsNotNone(processes[0].returncode)
                        self.assertIsNone(job.processing_task)
                        self.assertFalse(web_api.job_lock.locked())
                    finally:
                        request.cancel()
                        await asyncio.gather(request, return_exceptions=True)
                response = await client.post('/api/transcribe', files={'file': ('next.mp3', b'audio')})
                self.assertEqual(response.status_code, 200, response.text)
                finished = next(job for job in web_api.endpoint_jobs.values() if job.status == 'completed')
                self.assertEqual((await client.delete(f'/api/endpoint-jobs/{finished.id}')).status_code, 409)

        asyncio.run(exercise())
        self.assert_clean()

    def test_cancel_during_download_preparation_discards_result(self):
        async def exercise():
            preparing = asyncio.Event()
            release = threading.Event()
            loop = asyncio.get_running_loop()

            async def finish_processing(job):
                job.status = 'running'
                job.return_code = 0

            def prepare_result(work_dir, song, filename, copy_no_vocals):
                loop.call_soon_threadsafe(preparing.set)
                if not release.wait(timeout=10):
                    raise TimeoutError('Download preparation was not released')
                self.assertTrue(song.is_file())
                return song, filename, 'application/octet-stream'

            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web_api.app), base_url='http://test') as client:
                with patch.object(web_api, 'run_job', finish_processing), patch.object(web_api, 'upload_result', prepare_result):
                    request = asyncio.create_task(client.post('/api/transcribe', files={'file': ('packing.mp3', b'audio')}))
                    try:
                        await asyncio.wait_for(preparing.wait(), timeout=10)
                        job = next(iter(web_api.endpoint_jobs.values()))
                        for attempt in range(2):
                            self.assertEqual((await client.delete(f'/api/endpoint-jobs/{job.id}')).status_code, 202)
                        self.assertTrue(job.work_dir.is_dir())
                    finally:
                        release.set()
                        response = await asyncio.wait_for(request, timeout=10)
                    self.assertEqual(response.status_code, 409)
                    self.assertNotIn('content-disposition', response.headers)
                    self.assertEqual(job.status, 'cancelled')

        asyncio.run(exercise())
        self.assert_clean()

    def test_disconnected_queued_upload_is_marked_cancelled(self):
        async def exercise():
            queued = asyncio.Event()
            original_run = web_api.run_job

            async def observe_queue(job):
                queued.set()
                await original_run(job)

            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web_api.app), base_url='http://test') as client:
                await web_api.job_lock.acquire()
                try:
                    with patch.object(web_api, 'run_job', observe_queue):
                        request = asyncio.create_task(client.post('/api/transcribe', files={'file': ('cancel.mp3', b'audio')}))
                        await asyncio.wait_for(queued.wait(), timeout=10)
                        request.cancel()
                        with self.assertRaises(asyncio.CancelledError):
                            await request
                    listed = (await client.get('/api/endpoint-jobs')).json()
                    self.assertEqual(listed[0]['status'], 'cancelled')
                    self.assertIsNotNone(listed[0]['finished_at'])
                finally:
                    web_api.job_lock.release()

        asyncio.run(exercise())
        self.assert_clean()


if __name__ == '__main__':
    unittest.main()