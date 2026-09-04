import { useEffect, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import { flushSync } from 'react-dom'
import {
  Activity,
  Archive,
  AudioLines,
  Ban,
  ChevronDown,
  CircleCheck,
  CircleX,
  Clock3,
  Cpu,
  FileAudio,
  FileText,
  FolderOpen,
  Gauge,
  History,
  LoaderCircle,
  Music2,
  Palette,
  Play,
  RefreshCw,
  Square,
  TerminalSquare,
  WandSparkles,
} from 'lucide-react'

type BackendConfig = {
  models: string[]
  default_model: string
  options: Record<string, unknown>
}

type AppConfig = {
  backends: Record<string, BackendConfig>
  devices: { value: string; label: string }[]
  lyrics_modes: string[]
}

type JobStatus = 'queued' | 'running' | 'completed' | 'failed' | 'cancelled'
type Theme = 'light' | 'dark' | 'royal-blue' | 'royal-purple' | 'black' | 'yellow'
type ViewTransitionDocument = Document & {
  startViewTransition?: (update: () => void) => { ready: Promise<void> }
}

type JobRequest = {
  file: string | null
  backend: string
  model: string
  device: string
  language: string | null
  vocal_separation: boolean
  demucs_mp3: boolean
  demucs_mp3_bitrate: number
  keep_promotions: boolean
  use_lyrics: boolean
  lyrics_mode: 'prompt' | 'align' | 'correct'
  save_previous_results: boolean
  opening_threshold: number
  fallback_viet_lyrics: boolean
  fallback_viet_lyrics_model: string
}

type Job = {
  id: string
  status: JobStatus
  created_at: string
  started_at: string | null
  finished_at: string | null
  return_code: number | null
  request: JobRequest
  log_count: number
  logs?: string[]
}

type Transcript = {
  path: string
  size: number
  modified_at: string
}

const defaultForm: JobRequest = {
  file: null,
  backend: 'faster-whisper',
  model: 'large-v3',
  device: 'auto',
  language: 'vi',
  vocal_separation: true,
  demucs_mp3: false,
  demucs_mp3_bitrate: 320,
  keep_promotions: false,
  use_lyrics: true,
  lyrics_mode: 'prompt',
  save_previous_results: true,
  opening_threshold: 1,
  fallback_viet_lyrics: false,
  fallback_viet_lyrics_model: 'kelvinbksoh/whisper-large-v2-vietnamese-lyrics-transcription',
}

const terminalStatuses: JobStatus[] = ['completed', 'failed', 'cancelled']
const themeOptions: { value: Theme; label: string; colors: [string, string] }[] = [
  { value: 'light', label: 'Light', colors: ['#fafaf6', '#17634f'] },
  { value: 'dark', label: 'Dark', colors: ['#202623', '#79c5a7'] },
  { value: 'royal-blue', label: 'Royal Blue', colors: ['#14213d', '#4f8cff'] },
  { value: 'royal-purple', label: 'Royal Purple', colors: ['#281b3d', '#ad7bea'] },
  { value: 'black', label: 'Black', colors: ['#090909', '#f2f2f2'] },
  { value: 'yellow', label: 'Yellow', colors: ['#f3cf3f', '#3d3208'] },
]
const darkThemes = new Set<Theme>(['dark', 'royal-blue', 'royal-purple', 'black'])

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init)
  if (!response.ok) {
    const body = await response.json().catch(() => null)
    throw new Error(body?.detail || `Request failed (${response.status})`)
  }
  return response.json()
}

function timeLabel(value: string | null) {
  if (!value) return 'Pending'
  return new Intl.DateTimeFormat(undefined, {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  }).format(new Date(value))
}

function statusIcon(status: JobStatus) {
  if (status === 'running') return <LoaderCircle className="spin" />
  if (status === 'completed') return <CircleCheck />
  if (status === 'failed') return <CircleX />
  if (status === 'cancelled') return <Ban />
  return <Clock3 />
}

function progressFor(job: Job | null) {
  if (!job) return 0
  if (job.status === 'completed') return 100
  if (job.status === 'queued') return 2
  const matches = (job.logs ?? []).flatMap((line) => [...line.matchAll(/transcription (\d+)%/gi)])
  return matches.length ? Number(matches.at(-1)?.[1]) : job.status === 'running' ? 8 : 0
}

function Toggle({ checked, onChange, label }: { checked: boolean; onChange: (checked: boolean) => void; label: string }) {
  return (
    <label className="toggle-row">
      <span>{label}</span>
      <input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} />
      <span className="toggle" aria-hidden="true" />
    </label>
  )
}

export default function App() {
  const [theme, setTheme] = useState<Theme>(() => {
    const initialTheme = document.documentElement.dataset.theme as Theme
    return themeOptions.some((option) => option.value === initialTheme) ? initialTheme : 'black'
  })
  const [config, setConfig] = useState<AppConfig | null>(null)
  const [files, setFiles] = useState<string[]>([])
  const [jobs, setJobs] = useState<Job[]>([])
  const [selectedJob, setSelectedJob] = useState<Job | null>(null)
  const [form, setForm] = useState<JobRequest>(defaultForm)
  const [transcripts, setTranscripts] = useState<Transcript[]>([])
  const [selectedTranscript, setSelectedTranscript] = useState<string | null>(null)
  const [transcriptText, setTranscriptText] = useState('')
  const [view, setView] = useState<'run' | 'results'>('run')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const logRef = useRef<HTMLPreElement>(null)
  const themeMenuRef = useRef<HTMLDetailsElement>(null)

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    document.documentElement.dataset.mode = darkThemes.has(theme) ? 'dark' : 'light'
    document.documentElement.style.colorScheme = darkThemes.has(theme) ? 'dark' : 'light'
    localStorage.setItem('ss-transcriber-theme', theme)
  }, [theme])

  const refreshTranscripts = async () => {
    const result = await api<{ files: Transcript[] }>('/api/transcripts')
    setTranscripts(result.files)
  }

  useEffect(() => {
    Promise.all([
      api<AppConfig>('/api/config'),
      api<{ files: string[] }>('/api/files'),
      api<Job[]>('/api/jobs'),
      api<{ files: Transcript[] }>('/api/transcripts'),
    ])
      .then(([configResult, fileResult, jobResult, transcriptResult]) => {
        setConfig(configResult)
        setFiles(fileResult.files)
        setJobs(jobResult)
        setTranscripts(transcriptResult.files)
        if (jobResult[0]) setSelectedJob(jobResult[0])
      })
      .catch((reason) => setError(reason.message))
  }, [])

  useEffect(() => {
    const timer = window.setInterval(async () => {
      try {
        const latestJobs = await api<Job[]>('/api/jobs')
        setJobs(latestJobs)
        if (selectedJob) {
          const detail = await api<Job>(`/api/jobs/${selectedJob.id}`)
          const justFinished = !terminalStatuses.includes(selectedJob.status) && terminalStatuses.includes(detail.status)
          setSelectedJob(detail)
          if (justFinished) await refreshTranscripts()
        }
      } catch (reason) {
        setError(reason instanceof Error ? reason.message : String(reason))
      }
    }, 1200)
    return () => window.clearInterval(timer)
  }, [selectedJob?.id, selectedJob?.status])

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight
  }, [selectedJob?.logs])

  const update = <Key extends keyof JobRequest>(key: Key, value: JobRequest[Key]) => {
    setForm((current) => ({ ...current, [key]: value }))
  }

  const changeBackend = (backend: string) => {
    const backendConfig = config?.backends[backend]
    setForm((current) => ({
      ...current,
      backend,
      model: backendConfig?.default_model ?? current.model,
    }))
  }

  const submit = async (event: FormEvent) => {
    event.preventDefault()
    setBusy(true)
    setError(null)
    try {
      const job = await api<Job>('/api/jobs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form),
      })
      setSelectedJob(job)
      setJobs((current) => [job, ...current])
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason))
    } finally {
      setBusy(false)
    }
  }

  const cancel = async () => {
    if (!selectedJob) return
    setBusy(true)
    try {
      setSelectedJob(await api<Job>(`/api/jobs/${selectedJob.id}`, { method: 'DELETE' }))
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason))
    } finally {
      setBusy(false)
    }
  }

  const openJob = async (job: Job) => {
    setView('run')
    try {
      setSelectedJob(await api<Job>(`/api/jobs/${job.id}`))
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason))
    }
  }

  const openTranscript = async (transcript: Transcript) => {
    setSelectedTranscript(transcript.path)
    setTranscriptText('Loading...')
    try {
      const response = await fetch(`/api/transcripts/${encodeURI(transcript.path)}`)
      if (!response.ok) throw new Error('Unable to open transcript')
      setTranscriptText(await response.text())
    } catch (reason) {
      setTranscriptText(reason instanceof Error ? reason.message : String(reason))
    }
  }

  const selectTheme = (nextTheme: Theme) => {
    if (themeMenuRef.current) themeMenuRef.current.open = false
    if (nextTheme === theme) return

    const applyTheme = () => {
      flushSync(() => setTheme(nextTheme))
      document.documentElement.dataset.theme = nextTheme
      document.documentElement.dataset.mode = darkThemes.has(nextTheme) ? 'dark' : 'light'
      document.documentElement.style.colorScheme = darkThemes.has(nextTheme) ? 'dark' : 'light'
    }
    const transitionDocument = document as ViewTransitionDocument

    if (!transitionDocument.startViewTransition || window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      applyTheme()
      return
    }

    const transition = transitionDocument.startViewTransition(applyTheme)
    transition.ready.then(() => {
      const radius = Math.hypot(window.innerWidth / 2, window.innerHeight / 2)
      const options: KeyframeAnimationOptions & { pseudoElement: string } = {
        duration: 700,
        easing: 'cubic-bezier(.2, .72, .2, 1)',
        pseudoElement: '::view-transition-new(root)',
      }
      document.documentElement.animate(
        { clipPath: ['circle(0 at 50% 50%)', `circle(${radius}px at 50% 50%)`] },
        options,
      )
    }).catch(() => undefined)
  }

  const active = selectedJob && !terminalStatuses.includes(selectedJob.status)
  const progress = progressFor(selectedJob)
  const backendConfig = config?.backends[form.backend]

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark"><AudioLines /></span>
          <span><strong>ss</strong>Transcriber</span>
        </div>
        <nav aria-label="Workspace">
          <button className={view === 'run' ? 'active' : ''} onClick={() => setView('run')}>
            <Music2 /> <span className="nav-label">Transcribe</span>
          </button>
          <button className={view === 'results' ? 'active' : ''} onClick={() => setView('results')}>
            <Archive /> <span className="nav-label">Results</span> <span className="nav-count">{transcripts.length}</span>
          </button>
        </nav>
        <div className="sidebar-foot">
          <span className="health-dot" /> API connected
          <small>Local workspace</small>
        </div>
      </aside>

      <main className="workspace">
        <header className="topbar">
          <div>
            <span className="eyebrow">Local audio workspace</span>
            <h1>{view === 'run' ? 'Transcription desk' : 'Transcript archive'}</h1>
          </div>
          <div className="topbar-actions">
            <details className="theme-picker" ref={themeMenuRef}>
              <summary className="theme-button" aria-label="Choose color scheme" title="Choose color scheme">
                <Palette />
              </summary>
              <div className="theme-menu">
                <span className="theme-menu-title">Color scheme</span>
                {themeOptions.map((option) => (
                  <button
                    key={option.value}
                    type="button"
                    className={theme === option.value ? 'active' : ''}
                    aria-pressed={theme === option.value}
                    onClick={() => selectTheme(option.value)}
                  >
                    <span className="theme-swatches" aria-hidden="true">
                      <i style={{ backgroundColor: option.colors[0] }} />
                      <i style={{ backgroundColor: option.colors[1] }} />
                    </span>
                    <span>{option.label}</span>
                    <CircleCheck />
                  </button>
                ))}
              </div>
            </details>
            <div className="topbar-stat">
              <Activity />
              <span><strong>{jobs.filter((job) => job.status === 'running').length}</strong> running</span>
            </div>
          </div>
        </header>

        {error && (
          <div className="error-banner" role="alert">
            <CircleX /> <span>{error}</span>
            <button aria-label="Dismiss error" title="Dismiss" onClick={() => setError(null)}>×</button>
          </div>
        )}

        {view === 'run' ? (
          <div className="run-layout">
            <form className="control-panel" onSubmit={submit}>
              <section className="panel-section source-section">
                <div className="section-heading">
                  <span className="step">01</span>
                  <div><h2>Source</h2><p>{files.length} audio files available</p></div>
                  <button className="icon-button" type="button" title="Refresh files" onClick={() => api<{ files: string[] }>('/api/files').then((data) => setFiles(data.files))}>
                    <RefreshCw />
                  </button>
                </div>
                <label className="field full">
                  <span>Audio selection</span>
                  <div className="select-wrap">
                    <FolderOpen />
                    <select value={form.file ?? ''} onChange={(event) => update('file', event.target.value || null)}>
                      <option value="">All audio files</option>
                      {files.map((file) => <option key={file} value={file}>{file}</option>)}
                    </select>
                    <ChevronDown />
                  </div>
                </label>
              </section>

              <section className="panel-section">
                <div className="section-heading">
                  <span className="step">02</span>
                  <div><h2>Recognition</h2><p>Model and compute profile</p></div>
                </div>
                <div className="field-grid">
                  <label className="field">
                    <span>Backend</span>
                    <select value={form.backend} onChange={(event) => changeBackend(event.target.value)}>
                      {Object.keys(config?.backends ?? {}).map((backend) => <option key={backend}>{backend}</option>)}
                    </select>
                  </label>
                  <label className="field">
                    <span>Device</span>
                    <select value={form.device} onChange={(event) => update('device', event.target.value)}>
                      {(config?.devices ?? []).map((device) => <option key={device.value} value={device.value}>{device.label}</option>)}
                    </select>
                  </label>
                  <label className="field full">
                    <span>Model</span>
                    <select value={form.model} onChange={(event) => update('model', event.target.value)}>
                      {(backendConfig?.models ?? [form.model]).map((model) => <option key={model}>{model}</option>)}
                    </select>
                  </label>
                  <label className="field">
                    <span>Language</span>
                    <input value={form.language ?? ''} placeholder="Auto" maxLength={12} onChange={(event) => update('language', event.target.value || null)} />
                  </label>
                  <label className="field">
                    <span>Opening threshold</span>
                    <div className="unit-input"><input type="number" min="0" max="300" step="0.5" value={form.opening_threshold} onChange={(event) => update('opening_threshold', Number(event.target.value))} /><span>sec</span></div>
                  </label>
                </div>
              </section>

              <section className="panel-section">
                <div className="section-heading">
                  <span className="step">03</span>
                  <div><h2>Lyrics assist</h2><p>Same-stem text under input/lyrics</p></div>
                </div>
                <Toggle checked={form.use_lyrics} onChange={(value) => update('use_lyrics', value)} label="Use known lyrics" />
                <div className={`segmented ${!form.use_lyrics ? 'disabled' : ''}`}>
                  {(['prompt', 'align', 'correct'] as const).map((mode) => (
                    <button key={mode} type="button" disabled={!form.use_lyrics} className={form.lyrics_mode === mode ? 'active' : ''} onClick={() => update('lyrics_mode', mode)}>{mode}</button>
                  ))}
                </div>
              </section>

              <details className="advanced panel-section">
                <summary><WandSparkles /> Advanced <ChevronDown /></summary>
                <div className="advanced-body">
                  <Toggle checked={form.vocal_separation} onChange={(value) => update('vocal_separation', value)} label="Separate vocals with Demucs" />
                  <Toggle checked={form.demucs_mp3} onChange={(value) => update('demucs_mp3', value)} label="Store Demucs stem as MP3" />
                  <Toggle checked={form.keep_promotions} onChange={(value) => update('keep_promotions', value)} label="Keep promotional phrases" />
                  <Toggle checked={form.save_previous_results} onChange={(value) => update('save_previous_results', value)} label="Archive previous results" />
                  <Toggle checked={form.fallback_viet_lyrics} onChange={(value) => update('fallback_viet_lyrics', value)} label="Viet Lyrics fallback pass" />
                  {form.demucs_mp3 && (
                    <label className="field"><span>Demucs bitrate</span><div className="unit-input"><input type="number" min="64" max="512" value={form.demucs_mp3_bitrate} onChange={(event) => update('demucs_mp3_bitrate', Number(event.target.value))} /><span>kbps</span></div></label>
                  )}
                  {form.fallback_viet_lyrics && (
                    <label className="field full"><span>Fallback model</span><input value={form.fallback_viet_lyrics_model} onChange={(event) => update('fallback_viet_lyrics_model', event.target.value)} /></label>
                  )}
                  <details className="profile-details">
                    <summary>Backend profile <span>{Object.keys(backendConfig?.options ?? {}).length} groups</span></summary>
                    <pre>{JSON.stringify(backendConfig?.options ?? {}, null, 2)}</pre>
                  </details>
                </div>
              </details>

              <div className="action-bar">
                <div><strong>{form.file ? '1 file' : `${files.length} files`}</strong><span>{form.backend} · {form.device}</span></div>
                <button className="primary-button" type="submit" disabled={busy || !config}>
                  {busy ? <LoaderCircle className="spin" /> : <Play />} Queue transcription
                </button>
              </div>
            </form>

            <aside className="activity-panel">
              <div className="activity-head">
                <div><span className="eyebrow">Live activity</span><h2>{selectedJob ? selectedJob.request.file ?? 'Batch run' : 'No job selected'}</h2></div>
                {selectedJob && <span className={`status ${selectedJob.status}`}>{statusIcon(selectedJob.status)} {selectedJob.status}</span>}
              </div>
              {selectedJob ? (
                <>
                  <div className="progress-block">
                    <div><span>Progress</span><strong>{progress}%</strong></div>
                    <div className="progress-track"><span style={{ width: `${progress}%` }} /></div>
                    <div className="job-meta"><span><Cpu /> {selectedJob.request.device}</span><span><Clock3 /> {timeLabel(selectedJob.started_at)}</span><span><Gauge /> {selectedJob.request.model}</span></div>
                  </div>
                  <div className="console">
                    <div className="console-head"><span><TerminalSquare /> Process output</span><code>{selectedJob.id}</code></div>
                    <pre ref={logRef}>{selectedJob.logs?.length ? selectedJob.logs.join('\n') : 'Waiting for process output...'}</pre>
                  </div>
                  {active && <button className="stop-button" type="button" disabled={busy} onClick={cancel}><Square /> Stop process</button>}
                </>
              ) : (
                <div className="empty-state"><AudioLines /><h3>Ready for a run</h3><p>Choose a source and queue transcription.</p></div>
              )}

              <div className="history-list">
                <div className="subhead"><span><History /> Recent jobs</span><small>{jobs.length}</small></div>
                {jobs.slice(0, 8).map((job) => (
                  <button key={job.id} className={selectedJob?.id === job.id ? 'selected' : ''} onClick={() => openJob(job)}>
                    <span className={`job-icon ${job.status}`}>{statusIcon(job.status)}</span>
                    <span><strong>{job.request.file?.split('/').at(-1) ?? 'All audio files'}</strong><small>{job.request.backend} · {timeLabel(job.created_at)}</small></span>
                    <span className="job-state">{job.status}</span>
                  </button>
                ))}
              </div>
            </aside>
          </div>
        ) : (
          <div className="results-layout">
            <section className="result-list">
              <div className="result-toolbar"><span><FileText /> {transcripts.length} artifacts</span><button className="icon-button" title="Refresh results" onClick={() => refreshTranscripts()}><RefreshCw /></button></div>
              {transcripts.map((transcript) => (
                <button key={transcript.path} className={selectedTranscript === transcript.path ? 'selected' : ''} onClick={() => openTranscript(transcript)}>
                  <FileAudio />
                  <span><strong>{transcript.path.split('/').at(-1)}</strong><small>{transcript.path.split('/')[0]} · {(transcript.size / 1024).toFixed(1)} KB</small></span>
                  <time>{timeLabel(transcript.modified_at)}</time>
                </button>
              ))}
              {!transcripts.length && <div className="empty-state"><FileText /><h3>No transcripts yet</h3></div>}
            </section>
            <section className="transcript-viewer">
              <div className="viewer-head"><span className="eyebrow">Preview</span><h2>{selectedTranscript?.split('/').at(-1) ?? 'Select an artifact'}</h2></div>
              <pre>{transcriptText || 'Transcript content will appear here.'}</pre>
            </section>
          </div>
        )}
      </main>
    </div>
  )
}