import { useEffect, useId, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import { createPortal, flushSync } from 'react-dom'
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
  Info,
  LoaderCircle,
  Maximize2,
  Minimize2,
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
  backend_options: Record<string, unknown>
  fallback_viet_lyrics_options: Record<string, unknown>
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
  backend_options: {},
  fallback_viet_lyrics_options: {},
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

function profileText(options: Record<string, unknown> | undefined) {
  return JSON.stringify(options ?? {}, null, 2)
}

function parseProfile(label: string, value: string) {
  let profile: unknown
  try {
    profile = JSON.parse(value)
  } catch (reason) {
    throw new Error(`${label} must be valid JSON: ${reason instanceof Error ? reason.message : String(reason)}`)
  }
  if (!profile || typeof profile !== 'object' || Array.isArray(profile)) {
    throw new Error(`${label} must be a JSON object.`)
  }
  return profile as Record<string, unknown>
}

function InfoTip({ text }: { text: string }) {
  const tooltipId = useId()
  return (
    <span
      className="info-tip"
      role="button"
      tabIndex={0}
      aria-label="More information"
      aria-describedby={tooltipId}
      onClick={(event) => {
        event.preventDefault()
        event.stopPropagation()
      }}
    >
      <Info aria-hidden="true" />
      <span className="info-tooltip" id={tooltipId} role="tooltip">{text}</span>
    </span>
  )
}

function FieldLabel({ children, info }: { children: string; info: string }) {
  return <span className="field-label">{children}<InfoTip text={info} /></span>
}

function Toggle({ checked, onChange, label, info, disabled = false }: { checked: boolean; onChange: (checked: boolean) => void; label: string; info: string; disabled?: boolean }) {
  return (
    <label className={`toggle-row ${disabled ? 'disabled' : ''}`}>
      <span className="toggle-label">{label}<InfoTip text={info} /></span>
      <input type="checkbox" checked={checked} disabled={disabled} onChange={(event) => onChange(event.target.checked)} />
      <span className="toggle" aria-hidden="true" />
    </label>
  )
}

function TranscriptGroup({
  folder,
  files,
  selectedTranscript,
  onOpen,
}: {
  folder: string
  files: Transcript[]
  selectedTranscript: string | null
  onOpen: (transcript: Transcript) => void
}) {
  const [open, setOpen] = useState(folder === 'transcripts' || selectedTranscript?.startsWith(`${folder}/`))

  return (
    <details className="result-group" open={open} onToggle={(event) => setOpen(event.currentTarget.open)}>
      <summary>
        <FolderOpen />
        <span><strong>{folder}</strong><small>{folder === 'transcripts' ? 'Current results' : 'Archived results'}</small></span>
        <span className="folder-count">{files.length}</span>
        <ChevronDown />
      </summary>
      <div className="result-group-files">
        {files.map((transcript) => {
          const relativePath = transcript.path.split('/').slice(1).join('/')
          return (
            <button key={transcript.path} className={selectedTranscript === transcript.path ? 'selected' : ''} onClick={() => onOpen(transcript)}>
              <FileAudio />
              <span><strong>{relativePath}</strong><small>{(transcript.size / 1024).toFixed(1)} KB</small></span>
              <time>{timeLabel(transcript.modified_at)}</time>
            </button>
          )
        })}
      </div>
    </details>
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
  const [consoleExpanded, setConsoleExpanded] = useState(false)
  const [backendProfileText, setBackendProfileText] = useState('{}')
  const [fallbackProfileText, setFallbackProfileText] = useState('{}')
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
        setBackendProfileText(profileText(configResult.backends[defaultForm.backend]?.options))
        setFallbackProfileText(profileText(configResult.backends['viet-lyrics']?.options))
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

  useEffect(() => {
    if (!consoleExpanded) return
    const previousOverflow = document.body.style.overflow
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setConsoleExpanded(false)
    }
    document.body.style.overflow = 'hidden'
    document.addEventListener('keydown', closeOnEscape)
    return () => {
      document.body.style.overflow = previousOverflow
      document.removeEventListener('keydown', closeOnEscape)
    }
  }, [consoleExpanded])

  const update = <Key extends keyof JobRequest>(key: Key, value: JobRequest[Key]) => {
    setForm((current) => ({ ...current, [key]: value }))
  }

  const changeBackend = (backend: string) => {
    const backendConfig = config?.backends[backend]
    setBackendProfileText(profileText(backendConfig?.options))
    setForm((current) => ({
      ...current,
      backend,
      model: backendConfig?.default_model ?? current.model,
      fallback_viet_lyrics: backend === 'viet-lyrics' ? false : current.fallback_viet_lyrics,
    }))
  }

  const submit = async (event: FormEvent) => {
    event.preventDefault()
    setBusy(true)
    setError(null)
    try {
      const backendOptions = parseProfile('Backend profile', backendProfileText)
      const fallbackOptions = parseProfile('Fallback profile', fallbackProfileText)
      const job = await api<Job>('/api/jobs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...form,
          backend_options: backendOptions,
          fallback_viet_lyrics_options: fallbackOptions,
        }),
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
  const fallbackModels = config?.backends['viet-lyrics']?.models ?? []
  const processConsole = selectedJob ? (
    <div className={`console ${consoleExpanded ? 'expanded' : ''}`}>
      <div className="console-head">
        <span><TerminalSquare /> Process output</span>
        <div className="console-actions">
          <code>{selectedJob.id}</code>
          <button
            type="button"
            aria-label={consoleExpanded ? 'Exit full window' : 'Expand process output'}
            aria-pressed={consoleExpanded}
            title={consoleExpanded ? 'Exit full window (Esc)' : 'Expand process output'}
            onClick={() => setConsoleExpanded((current) => !current)}
          >
            {consoleExpanded ? <Minimize2 /> : <Maximize2 />}
          </button>
        </div>
      </div>
      <pre ref={logRef}>{selectedJob.logs?.length ? selectedJob.logs.join('\n') : 'Waiting for process output...'}</pre>
    </div>
  ) : null
  const transcriptGroups = Object.entries(
    transcripts.reduce<Record<string, Transcript[]>>((groups, transcript) => {
      const folder = transcript.path.split('/')[0]
      groups[folder] = [...(groups[folder] ?? []), transcript]
      return groups
    }, {}),
  )

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
                  <FieldLabel info="Choose one supported audio file from input/, or select all files to process the complete folder.">Audio selection</FieldLabel>
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
                    <FieldLabel info="Select the speech-recognition engine. Faster-Whisper is the general default; PhoWhisper and Viet Lyrics specialize in Vietnamese; Parakeet and SenseVoice require their optional runtimes.">Backend</FieldLabel>
                    <select value={form.backend} onChange={(event) => changeBackend(event.target.value)}>
                      {Object.keys(config?.backends ?? {}).map((backend) => <option key={backend}>{backend}</option>)}
                    </select>
                  </label>
                  <label className="field">
                    <FieldLabel info="Choose where inference runs. Automatic prefers an available GPU, CPU uses no CUDA, and an NVIDIA device runs the model on that specific GPU.">Device</FieldLabel>
                    <select value={form.device} onChange={(event) => update('device', event.target.value)}>
                      {(config?.devices ?? []).map((device) => <option key={device.value} value={device.value}>{device.label}</option>)}
                    </select>
                  </label>
                  <label className="field full">
                    <FieldLabel info="Select the checkpoint used by the current backend. Larger models generally improve accuracy but take more memory and processing time.">Model</FieldLabel>
                    <select value={form.model} onChange={(event) => update('model', event.target.value)}>
                      {(backendConfig?.models ?? [form.model]).map((model) => <option key={model}>{model}</option>)}
                    </select>
                  </label>
                  <label className="field">
                    <FieldLabel info="Enter an ISO language code such as vi or en to force recognition, or leave it empty for automatic language detection.">Language</FieldLabel>
                    <input value={form.language ?? ''} placeholder="Auto" maxLength={12} onChange={(event) => update('language', event.target.value || null)} />
                  </label>
                  <label className="field">
                    <FieldLabel info="If separated-vocal transcription starts later than this many seconds, retry the original mix to recover a potentially clipped opening.">Opening threshold</FieldLabel>
                    <div className="unit-input"><input type="number" min="0" max="300" step="0.5" value={form.opening_threshold} onChange={(event) => update('opening_threshold', Number(event.target.value))} /><span>sec</span></div>
                  </label>
                </div>
              </section>

              <section className="panel-section">
                <div className="section-heading">
                  <span className="step">03</span>
                  <div><h2>Lyrics assist</h2><p>Same-stem text under input/lyrics</p></div>
                </div>
                <Toggle checked={form.use_lyrics} onChange={(value) => update('use_lyrics', value)} label="Use known lyrics" info="Use a same-stem UTF-8 text file under input/lyrics to guide, align, or correct the recognized lyrics." />
                <div className="option-label">Lyrics mode<InfoTip text="Prompt biases recognition toward known words. Align maps authoritative lyric lines onto ASR timing. Correct replaces recognized text while preserving ASR segment timing." /></div>
                <div className={`segmented ${!form.use_lyrics ? 'disabled' : ''}`}>
                  {(['prompt', 'align', 'correct'] as const).map((mode) => (
                    <button key={mode} type="button" disabled={!form.use_lyrics} className={form.lyrics_mode === mode ? 'active' : ''} onClick={() => update('lyrics_mode', mode)}>{mode}</button>
                  ))}
                </div>
              </section>

              <details className="advanced panel-section" open>
                <summary><WandSparkles /> Advanced <ChevronDown /></summary>
                <div className="advanced-body">
                  <Toggle checked={form.vocal_separation} onChange={(value) => update('vocal_separation', value)} label="Separate vocals with Demucs" info="Isolate the vocal stem before recognition. This often improves song transcription but adds processing time and can occasionally clip quiet openings." />
                  <Toggle checked={form.demucs_mp3} onChange={(value) => update('demucs_mp3', value)} label="Store Demucs stem as MP3" info="Save separated vocals as a smaller lossy MP3 instead of the default WAV file." />
                  <Toggle checked={form.keep_promotions} onChange={(value) => update('keep_promotions', value)} label="Keep promotional phrases" info="Retain phrases such as subscribe, like, and thanks for watching instead of filtering them from transcripts." />
                  <Toggle checked={form.save_previous_results} onChange={(value) => update('save_previous_results', value)} label="Archive previous results" info="Rename existing transcript and Demucs folders to numbered archives before creating results for this run." />
                  <Toggle
                    checked={form.fallback_viet_lyrics}
                    disabled={form.backend === 'viet-lyrics'}
                    onChange={(value) => update('fallback_viet_lyrics', value)}
                    label="Viet Lyrics fallback pass"
                    info={form.backend === 'viet-lyrics' ? 'Unavailable because Viet Lyrics is already the primary transcription backend.' : 'When the opening retry triggers, run an additional Vietnamese lyrics model pass on the separated vocals.'}
                  />
                  {form.demucs_mp3 && (
                    <label className="field"><FieldLabel info="Set the MP3 bitrate for saved Demucs stems. Higher values preserve more audio detail but create larger files.">Demucs bitrate</FieldLabel><div className="unit-input"><input type="number" min="64" max="512" value={form.demucs_mp3_bitrate} onChange={(event) => update('demucs_mp3_bitrate', Number(event.target.value))} /><span>kbps</span></div></label>
                  )}
                  {form.fallback_viet_lyrics && (
                    <>
                      <label className="field full">
                        <FieldLabel info="Choose a suggested Viet Lyrics checkpoint or enter any custom Hugging Face model ID.">Fallback model</FieldLabel>
                        <input
                          list="fallback-model-options"
                          value={form.fallback_viet_lyrics_model}
                          autoComplete="off"
                          onChange={(event) => update('fallback_viet_lyrics_model', event.target.value)}
                        />
                        <datalist id="fallback-model-options">
                          {fallbackModels.map((model) => <option key={model} value={model} />)}
                        </datalist>
                      </label>
                      <details className="profile-details">
                        <summary>Fallback profile <span>{Object.keys(config?.backends['viet-lyrics']?.options ?? {}).length} groups</span></summary>
                        <div className="profile-editor">
                          <label className="field full">
                            <FieldLabel info="Edit the JSON options passed to the isolated Viet Lyrics fallback backend. Nested option groups are merged with backend defaults.">Fallback backend options</FieldLabel>
                            <textarea spellCheck={false} value={fallbackProfileText} onChange={(event) => setFallbackProfileText(event.target.value)} />
                          </label>
                          <button type="button" className="profile-reset" onClick={() => setFallbackProfileText(profileText(config?.backends['viet-lyrics']?.options))}>Reset defaults</button>
                        </div>
                      </details>
                    </>
                  )}
                  <details className="profile-details">
                    <summary>Backend profile <span>{Object.keys(backendConfig?.options ?? {}).length} groups</span></summary>
                    <div className="profile-editor">
                      <label className="field full">
                        <FieldLabel info="Edit the JSON options passed to the selected backend. Nested option groups are merged with backend defaults.">Primary backend options</FieldLabel>
                        <textarea spellCheck={false} value={backendProfileText} onChange={(event) => setBackendProfileText(event.target.value)} />
                      </label>
                      <button type="button" className="profile-reset" onClick={() => setBackendProfileText(profileText(backendConfig?.options))}>Reset defaults</button>
                    </div>
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
                  {consoleExpanded ? createPortal(processConsole, document.body) : processConsole}
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
              <div className="result-toolbar"><span><FileText /> {transcripts.length} artifacts in {transcriptGroups.length} folders</span><button className="icon-button" title="Refresh results" onClick={() => refreshTranscripts()}><RefreshCw /></button></div>
              {transcriptGroups.map(([folder, files]) => (
                <TranscriptGroup key={folder} folder={folder} files={files} selectedTranscript={selectedTranscript} onOpen={openTranscript} />
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