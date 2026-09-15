import { useEffect, useRef, useState } from 'react'
import { Ban, CircleCheck, CircleX, Clock3, Cpu, FileAudio, Gauge, LoaderCircle, RefreshCw, TerminalSquare } from 'lucide-react'

type JobStatus = 'queued' | 'running' | 'completed' | 'failed' | 'cancelled'
type EndpointJob = {
  id: string
  filename: string
  status: JobStatus
  created_at: string
  started_at: string | null
  finished_at: string | null
  return_code: number | null
  log_count: number
  logs?: string[]
  request: {
    backend: string
    model: string
    device: string
    use_lyrics: boolean
    lyrics_mode: string
    copy_no_vocals: boolean
  }
}

const active = (job: EndpointJob) => job.status === 'queued' || job.status === 'running'
const timeLabel = (value: string | null) => value ? new Date(value).toLocaleString() : '—'

function StatusIcon({ status }: { status: JobStatus }) {
  if (status === 'running') return <LoaderCircle className="spin" />
  if (status === 'completed') return <CircleCheck />
  if (status === 'failed') return <CircleX />
  if (status === 'cancelled') return <Ban />
  return <Clock3 />
}

async function load<T>(path: string, signal: AbortSignal): Promise<T> {
  const response = await fetch(path, { signal, cache: 'no-store' })
  if (!response.ok) throw new Error(`Could not load endpoint jobs (${response.status}).`)
  return response.json()
}

export default function EndpointJobs() {
  const [jobs, setJobs] = useState<EndpointJob[]>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const selectedIdRef = useRef<string | null>(null)
  const [detail, setDetail] = useState<EndpointJob | null>(null)
  const [filter, setFilter] = useState<'active' | 'all'>('active')
  const [error, setError] = useState<string | null>(null)
  const [loaded, setLoaded] = useState(false)
  const [refresh, setRefresh] = useState(0)
  const [updatedAt, setUpdatedAt] = useState<string | null>(null)
  const logRef = useRef<HTMLPreElement>(null)

  const select = (id: string) => {
    selectedIdRef.current = id
    setSelectedId(id)
    setDetail(null)
    setRefresh((current) => current + 1)
  }

  useEffect(() => {
    const controller = new AbortController()
    let timer: ReturnType<typeof setTimeout>
    async function poll() {
      try {
        const signal = AbortSignal.any([controller.signal, AbortSignal.timeout(10000)])
        const latest = await load<EndpointJob[]>('/api/endpoint-jobs', signal)
        if (controller.signal.aborted) return
        setJobs(latest)
        setLoaded(true)
        // Keep a selected job visible when it finishes; otherwise prefer a running upload.
        const id = latest.some((job) => job.id === selectedIdRef.current)
          ? selectedIdRef.current
          : latest.find((job) => job.status === 'running')?.id
            ?? latest.find(active)?.id ?? (filter === 'all' ? latest[0]?.id : null) ?? null
        selectedIdRef.current = id
        setSelectedId(id)
        if (id) {
          const next = await load<EndpointJob>(`/api/endpoint-jobs/${id}`, signal)
          if (controller.signal.aborted || selectedIdRef.current !== id) return
          setDetail(next)
        } else {
          setDetail(null)
        }
        setUpdatedAt(new Date().toISOString())
        setError(null)
      } catch (reason) {
        if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : 'Endpoint jobs are unavailable.')
      } finally {
        // Schedule after completion so slow responses never overlap or reorder snapshots.
        if (!controller.signal.aborted) timer = setTimeout(poll, 1200)
      }
    }
    void poll()
    return () => {
      controller.abort()
      clearTimeout(timer)
    }
  }, [refresh, filter])

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight
  }, [detail?.id, detail?.logs])

  const visibleJobs = jobs.filter((job) => filter === 'all' || active(job))
    .sort((a, b) => Number(active(b)) - Number(active(a)))
  const selected = detail?.id === selectedId ? detail : jobs.find((job) => job.id === selectedId)

  return (
    <div className="endpoint-jobs-workspace">
      <div className="health-toolbar">
        <span role="status">{error ? 'Connection interrupted' : !loaded ? 'Connecting' : `${jobs.filter((job) => job.status === 'running').length} running · ${jobs.filter((job) => job.status === 'queued').length} queued`}
          {updatedAt && <time dateTime={updatedAt}>Updated {new Date(updatedAt).toLocaleTimeString()}</time>}
        </span>
        <button className="icon-button" type="button" title="Refresh endpoint jobs" aria-label="Refresh endpoint jobs" onClick={() => setRefresh((current) => current + 1)}><RefreshCw /></button>
      </div>
      {error && <div className="error-banner" role="alert">{error} Retrying automatically; displayed data may be stale.</div>}
      <div className="endpoint-jobs-layout">
        <section className="activity-panel endpoint-job-list">
          <div className="activity-head"><div><span className="eyebrow">POST /api/transcribe</span><h2>Upload requests</h2></div></div>
          <div className="endpoint-job-filters segmented" aria-label="Filter endpoint jobs">
            <button type="button" className={filter === 'active' ? 'active' : ''} aria-pressed={filter === 'active'} onClick={() => setFilter('active')}>Running &amp; queued</button>
            <button type="button" className={filter === 'all' ? 'active' : ''} aria-pressed={filter === 'all'} onClick={() => setFilter('all')}>Recent history</button>
          </div>
          <div className="history-list">
            {visibleJobs.map((job) => (
              <button key={job.id} className={selectedId === job.id ? 'selected' : ''} aria-pressed={selectedId === job.id} onClick={() => select(job.id)}>
                <span className={`job-icon ${job.status}`}><StatusIcon status={job.status} /></span>
                <span><strong title={job.filename}>{job.filename}</strong><small>{job.request.backend} · {timeLabel(job.created_at)}</small><small>{job.id}</small></span>
                <span className="job-state">{job.status}</span>
              </button>
            ))}
          </div>
          {!visibleJobs.length && <div className="empty-state"><FileAudio /><h3>{!loaded ? error ? 'Jobs unavailable' : 'Loading jobs…' : filter === 'active' ? 'No running uploads' : 'No upload requests yet'}</h3><p>{filter === 'active' ? 'New API uploads appear here automatically, including those waiting for the processing queue.' : 'The latest 100 finished uploads are retained until the server restarts.'}</p></div>}
        </section>
        <section className="activity-panel endpoint-job-detail">
          <div className="activity-head">
            <div><span className="eyebrow">Live processing details</span><h2>{selected?.filename ?? 'Select an upload'}</h2></div>
            {selected && <span className={`status ${selected.status}`}><StatusIcon status={selected.status} /> {selected.status}</span>}
          </div>
          {selected ? <>
            <div className="progress-block">
              <div className="job-meta"><span><Cpu /> {selected.request.device}</span><span><Gauge /> {selected.request.backend}</span></div>
              <dl className="endpoint-job-metadata">
                <dt>Model</dt><dd>{selected.request.model}</dd>
                <dt>Received</dt><dd>{timeLabel(selected.created_at)}</dd>
                <dt>Started</dt><dd>{timeLabel(selected.started_at)}</dd>
                <dt>Finished</dt><dd>{timeLabel(selected.finished_at)}</dd>
                <dt>Lyrics</dt><dd>{selected.request.use_lyrics ? selected.request.lyrics_mode : 'Automatic transcription'}</dd>
                <dt>Response</dt><dd>{selected.request.copy_no_vocals ? 'ZIP · song + no-vocals song' : 'Embedded song'}</dd>
              </dl>
              <p className="endpoint-job-note">{selected.status === 'queued' ? 'Waiting for the shared transcription queue.' : selected.status === 'completed' ? 'Processing finished and the download was prepared for the API caller.' : selected.status === 'failed' ? 'Processing failed. See the logs below for details.' : selected.status === 'cancelled' ? 'The upload request was cancelled.' : 'Processing audio or preparing the download. Logs update automatically.'}</p>
            </div>
            <div className="console">
              <div className="console-head"><span><TerminalSquare /> Processing log</span><span>{selected.id}</span></div>
              <pre ref={logRef} aria-label="Endpoint job processing log">{detail?.id === selected.id ? detail.logs?.join('\n') || 'Waiting for processing output…' : 'Loading log…'}</pre>
            </div>
          </> : <div className="empty-state"><TerminalSquare /><h3>Monitor an API request</h3><p>Select a job to view its settings, timestamps, and live processing log. Downloads are sent to the original API caller.</p></div>}
        </section>
      </div>
    </div>
  )
}