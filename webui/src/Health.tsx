import { useEffect, useState } from 'react'
import { ArrowDown, ArrowUp, Cpu, HardDrive, LoaderCircle, MemoryStick, MonitorCog, Network, RefreshCw } from 'lucide-react'

type HealthSnapshot = {
  sampled_at: string
  cpu: { percent: number; cores: number[] }
  memory: { total: number; used: number; available: number; percent: number }
  storage: { path: string; device: string; total: number; used: number; free: number; percent: number }[]
  network: { received_per_second: number; sent_per_second: number; received_total: number; sent_total: number } | null
  gpu: {
    devices: { id: string; name: string; percent: number | null; memory_used: number | null; memory_total: number | null; temperature: number | null }[]
    unavailable_reason: string | null
  }
}

function bytes(value: number | null) {
  if (value === null) return 'Unavailable'
  const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB']
  const index = value > 0 ? Math.min(Math.floor(Math.log(value) / Math.log(1024)), units.length - 1) : 0
  const unit = Math.max(0, index)
  return `${(value / 1024 ** unit).toLocaleString(undefined, { maximumFractionDigits: 1 })} ${units[unit]}`
}

function percent(value: number | null) {
  return value === null ? 'Unavailable' : `${value.toFixed(1)}%`
}

function UsageMeter({ value, label }: { value: number | null; label: string }) {
  return value === null ? null : (
    <meter className="health-meter" min={0} max={100} value={value} aria-label={label}>
      {percent(value)}
    </meter>
  )
}

export default function Health() {
  const [snapshot, setSnapshot] = useState<HealthSnapshot | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [refresh, setRefresh] = useState(0)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const controller = new AbortController()
    let timer: ReturnType<typeof setTimeout>
    async function poll() {
      if (document.hidden) {
        timer = setTimeout(poll, 2000)
        return
      }
      setLoading(true)
      try {
        const response = await fetch('/api/health', {
          signal: AbortSignal.any([controller.signal, AbortSignal.timeout(10000)]),
          cache: 'no-store',
        })
        if (!response.ok) throw new Error(`System metrics request failed (${response.status}).`)
        const data: HealthSnapshot = await response.json()
        if (!controller.signal.aborted) {
          setSnapshot(data)
          setError(null)
        }
      } catch (reason) {
        if (!controller.signal.aborted) {
          setError(reason instanceof Error ? reason.message : 'System metrics are unavailable.')
        }
      } finally {
        if (!controller.signal.aborted) {
          setLoading(false)
          timer = setTimeout(poll, 2000)
        }
      }
    }
    void poll()
    return () => {
      controller.abort()
      clearTimeout(timer)
    }
  }, [refresh])

  return (
    <div className="health-workspace">
      <div className="health-toolbar">
        <span role="status">
          {error ? (snapshot ? 'Stale readings' : 'Metrics unavailable') : snapshot ? 'Live system metrics' : 'Connecting'}
          {snapshot && <time dateTime={snapshot.sampled_at}>Updated {new Date(snapshot.sampled_at).toLocaleTimeString()}</time>}
        </span>
        <button className="icon-button" type="button" title="Refresh metrics" aria-label="Refresh metrics" disabled={loading} onClick={() => setRefresh((current) => current + 1)}>
          <RefreshCw />
        </button>
      </div>
      {error && <div className="error-banner" role="alert">{error} Retrying automatically.</div>}
      {!snapshot && !error && <div className="health-loading" role="status"><LoaderCircle /> Loading system metrics...</div>}
      {snapshot && <>
        <div className="health-overview">
          <section className="health-section health-cpu">
            <h2><Cpu /> CPU</h2>
            <strong className="health-value">{percent(snapshot.cpu.percent)}</strong>
            <UsageMeter value={snapshot.cpu.percent} label="Total CPU usage" />
            <p className="health-detail">{snapshot.cpu.cores.length} logical processors</p>
            <div className="health-cores">
              {snapshot.cpu.cores.map((value, index) => (
                <div key={index}>
                  <span>CPU {index + 1}<strong>{percent(value)}</strong></span>
                  <UsageMeter value={value} label={`CPU ${index + 1} usage`} />
                </div>
              ))}
            </div>
          </section>
          <section className="health-section health-memory">
            <h2><MemoryStick /> Memory</h2>
            <strong className="health-value">{percent(snapshot.memory.percent)}</strong>
            <UsageMeter value={snapshot.memory.percent} label="Memory usage" />
            <dl className="health-facts">
              <div><dt>In use</dt><dd>{bytes(snapshot.memory.used)}</dd></div>
              <div><dt>Available</dt><dd>{bytes(snapshot.memory.available)}</dd></div>
              <div><dt>Total RAM</dt><dd>{bytes(snapshot.memory.total)}</dd></div>
            </dl>
          </section>
          <section className="health-section health-network">
            <h2><Network /> Network</h2>
            {snapshot.network ? <>
              <dl className="health-facts health-transfers">
                <div><dt><ArrowDown /> Receive</dt><dd>{bytes(snapshot.network.received_per_second)}/s</dd></div>
                <div><dt><ArrowUp /> Send</dt><dd>{bytes(snapshot.network.sent_per_second)}/s</dd></div>
              </dl>
              <dl className="health-facts">
                <div><dt>Total received</dt><dd>{bytes(snapshot.network.received_total)}</dd></div>
                <div><dt>Total sent</dt><dd>{bytes(snapshot.network.sent_total)}</dd></div>
              </dl>
              <p className="health-detail">All network interfaces</p>
            </> : <p className="health-unavailable">Network counters unavailable.</p>}
          </section>
        </div>
        <section className="health-section health-storage">
          <h2><HardDrive /> Storage</h2>
          <div className="health-device-grid">
            {snapshot.storage.map((disk) => (
              <article className="health-device" key={disk.path}>
                <h3 title={disk.device}>{disk.path}</h3>
                <div className="health-reading"><strong>{percent(disk.percent)}</strong><span>{bytes(disk.free)} free</span></div>
                <UsageMeter value={disk.percent} label={`${disk.path} storage usage`} />
                <p className="health-detail">{bytes(disk.used)} used / {bytes(disk.total)}</p>
              </article>
            ))}
          </div>
          {!snapshot.storage.length && <p className="health-unavailable">No accessible storage volumes.</p>}
        </section>
        <section className="health-section health-gpu">
          <h2><MonitorCog /> GPU</h2>
          {snapshot.gpu.unavailable_reason && <p className="health-unavailable">{snapshot.gpu.unavailable_reason}</p>}
          <div className="health-device-grid">
            {snapshot.gpu.devices.map((gpu) => (
              <article className="health-device" key={gpu.id}>
                <h3><span className="health-detail">GPU {gpu.id}</span>{gpu.name}</h3>
                <div className="health-reading"><strong>{percent(gpu.percent)}</strong><span>{gpu.temperature === null ? 'Temperature unavailable' : `${gpu.temperature.toFixed(0)} \u00b0C`}</span></div>
                <UsageMeter value={gpu.percent} label={`GPU ${gpu.id} usage`} />
                <dl className="health-facts">
                  <div><dt>VRAM used</dt><dd>{bytes(gpu.memory_used)}</dd></div>
                  <div><dt>VRAM total</dt><dd>{bytes(gpu.memory_total)}</dd></div>
                </dl>
                <UsageMeter value={gpu.memory_used !== null && gpu.memory_total ? gpu.memory_used / gpu.memory_total * 100 : null} label={`GPU ${gpu.id} VRAM usage`} />
              </article>
            ))}
          </div>
        </section>
      </>}
    </div>
  )
}