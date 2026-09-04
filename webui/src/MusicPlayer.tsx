import { useEffect, useRef, useState } from 'react'
import { ChevronDown, FileAudio, Music2, Pause, Play, RefreshCw, Volume2 } from 'lucide-react'

type LyricEntry = {
  text: string
  time_ms: number
}

type LyricsResponse = {
  language: string | null
  entries: LyricEntry[]
  uslt: string
}

type LyricsMode = 'sylt' | 'uslt'

type MusicFile = {
  library: string
  group: string
  path: string
  name: string
}

type MusicPlayerProps = {
  files: MusicFile[]
  onRefresh: () => Promise<void>
}

function apiPath(prefix: string, file: MusicFile) {
  const path = file.path.split('/').map(encodeURIComponent).join('/')
  return `${prefix}/${encodeURIComponent(file.library)}/${path}`
}

function durationLabel(seconds: number) {
  if (!Number.isFinite(seconds)) return '0:00'
  const minutes = Math.floor(seconds / 60)
  const remainder = Math.floor(seconds % 60)
  return `${minutes}:${remainder.toString().padStart(2, '0')}`
}

export default function MusicPlayer({ files, onRefresh }: MusicPlayerProps) {
  const [selectedFile, setSelectedFile] = useState<MusicFile | null>(null)
  const [lyrics, setLyrics] = useState<LyricEntry[]>([])
  const [uslt, setUslt] = useState('')
  const [lyricsMode, setLyricsMode] = useState<LyricsMode>('sylt')
  const [language, setLanguage] = useState<string | null>(null)
  const [lyricsLoading, setLyricsLoading] = useState(false)
  const [lyricsError, setLyricsError] = useState<string | null>(null)
  const [playing, setPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [volume, setVolume] = useState(1)
  const audioRef = useRef<HTMLAudioElement>(null)
  const activeLyricRef = useRef<HTMLButtonElement>(null)

  useEffect(() => {
    if (!files.length) {
      setSelectedFile(null)
    } else if (!selectedFile || !files.some((file) => file.library === selectedFile.library && file.path === selectedFile.path)) {
      setSelectedFile(files[0])
    }
  }, [files, selectedFile])

  useEffect(() => {
    setPlaying(false)
    setCurrentTime(0)
    setDuration(0)
    setLyrics([])
    setUslt('')
    setLanguage(null)
    setLyricsError(null)
    if (!selectedFile) return

    const controller = new AbortController()
    setLyricsLoading(true)
    fetch(apiPath('/api/music-lyrics', selectedFile), { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error(`Unable to read synchronized lyrics (${response.status})`)
        return response.json() as Promise<LyricsResponse>
      })
      .then((result) => {
        setLyrics(result.entries)
        setUslt(result.uslt)
        setLanguage(result.language)
      })
      .catch((reason) => {
        if (reason instanceof DOMException && reason.name === 'AbortError') return
        setLyricsError(reason instanceof Error ? reason.message : String(reason))
      })
      .finally(() => setLyricsLoading(false))

    return () => controller.abort()
  }, [selectedFile])

  const groupedFiles = Object.entries(
    files.reduce<Record<string, MusicFile[]>>((groups, file) => {
      ;(groups[file.group] ??= []).push(file)
      return groups
    }, {}),
  )

  const activeIndex = lyrics.findLastIndex((entry) => entry.time_ms <= currentTime * 1000)

  useEffect(() => {
    activeLyricRef.current?.scrollIntoView({ block: 'center', behavior: 'smooth' })
  }, [activeIndex])

  const togglePlayback = async () => {
    const audio = audioRef.current
    if (!audio) return
    if (audio.paused) {
      try {
        await audio.play()
      } catch (reason) {
        setLyricsError(reason instanceof Error ? reason.message : String(reason))
      }
    } else {
      audio.pause()
    }
  }

  const seek = (seconds: number) => {
    if (!audioRef.current) return
    audioRef.current.currentTime = seconds
    setCurrentTime(seconds)
  }

  const setAudioVolume = (nextVolume: number) => {
    if (audioRef.current) audioRef.current.volume = nextVolume
    setVolume(nextVolume)
  }

  return (
    <div className="music-layout">
      <section className="music-library">
        <div className="music-section-head">
          <div><span className="eyebrow">Input library</span><h2>Choose a track</h2></div>
          <button className="icon-button" type="button" title="Refresh audio files" aria-label="Refresh audio files" onClick={() => void onRefresh()}><RefreshCw /></button>
        </div>
        <div className="track-list">
          {groupedFiles.map(([group, groupFiles]) => (
            <details className="track-group" key={group} open>
              <summary className="track-group-heading"><span>{group}</span><small>{groupFiles.length}</small><ChevronDown /></summary>
              {groupFiles.map((file) => (
                <button key={`${file.library}:${file.path}`} type="button" className={selectedFile?.library === file.library && selectedFile.path === file.path ? 'selected' : ''} onClick={() => setSelectedFile(file)}>
                  <span className="track-icon"><FileAudio /></span>
                  <span><strong>{file.name}</strong><small>{file.library}/{file.path.split('/').slice(0, -1).join('/')}</small></span>
                  {selectedFile?.library === file.library && selectedFile.path === file.path && playing && <span className="playing-bars" aria-label="Playing"><i /><i /><i /></span>}
                </button>
              ))}
            </details>
          ))}
          {!files.length && <div className="empty-state"><Music2 /><h3>No audio files found</h3><p>Add music under input/ and refresh.</p></div>}
        </div>
      </section>

      <section className="music-player-panel">
        <div className="now-playing">
          <span className="album-mark"><Music2 /></span>
          <div><span className="eyebrow">Now playing</span><h2>{selectedFile?.name ?? 'Select a track'}</h2><p>{selectedFile ? `${selectedFile.library}/${selectedFile.path}` : 'Audio from input/ and output/songs/'}</p></div>
          {language && <span className="language-tag">{language}</span>}
        </div>

        {selectedFile ? (
          <>
            <audio
              ref={audioRef}
              src={apiPath('/api/music-audio', selectedFile)}
              preload="metadata"
              onLoadedMetadata={(event) => setDuration(event.currentTarget.duration)}
              onDurationChange={(event) => setDuration(event.currentTarget.duration)}
              onTimeUpdate={(event) => setCurrentTime(event.currentTarget.currentTime)}
              onPlay={() => setPlaying(true)}
              onPause={() => setPlaying(false)}
              onEnded={() => setPlaying(false)}
            />
            <div className="player-controls">
              <button className="play-button" type="button" onClick={() => void togglePlayback()} aria-label={playing ? 'Pause' : 'Play'} title={playing ? 'Pause' : 'Play'}>
                {playing ? <Pause /> : <Play />}
              </button>
              <span className="player-time">{durationLabel(currentTime)}</span>
              <input className="seek-slider" type="range" min="0" max={duration || 0} step="0.01" value={Math.min(currentTime, duration || 0)} onChange={(event) => seek(Number(event.target.value))} aria-label="Seek" />
              <span className="player-time">{durationLabel(duration)}</span>
              <Volume2 />
              <input className="volume-slider" type="range" min="0" max="1" step="0.05" value={volume} onChange={(event) => setAudioVolume(Number(event.target.value))} aria-label="Volume" />
            </div>
            <div className="lyrics-stage" aria-live="polite">
              <div className="lyrics-heading">
                <span>{lyricsMode === 'sylt' ? 'Synchronized lyrics' : 'Plain lyrics'}</span>
                <div className="lyrics-mode" role="group" aria-label="Lyrics format">
                  {(['sylt', 'uslt'] as const).map((mode) => (
                    <button key={mode} type="button" className={lyricsMode === mode ? 'active' : ''} aria-pressed={lyricsMode === mode} onClick={() => setLyricsMode(mode)}>{mode.toUpperCase()}</button>
                  ))}
                </div>
              </div>
              <div className={`lyrics-scroll ${lyricsMode}`}>
                {lyricsMode === 'sylt' && lyrics.map((entry, index) => (
                    <button
                      key={`${entry.time_ms}-${index}`}
                      ref={index === activeIndex ? activeLyricRef : undefined}
                      type="button"
                      className={index === activeIndex ? 'active' : index < activeIndex ? 'past' : ''}
                      onClick={() => seek(entry.time_ms / 1000)}
                    >
                      <time>{durationLabel(entry.time_ms / 1000)}</time>
                      <span>{entry.text}</span>
                    </button>
                  ))}
                {lyricsMode === 'uslt' && uslt && <pre className="uslt-text">{uslt}</pre>}
                {lyricsLoading && <div className="lyrics-message">Reading embedded lyrics...</div>}
                {!lyricsLoading && lyricsError && <div className="lyrics-message error">{lyricsError}</div>}
                {!lyricsLoading && !lyricsError && lyricsMode === 'sylt' && !lyrics.length && <div className="lyrics-message"><Music2 /><strong>No synchronized lyrics</strong><span>This file does not contain a millisecond-based SYLT frame.</span></div>}
                {!lyricsLoading && !lyricsError && lyricsMode === 'uslt' && !uslt && <div className="lyrics-message"><Music2 /><strong>No plain lyrics</strong><span>This file does not contain a USLT frame.</span></div>}
              </div>
            </div>
          </>
        ) : (
          <div className="empty-state"><Music2 /><h3>Select music to begin</h3></div>
        )}
      </section>
    </div>
  )
}
