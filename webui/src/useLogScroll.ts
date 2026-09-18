import { useCallback, useLayoutEffect, useRef } from 'react'
import type { UIEvent } from 'react'

export default function useLogScroll(jobId: string | null | undefined, logs: string[] | undefined) {
  const elementRef = useRef<HTMLPreElement | null>(null)
  const following = useRef(true)
  const position = useRef(0)
  const restoring = useRef(false)
  const restoreFrame = useRef<number | null>(null)
  const previousJobId = useRef(jobId)

  const restoreScroll = useCallback((element: HTMLPreElement, scrollTop: number) => {
    restoring.current = true
    if (restoreFrame.current !== null) cancelAnimationFrame(restoreFrame.current)
    element.scrollTop = scrollTop
    restoreFrame.current = requestAnimationFrame(() => {
      restoring.current = false
      restoreFrame.current = null
    })
  }, [])

  const ref = useCallback((element: HTMLPreElement | null) => {
    elementRef.current = element
    if (element) {
      restoreScroll(element, following.current ? element.scrollHeight : position.current)
    }
  }, [restoreScroll])

  useLayoutEffect(() => {
    if (previousJobId.current !== jobId) {
      previousJobId.current = jobId
      following.current = true
      position.current = 0
    }
    const element = elementRef.current
    if (element && following.current) {
      restoreScroll(element, element.scrollHeight)
    }
  }, [jobId, logs, restoreScroll])

  useLayoutEffect(() => () => {
    if (restoreFrame.current !== null) cancelAnimationFrame(restoreFrame.current)
  }, [])

  const onScroll = useCallback((event: UIEvent<HTMLPreElement>) => {
    if (restoring.current) return
    const element = event.currentTarget
    position.current = element.scrollTop
    // Allow for fractional scroll positions at the bottom.
    following.current = element.scrollHeight - element.clientHeight - element.scrollTop <= 2
  }, [])

  return { ref, onScroll }
}
