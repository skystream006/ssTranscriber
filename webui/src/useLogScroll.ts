import { useCallback, useLayoutEffect, useRef } from 'react'
import type { UIEvent } from 'react'

export default function useLogScroll(jobId: string | null | undefined, logs: string[] | undefined) {
  const elementRef = useRef<HTMLPreElement | null>(null)
  const following = useRef(true)
  const position = useRef(0)
  const previousJobId = useRef(jobId)

  const ref = useCallback((element: HTMLPreElement | null) => {
    elementRef.current = element
    if (element) {
      element.scrollTop = following.current ? element.scrollHeight : position.current
    }
  }, [])

  useLayoutEffect(() => {
    if (previousJobId.current !== jobId) {
      previousJobId.current = jobId
      following.current = true
      position.current = 0
    }
    const element = elementRef.current
    if (element && following.current) {
      element.scrollTop = element.scrollHeight
    }
  }, [jobId, logs])

  const onScroll = useCallback((event: UIEvent<HTMLPreElement>) => {
    const element = event.currentTarget
    position.current = element.scrollTop
    // Allow for fractional scroll positions at the bottom.
    following.current = element.scrollHeight - element.clientHeight - element.scrollTop <= 2
  }, [])

  return { ref, onScroll }
}
