import { streamUrl, type SimEvent } from "@/services/api"
import { createContext, useContext, useEffect, useRef, useState, type ReactNode } from "react"
import { useLocation } from "react-router-dom"

type LiveAlertState = {
  connected: boolean
  alerts: SimEvent[]
  unread: number
  updates: Record<string, DispatchUpdate>
}

export type DispatchUpdate = { event_id: string; action?: string; message: string; eta_min?: number | null }

const LiveAlertContext = createContext<LiveAlertState>({ connected: false, alerts: [], unread: 0, updates: {} })

export function LiveAlertProvider({ children }: { children: ReactNode }) {
  const location = useLocation()
  const path = useRef(location.pathname)
  const [connected, setConnected] = useState(false)
  const [alerts, setAlerts] = useState<SimEvent[]>([])
  const [unread, setUnread] = useState(0)
  const [updates, setUpdates] = useState<Record<string, DispatchUpdate>>({})
  path.current = location.pathname

  useEffect(() => {
    if (location.pathname === "/helper") setUnread(0)
  }, [location.pathname])

  useEffect(() => {
    if (typeof EventSource === "undefined") return
    const source = new EventSource(streamUrl())
    source.addEventListener("ready", () => setConnected(true))
    source.addEventListener("alert", (message) => acceptAlert(message, path.current, setAlerts, setUnread, setUpdates))
    source.onerror = () => setConnected(false)
    return () => source.close()
  }, [])

  return <LiveAlertContext.Provider value={{ connected, alerts, unread, updates }}>{children}</LiveAlertContext.Provider>
}

export function useLiveAlerts(): LiveAlertState {
  return useContext(LiveAlertContext)
}

function acceptAlert(
  message: Event,
  path: string,
  setAlerts: (value: SimEvent[] | ((current: SimEvent[]) => SimEvent[])) => void,
  setUnread: (value: number | ((count: number) => number)) => void,
  setUpdates: (value: Record<string, DispatchUpdate> | ((current: Record<string, DispatchUpdate>) => Record<string, DispatchUpdate>)) => void,
) {
  const event = JSON.parse((message as MessageEvent).data) as SimEvent & DispatchUpdate & { kind?: string }
  if (event.kind === "dispatch") {
    setUpdates((current) => ({ ...current, [event.event_id]: event }))
    return
  }
  setAlerts((current) => [event, ...current].slice(0, 20))
  if (path !== "/helper") setUnread((count) => count + 1)
  chime()
}

function chime() {
  try {
    const audio = window.AudioContext
    if (!audio) return
    const context = new audio()
    const tone = context.createOscillator()
    const gain = context.createGain()
    tone.frequency.value = 880
    gain.gain.value = 0.04
    tone.connect(gain).connect(context.destination)
    tone.start()
    tone.stop(context.currentTime + 0.18)
  } catch {
    return
  }
}
