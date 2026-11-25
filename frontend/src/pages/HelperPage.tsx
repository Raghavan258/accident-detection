import { CityMap } from "@/components/CityMap"
import { useLiveAlerts } from "@/components/LiveAlerts"
import { ProvenanceBanner } from "@/components/ProvenanceBanner"
import { Button } from "@/components/ui"
import { askHelper, dispatchAck, fetchDrivingRoute, getSimEvents, type DispatchHub, type HelperReply, type SimEvent } from "@/services/api"
import { useEffect, useState } from "react"

const CHIPS = [
  "Show latest live user alert",
  "Show severe crashes in Koramangala",
  "Filter small accidents (1-2g)",
  "Which locality has the highest crash density?",
  "Navigate to latest accident",
]

type RouteView = { line: [number, number][]; etaMin: number; distanceKm: number; mapsUrl: string; googleUrl: string; hub: DispatchHub }

export function HelperPage() {
  const live = useLiveAlerts()
  const [events, setEvents] = useState<SimEvent[]>([])
  const [label, setLabel] = useState("crash")
  const [locality, setLocality] = useState("all")
  const [focusId, setFocusId] = useState<string | null>(null)
  const [flyTo, setFlyTo] = useState<{ lat: number; lng: number; zoom: number } | null>(null)
  const [chat, setChat] = useState<HelperReply | null>(null)
  const [route, setRoute] = useState<RouteView | null>(null)
  const [draft, setDraft] = useState("")
  const banner = live.alerts.find((event) => event.alert_triggered || event.sos_requested)
  const pins = merge(chat?.matched_events?.length ? chat.matched_events : events, live.alerts)

  useEffect(() => {
    const incoming = live.alerts.find((event) => event.alert_triggered || event.sos_requested)
    if (!incoming) return
    setFocusId(incoming.event_id)
    setFlyTo({ lat: incoming.latitude, lng: incoming.longitude, zoom: 15 })
  }, [live.alerts])

  useEffect(() => {
    void getSimEvents(label, locality).then((payload) => setEvents(payload.events)).catch(() => setEvents([]))
  }, [label, locality, live.alerts.length])

  return (
    <div className="space-y-3 rounded-xl bg-[#0b1120] p-4 text-[#e7ecf3]">
      <ProvenanceBanner />
      <div className="flex items-center justify-between text-xs">
        <span className={live.connected ? "text-[#14b8a6]" : "text-[#f59e0b]"}>{live.connected ? "LIVE STREAM CONNECTED" : "LIVE STREAM CONNECTING"}</span>
        <Filters label={label} locality={locality} onLabel={(value) => { setChat(null); setLabel(value) }} onLocality={(value) => { setChat(null); setLocality(value) }} />
      </div>
      {banner ? <CrashBanner event={banner} onFocus={() => reveal(banner, setFocusId, setFlyTo)} onNavigate={() => void navigate(banner, setFocusId, setFlyTo, setRoute)} /> : null}
      <div className="grid h-[70vh] min-h-[32rem] gap-3 min-[880px]:grid-cols-[60%_40%]">
        <div className="min-h-0">
          <CityMap events={pins} focusId={focusId} flyTo={flyTo} route={route?.line} hub={route?.hub} onSelect={(id) => setFocusId(id)} />
        </div>
        <aside className="space-y-3 overflow-auto rounded-lg bg-[#111827] p-3">
          <ChatPanel draft={draft} chat={chat} onDraft={setDraft} onAsk={(message) => void ask(message, setChat, setDraft, setFocusId, setFlyTo, setRoute, route?.etaMin)} onOpen={(event) => void navigate(event, setFocusId, setFlyTo, setRoute)} />
          {route ? <p className="text-sm">ETA {route.etaMin.toFixed(0)} min · {route.distanceKm.toFixed(1)} km · <a className="text-[#14b8a6] underline" href={route.mapsUrl} target="_blank" rel="noreferrer">OpenStreetMap</a> · <a className="text-[#14b8a6] underline" href={route.googleUrl} target="_blank" rel="noreferrer">Google Maps</a></p> : null}
          <AlertQueue alerts={live.alerts} onOpen={(event) => void navigate(event, setFocusId, setFlyTo, setRoute)} />
        </aside>
      </div>
    </div>
  )
}

function Filters({ label, locality, onLabel, onLocality }: { label: string; locality: string; onLabel: (value: string) => void; onLocality: (value: string) => void }) {
  return (
    <div className="flex gap-2">
      <select className="rounded bg-[#111827] p-1" value={label} onChange={(event) => onLabel(event.target.value)}>
        <option value="all">all</option>
        <option value="crash">crash</option>
        <option value="near_miss">near_miss</option>
        <option value="normal_driving">normal_driving</option>
      </select>
      <select className="rounded bg-[#111827] p-1" value={locality} onChange={(event) => onLocality(event.target.value)}>
        {["all", "Koramangala", "Indiranagar", "Whitefield", "Jayanagar", "Malleswaram", "Hebbal", "Electronic City", "Yelahanka"].map((name) => <option key={name}>{name}</option>)}
      </select>
    </div>
  )
}

function CrashBanner({ event, onFocus, onNavigate }: { event: SimEvent; onFocus: () => void; onNavigate: () => void }) {
  const severe = event.severity_tier === "Severe Crash"
  return (
    <div className="rounded-lg border p-3" style={{ borderColor: severe ? "#ef4444" : "#f59e0b", background: severe ? "#450a0a" : "#451a03" }}>
      <p className="text-sm font-semibold" style={{ color: severe ? "#ef4444" : "#f59e0b" }}>INCOMING ACCIDENT ALERT</p>
      <p className="text-sm">{event.event_id} · {event.locality} · {event.latitude.toFixed(5)}, {event.longitude.toFixed(5)}</p>
      <p className="text-sm">Probability {(event.probability ?? 0).toFixed(4)} vs {(event.threshold ?? 0.6152).toFixed(4)} · {event.severity_tier ?? event.label_name}</p>
      <p className="text-xs text-[#e7ecf3]">{(event.top_triggers ?? []).map((row) => `${row.label}: ${row.value} [${row.tier || row.sign}]`).join(" · ")}</p>
      <div className="mt-2 flex gap-2">
        <Button onClick={onFocus}>Focus & Highlight</Button>
        <Button onClick={onNavigate}>Navigate</Button>
      </div>
    </div>
  )
}

function ChatPanel({ draft, chat, onDraft, onAsk, onOpen }: { draft: string; chat: HelperReply | null; onDraft: (value: string) => void; onAsk: (message: string) => void; onOpen: (event: SimEvent) => void }) {
  return (
    <div className="space-y-2">
      <h2 className="text-sm font-medium">Requirement filter</h2>
      <div className="flex flex-wrap gap-2">
        {CHIPS.map((chip) => <Button key={chip} onClick={() => onAsk(chip)}>{chip}</Button>)}
        <Button onClick={() => onAsk("Dispatch Ambulance (108) & Notify User")}>Dispatch Ambulance (108) & Notify User</Button>
        <Button onClick={() => onAsk("Send ETA update to driver")}>Send ETA update to driver</Button>
        <Button onClick={() => onAsk("Mark Scene Secured")}>Mark Scene Secured</Button>
      </div>
      <form className="flex gap-2" onSubmit={(event) => { event.preventDefault(); onAsk(draft) }}>
        <input className="min-w-0 flex-1 rounded bg-[#0b1120] p-2 text-sm" value={draft} onChange={(event) => onDraft(event.target.value)} placeholder="Filter by zone, severity, or event id" />
        <Button type="submit">Ask</Button>
      </form>
      {chat ? <p className="text-sm">{chat.reply}</p> : null}
      {chat?.matched_events.slice(0, 6).map((event) => (
        <button key={event.event_id} className="block w-full rounded bg-[#0b1120] p-2 text-left text-xs" onClick={() => onOpen(event)}>
          {event.event_id} · {event.locality} · {event.label_name} · {event.peak_horizontal_g.toFixed(1)} g
        </button>
      ))}
    </div>
  )
}

function AlertQueue({ alerts, onOpen }: { alerts: SimEvent[]; onOpen: (event: SimEvent) => void }) {
  return (
    <div className="space-y-2">
      <h2 className="text-sm font-medium">Alert queue</h2>
      {alerts.length === 0 ? <p className="text-xs text-[#9aa8bd]">Waiting for a user transmission.</p> : null}
      {alerts.map((event) => (
        <button key={event.event_id} className="block w-full rounded p-2 text-left text-xs" style={{ background: event.label_name === "crash" ? "#450a0a" : "#451a03" }} onClick={() => onOpen(event)}>
          {event.event_id} · {event.label_name} · {event.locality}
        </button>
      ))}
    </div>
  )
}

async function ask(
  message: string,
  setChat: (value: HelperReply) => void,
  setDraft: (value: string) => void,
  setFocusId: (value: string) => void,
  setFlyTo: (value: { lat: number; lng: number; zoom: number }) => void,
  setRoute: (value: RouteView | null) => void,
  etaMin?: number,
) {
  if (!message.trim()) return
  try {
    const reply = await askHelper(message.trim())
    setChat(reply)
    setDraft("")
    if (reply.focus_target) {
      setFocusId(reply.focus_target.event_id)
      setFlyTo(reply.focus_target)
    }
    if (reply.action === "ambulance" || reply.action === "eta" || reply.action === "secured") {
      if (reply.focus_target) await dispatchAck(reply.focus_target.event_id, reply.action, etaMin)
      return
    }
    const first = reply.matched_events[0]
    if (first) await navigate(first, setFocusId, setFlyTo, setRoute)
  } catch {
    setChat({ reply: "The filter request failed. Check that the API is running.", matched_events: [], focus_target: null, hub: null, synthetic: true })
  }
}

async function navigate(
  event: SimEvent,
  setFocusId: (value: string) => void,
  setFlyTo: (value: { lat: number; lng: number; zoom: number }) => void,
  setRoute: (value: RouteView | null) => void,
) {
  reveal(event, setFocusId, setFlyTo)
  if (!event.hub) return
  try {
    const payload = await fetchDrivingRoute(event.hub.lat, event.hub.lng, event.latitude, event.longitude)
    const leg = payload.routes?.[0]
    if (!leg) return
    setRoute({
      line: leg.geometry.coordinates.map(([lng, lat]) => [lat, lng]),
      etaMin: leg.duration / 60,
      distanceKm: leg.distance / 1000,
      mapsUrl: osmLink(event),
      googleUrl: googleLink(event),
      hub: event.hub,
    })
  } catch {
    setRoute(null)
  }
}

function reveal(event: SimEvent, setFocusId: (value: string) => void, setFlyTo: (value: { lat: number; lng: number; zoom: number }) => void) {
  setFocusId(event.event_id)
  setFlyTo({ lat: event.latitude, lng: event.longitude, zoom: 15 })
}

function osmLink(event: SimEvent): string {
  const hub = event.hub
  if (!hub) return "https://www.openstreetmap.org"
  return `https://www.openstreetmap.org/directions?engine=fossgis_osrm_car&route=${hub.lat},${hub.lng};${event.latitude},${event.longitude}`
}

function googleLink(event: SimEvent): string {
  const hub = event.hub
  if (!hub) return "https://www.google.com/maps"
  return `https://www.google.com/maps/dir/?api=1&origin=${hub.lat},${hub.lng}&destination=${event.latitude},${event.longitude}&travelmode=driving`
}

function merge(events: SimEvent[], alerts: SimEvent[]): SimEvent[] {
  const ids = new Set(events.map((event) => event.event_id))
  return [...alerts.filter((event) => !ids.has(event.event_id)), ...events]
}
