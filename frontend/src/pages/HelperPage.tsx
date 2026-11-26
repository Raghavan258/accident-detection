import { CityMap } from "@/components/CityMap"
import { useLiveAlerts } from "@/components/LiveAlerts"
import { ProvenanceBanner } from "@/components/ProvenanceBanner"
import { Button } from "@/components/ui"
import { askHelper, dispatchAck, fetchDrivingRoute, getSimEvents, type DispatchHub, type HelperReply, type SimEvent } from "@/services/api"
import { useEffect, useState } from "react"

const CHIPS = [
  ["All Live Alerts", "Show latest live user alert"],
  ["Crashes in Koramangala", "Show crashes in Koramangala"],
  ["Severe >3.5g", "Show severe crashes"],
  ["Highest Density Zone", "Which locality has the highest crash density?"],
] as const

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
  const [tiles, setTiles] = useState<"dark" | "osm">("dark")
  const banner = live.alerts.find((event) => event.alert_triggered || event.sos_requested)
  const pins = merge(chat?.matched_events?.length ? chat.matched_events : events, live.alerts)
  const selected = pins.find((event) => event.event_id === focusId) ?? null

  useEffect(() => {
    const incoming = live.alerts.find((event) => event.alert_triggered || event.sos_requested)
    if (!incoming) return
    setFocusId(incoming.event_id)
    setFlyTo({ lat: incoming.latitude, lng: incoming.longitude, zoom: 16 })
  }, [live.alerts])

  useEffect(() => {
    void getSimEvents(label, locality).then((payload) => setEvents(payload.events)).catch(() => setEvents([]))
  }, [label, locality, live.alerts.length])

  return (
    <div className="grid min-h-[70vh] gap-3 md:grid-cols-[420px_1fr]">
      <aside className="flex max-h-[78vh] flex-col overflow-hidden rounded-lg border border-[rgba(148,163,184,0.12)] bg-[#0f1623] shadow-[inset_0_1px_0_rgba(255,255,255,0.05)]">
        <div className="sticky top-0 z-10 flex items-center justify-between gap-2 border-b border-[rgba(148,163,184,0.12)] bg-[#0f1623]/90 px-3 py-2 backdrop-blur-md">
          <div>
            <p className="text-xs tracking-[0.16em]">DISPATCH COPILOT</p>
            <p className={live.connected ? "text-[10px] text-[#10b981]" : "text-[10px] text-[#f59e0b]"}>{live.connected ? "SSE LIVE" : "SSE CONNECTING"}</p>
          </div>
          <ProvenanceBanner />
          <Button onClick={() => resetMap(setFocusId, setFlyTo, setRoute)}>Reset Map</Button>
        </div>
        <div className="space-y-3 overflow-auto p-3">
          <ChatPanel draft={draft} chat={chat} focusId={focusId} alerts={live.alerts} onDraft={setDraft} onAsk={(message) => void ask(message, setChat, setDraft, setFocusId, setFlyTo, setRoute, route?.etaMin)} onOpen={(event) => void navigate(event, setFocusId, setFlyTo, setRoute)} />
          <Filters label={label} locality={locality} onLabel={(value) => { setChat(null); setLabel(value) }} onLocality={(value) => { setChat(null); setLocality(value) }} />
          {banner ? <CrashBanner event={banner} onFocus={() => reveal(banner, setFocusId, setFlyTo)} onNavigate={() => void navigate(banner, setFocusId, setFlyTo, setRoute)} /> : null}
        </div>
      </aside>
      <div className="relative min-h-[32rem] overflow-hidden rounded-lg border border-[rgba(148,163,184,0.12)]">
        <CityMap events={pins} focusId={focusId} flyTo={flyTo} tiles={tiles} zones route={route?.line} hub={route?.hub} onSelect={(id) => { const event = pins.find((item) => item.event_id === id); if (event) reveal(event, setFocusId, setFlyTo) }} />
        <div className="absolute right-3 top-3 z-[500] flex gap-1">
          <Button className={tiles === "dark" ? "border-[#14b8a6] text-[#14b8a6]" : ""} onClick={() => setTiles("dark")}>Dark Tactical</Button>
          <Button className={tiles === "osm" ? "border-[#14b8a6] text-[#14b8a6]" : ""} onClick={() => setTiles("osm")}>Standard OSM</Button>
        </div>
        {selected ? <Inspector event={selected} route={route} onAck={() => void acknowledge(selected, route?.etaMin)} /> : null}
      </div>
    </div>
  )
}

function Filters({ label, locality, onLabel, onLocality }: { label: string; locality: string; onLabel: (value: string) => void; onLocality: (value: string) => void }) {
  return (
    <div className="flex gap-2">
      <select className="rounded border border-[rgba(148,163,184,0.12)] bg-[#060911] p-1 text-xs" value={label} onChange={(event) => onLabel(event.target.value)}>
        <option value="all">all</option>
        <option value="crash">crash</option>
        <option value="near_miss">near_miss</option>
        <option value="normal_driving">normal_driving</option>
      </select>
      <select className="rounded border border-[rgba(148,163,184,0.12)] bg-[#060911] p-1 text-xs" value={locality} onChange={(event) => onLocality(event.target.value)}>
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

function ChatPanel({ draft, chat, focusId, alerts, onDraft, onAsk, onOpen }: { draft: string; chat: HelperReply | null; focusId: string | null; alerts: SimEvent[]; onDraft: (value: string) => void; onAsk: (message: string) => void; onOpen: (event: SimEvent) => void }) {
  const cards = [...alerts, ...(chat?.matched_events ?? [])].slice(0, 8)
  return (
    <div className="space-y-2">
      <div className="flex gap-1 overflow-x-auto">
        {CHIPS.map(([label, query]) => <Button key={label} onClick={() => onAsk(query)}>{label}</Button>)}
      </div>
      <form className="flex gap-2" onSubmit={(event) => { event.preventDefault(); onAsk(draft) }}>
        <input className="min-w-0 flex-1 rounded border border-[rgba(148,163,184,0.12)] bg-[#060911] p-2 text-sm" value={draft} onChange={(event) => onDraft(event.target.value)} placeholder="Filter by zone, severity, or event id" />
        <Button type="submit">Ask</Button>
      </form>
      {chat ? <p className="rounded border border-[rgba(148,163,184,0.12)] p-2 text-sm">{chat.reply}</p> : null}
      {cards.map((event) => <IncidentCard key={`${event.event_id}-${event.observed_at ?? ""}`} event={event} active={event.event_id === focusId} onOpen={() => onOpen(event)} />)}
      <div className="flex flex-wrap gap-2">
        <Button onClick={() => onAsk("Dispatch Ambulance (108) & Notify User")}>Dispatch Ambulance (108)</Button>
        <Button onClick={() => onAsk("Send ETA update to driver")}>Send ETA</Button>
        <Button onClick={() => onAsk("Mark Scene Secured")}>Mark Scene Secured</Button>
      </div>
    </div>
  )
}

function IncidentCard({ event, active, onOpen }: { event: SimEvent; active: boolean; onOpen: () => void }) {
  return (
    <button className={`block w-full rounded-lg border bg-[#060911] p-2 text-left text-xs ${active ? "border-[#14b8a6]" : "border-[rgba(148,163,184,0.12)]"}`} onClick={onOpen}>
      <span className="font-mono">{event.event_id}</span> · {event.locality}
      <span className="mt-1 block font-mono tabular-nums text-[#9aa8bd]">Peak {event.peak_horizontal_g.toFixed(2)} g · Drop {event.gps_drop_kmh.toFixed(0)} km/h · Prob {(event.probability ?? 0).toFixed(4)} · {event.observed_at ?? "live"}</span>
    </button>
  )
}

function Inspector({ event, route, onAck }: { event: SimEvent; route: RouteView | null; onAck: () => void }) {
  return (
    <div className="absolute bottom-3 left-3 z-[500] w-72 rounded-lg border border-[rgba(148,163,184,0.12)] bg-[#0f1623]/90 p-3 text-xs shadow-[inset_0_1px_0_rgba(255,255,255,0.05)] backdrop-blur-md">
      <p className="font-mono">{event.event_id} · {event.locality}</p>
      <p className="mt-1 font-mono tabular-nums">{event.latitude.toFixed(5)}, {event.longitude.toFixed(5)}</p>
      <p className="mt-1 font-mono tabular-nums">Prob {(event.probability ?? 0).toFixed(4)} · {route ? `ETA ${route.etaMin.toFixed(0)} min · ${route.distanceKm.toFixed(1)} km` : "Route pending"}</p>
      {route ? <a className="mt-1 block text-[#14b8a6] underline" href={route.googleUrl} target="_blank" rel="noreferrer">Open in External Maps</a> : null}
      <Button className="mt-2 border-[#ef4444] text-[#ef4444]" onClick={onAck}>Acknowledge & Dispatch Unit</Button>
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
  setFlyTo({ lat: event.latitude, lng: event.longitude, zoom: 16 })
}

function resetMap(setFocusId: (value: string | null) => void, setFlyTo: (value: { lat: number; lng: number; zoom: number }) => void, setRoute: (value: RouteView | null) => void) {
  setFocusId(null)
  setRoute(null)
  setFlyTo({ lat: 12.97, lng: 77.59, zoom: 11 })
}

async function acknowledge(event: SimEvent, etaMin?: number) {
  try {
    await dispatchAck(event.event_id, "ambulance", etaMin)
  } catch {
    return
  }
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
