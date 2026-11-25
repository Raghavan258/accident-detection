import type { SimEvent } from "@/services/api"
import type { CircleMarker as CircleMarkerLayer } from "leaflet"
import { useEffect, useRef } from "react"
import { CircleMarker, MapContainer, Polyline, Popup, TileLayer, useMap, useMapEvents } from "react-leaflet"
import "leaflet/dist/leaflet.css"

const COLORS: Record<string, string> = {
  crash: "#ef4444",
  near_miss: "#f59e0b",
  normal_driving: "#14b8a6",
}

type FlyTarget = { lat: number; lng: number; zoom: number } | null

type MapProps = {
  events: SimEvent[]
  focusId?: string | null
  route?: [number, number][]
  hub?: { lat: number; lng: number; name: string } | null
  pin?: { lat: number; lng: number } | null
  flyTo?: FlyTarget
  onPick?: (lat: number, lng: number) => void
  onSelect?: (eventId: string) => void
}

export function CityMap({ events, focusId, route, hub, pin, flyTo, onPick, onSelect }: MapProps) {
  return (
    <MapContainer center={[12.97, 77.59]} zoom={11} className="h-full w-full rounded-lg" scrollWheelZoom>
      <TileLayer attribution='&copy; OpenStreetMap' url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
      <FlyTo target={flyTo ?? null} />
      {onPick ? <Picker onPick={onPick} /> : null}
      {events.map((event) => (
        <EventPin key={event.event_id} event={event} active={event.event_id === focusId} onSelect={onSelect} />
      ))}
      {pin ? (
        <>
          <CircleMarker center={[pin.lat, pin.lng]} radius={16} className="pulse-ring" pathOptions={{ color: "#14b8a6", fillOpacity: 0.15, weight: 2 }} />
          <CircleMarker center={[pin.lat, pin.lng]} radius={8} pathOptions={{ color: "#14b8a6", fillColor: "#14b8a6", fillOpacity: 0.95 }} />
        </>
      ) : null}
      {hub ? <CircleMarker center={[hub.lat, hub.lng]} radius={8} pathOptions={{ color: "#e7ecf3", fillColor: "#14b8a6", fillOpacity: 1 }}><Popup>{hub.name}</Popup></CircleMarker> : null}
      {route && route.length > 1 ? <Polyline positions={route} pathOptions={{ color: "#14b8a6", weight: 5 }} /> : null}
    </MapContainer>
  )
}

function EventPin({ event, active, onSelect }: { event: SimEvent; active: boolean; onSelect?: (eventId: string) => void }) {
  const color = COLORS[event.label_name] ?? "#14b8a6"
  const marker = useRef<CircleMarkerLayer | null>(null)
  useEffect(() => {
    if (active) marker.current?.openPopup()
  }, [active])
  return (
    <CircleMarker
      ref={marker}
      center={[event.latitude, event.longitude]}
      radius={active ? 14 : 6}
      className={active ? "pulse-ring" : undefined}
      pathOptions={{ color, fillColor: color, fillOpacity: 0.85, weight: active ? 3 : 1 }}
      eventHandlers={{ click: () => onSelect?.(event.event_id) }}
    >
      <Popup>
        <strong>{event.event_id}</strong>
        <br />
        {event.locality} · {event.label_name}
        <br />
        {event.peak_horizontal_g.toFixed(2)} g · drop {event.gps_drop_kmh.toFixed(0)} km/h
      </Popup>
    </CircleMarker>
  )
}

function FlyTo({ target }: { target: FlyTarget }) {
  const map = useMap()
  const lat = target?.lat
  const lng = target?.lng
  const zoom = target?.zoom
  useEffect(() => {
    if (lat != null && lng != null && zoom != null) map.flyTo([lat, lng], zoom)
  }, [map, lat, lng, zoom])
  return null
}

function Picker({ onPick }: { onPick: (lat: number, lng: number) => void }) {
  useMapEvents({ click: (event) => onPick(event.latlng.lat, event.latlng.lng) })
  return null
}
