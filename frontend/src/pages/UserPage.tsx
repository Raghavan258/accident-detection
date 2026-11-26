import { CityMap } from "@/components/CityMap"
import { useLiveAlerts } from "@/components/LiveAlerts"
import { ProvenanceBanner } from "@/components/ProvenanceBanner"
import { Button } from "@/components/ui"
import { ApiError, getDispatchStatus, getSchemaRanges, simulateTelemetry, type RangeColumn, type RangePreset, type SchemaRanges, type SimEvent, type SimulateBody } from "@/services/api"
import { ColumnPills } from "@/sim/ColumnPills"
import { ExplainCard } from "@/sim/ExplainCard"
import { activeTier, linkedPeak, linkedSize, linkedSpeed, speedForDrop } from "@/sim/featureMath"
import { insideCity, nearestZone, zoneCenter, ZONES } from "@/sim/zones"
import { useEffect, useState } from "react"

const NUMERIC = ["peak_horizontal_g", "peak_magnitude_g", "min_ax_g", "max_ax_g", "std_horizontal_g", "gps_start_kmh", "gps_end_kmh", "gps_min_kmh", "gps_drop_kmh"]

export function UserPage() {
  const live = useLiveAlerts()
  const [schema, setSchema] = useState<SchemaRanges | null>(null)
  const [place, setPlace] = useState({ ...zoneCenter("Koramangala"), locality: "Koramangala" })
  const [stamp, setStamp] = useState(() => new Date().toISOString())
  const [features, setFeatures] = useState<Record<string, number>>({})
  const [gyro, setGyro] = useState(70)
  const [preset, setPreset] = useState<RangePreset | null>(null)
  const [result, setResult] = useState<SimEvent | null>(null)
  const [polled, setPolled] = useState("")
  const [error, setError] = useState("")
  const [pending, setPending] = useState(false)

  useEffect(() => {
    void getSchemaRanges().then((payload) => {
      const severe = payload.presets.find((item) => item.id === "crash") ?? payload.presets[0]
      setSchema(payload)
      setPreset(severe)
      setFeatures(severe.features)
      setGyro(severe.gyro_deg_s)
    }).catch(() => setError("Feature ranges could not be loaded."))
  }, [])

  useEffect(() => {
    if (!result?.event_id) return
    const timer = setInterval(() => {
      void getDispatchStatus(result.event_id).then((status) => { if (status.message) setPolled(status.message) })
    }, 4000)
    return () => clearInterval(timer)
  }, [result?.event_id])

  const status = (result && live.updates[result.event_id]?.message) || polled

  async function classify(force: boolean, nextFeatures?: Record<string, number>, nextPreset?: RangePreset, nextGyro?: number) {
    const usedFeatures = nextFeatures ?? features
    const usedPreset = nextPreset ?? preset
    const usedGyro = nextGyro ?? gyro
    if (!schema || !usedPreset) return
    setPending(true)
    setError("")
    try {
      setResult(await simulateTelemetry(bodyOf(place, usedFeatures, usedGyro, usedPreset, schema, force)))
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : "The telemetry request failed.")
    } finally {
      setPending(false)
    }
  }

  function move(lat: number, lng: number) {
    if (!insideCity(lat, lng)) {
      setError("That point is outside the Bengaluru simulation box.")
      return
    }
    setPlace({ lat, lng, locality: nearestZone(lat, lng) })
    setStamp(new Date().toISOString())
    setError("")
  }

  return (
    <div className="grid gap-4 md:grid-cols-2">
      <section className="space-y-3 rounded-lg border border-[rgba(148,163,184,0.12)] bg-[#0f1623] p-4 shadow-[inset_0_1px_0_rgba(255,255,255,0.05)]">
        <div className="flex items-center justify-between gap-2">
          <h1 className="text-sm font-semibold tracking-wide">Tactical map</h1>
          <ProvenanceBanner />
        </div>
        <GpsBar locality={place.locality} onZone={(name) => { const center = zoneCenter(name); move(center.lat, center.lng) }} onGps={() => readGps(move, setError)} />
        <div className="h-[28rem] overflow-hidden rounded-lg"><CityMap events={[]} pin={place} zones flyTo={{ ...place, zoom: 13 }} onPick={move} /></div>
        <p className="rounded border border-[#14b8a6]/40 px-3 py-2 font-mono text-xs tabular-nums text-[#99f6e4]">Lat {place.lat.toFixed(5)} · Lng {place.lng.toFixed(5)} · {place.locality} · {stamp}</p>
      </section>
      <section className="max-h-[52rem] space-y-3 overflow-auto rounded-lg border border-[rgba(148,163,184,0.12)] bg-[#0f1623] p-4 shadow-[inset_0_1px_0_rgba(255,255,255,0.05)]">
        {schema ? <FeatureStep schema={schema} features={features} gyro={gyro} onFeatures={(next) => { setFeatures(next); void classify(false, next) }} onGyro={(value) => { setGyro(value); void classify(false, undefined, undefined, value) }} onApply={(item) => { setFeatures(item.features); setPreset(item); setGyro(item.gyro_deg_s); void classify(false, item.features, item, item.gyro_deg_s) }} /> : <p className="text-sm">Loading the 12 model columns…</p>}
        <div className="flex flex-wrap gap-2">
          <Button className="border-[#14b8a6] bg-[#14b8a6] text-[#042f2e]" disabled={pending || !schema} onClick={() => void classify(false)}>{pending ? "Scoring…" : "Simulate & Classify"}</Button>
          <Button className="border-[#ef4444] text-[#ef4444]" disabled={pending || !schema} onClick={() => void classify(true)}>Send Emergency SOS Alert</Button>
        </div>
        {error ? <p className="text-sm text-[#ef4444]">{error}</p> : null}
        <ExplainCard result={result} status={status} />
        <DispatchStepper result={result} status={status} />
      </section>
    </div>
  )
}

function GpsBar({ locality, onZone, onGps }: { locality: string; onZone: (name: string) => void; onGps: () => void }) {
  return (
    <div className="flex flex-wrap gap-1.5">
      <Button onClick={onGps}>Use Browser GPS</Button>
      {ZONES.map((zone) => (
        <button key={zone.name} className={`rounded-full border px-2 py-1 text-[11px] ${locality === zone.name ? "border-[#14b8a6] text-[#14b8a6]" : "border-[rgba(148,163,184,0.12)] text-[#9aa8bd]"}`} onClick={() => onZone(zone.name)}>
          {zone.name}
        </button>
      ))}
    </div>
  )
}

function DispatchStepper({ result, status }: { result: SimEvent | null; status: string }) {
  const steps = [
    ["1. Telemetry Scored", Boolean(result)],
    ["2. Alert Broadcasted to Helper", Boolean(result?.broadcast || result?.sos_requested)],
    ["3. Responder Dispatched", /en route|secured|ambulance/i.test(status)],
  ] as const
  return (
    <ol className="grid gap-2">
      {steps.map(([label, on]) => (
        <li key={label} className={`rounded border px-3 py-2 text-xs ${on ? "border-[#10b981] text-[#10b981]" : "border-[rgba(148,163,184,0.12)] text-[#9aa8bd]"}`}>{label}</li>
      ))}
    </ol>
  )
}

function FeatureStep({ schema, features, gyro, onFeatures, onGyro, onApply }: { schema: SchemaRanges; features: Record<string, number>; gyro: number; onFeatures: (value: Record<string, number>) => void; onGyro: (value: number) => void; onApply: (preset: RangePreset) => void }) {
  function choosePreset(item: RangePreset) {
    onApply(item)
  }
  return (
    <div className="space-y-3">
      <h2 className="text-lg font-medium">Step B · Labeled feature ranges</h2>
      <div className="flex flex-wrap gap-2">
        {schema.presets.map((item) => <Button key={item.id} onClick={() => choosePreset(item)}>Apply Preset Across All Columns · {item.name}</Button>)}
      </div>
      {schema.columns.filter((column) => NUMERIC.includes(column.name)).map((column) => (
        <ColumnPills key={column.name} column={column} value={features[column.name] ?? 0} onPick={(tier) => onFeatures(nextFeatures(features, column, tier))} />
      ))}
      <SizeRow features={features} onFeatures={onFeatures} />
      <ColumnPills column={schema.gyro} value={gyro} onPick={(tier) => onGyro(tier.value)} />
      <p className="text-xs text-[#9aa8bd]">{schema.gyro.reason}</p>
    </div>
  )
}

function SizeRow({ features, onFeatures }: { features: Record<string, number>; onFeatures: (value: Record<string, number>) => void }) {
  const current = features.size_heavy === 1 ? "Heavy" : features.size_light === 1 ? "Light" : "Medium"
  return (
    <div className="flex flex-wrap gap-2">
      {(["Light", "Medium", "Heavy"] as const).map((size) => (
        <Button key={size} className={current === size ? "border-[#14b8a6] text-[#14b8a6]" : ""} onClick={() => onFeatures({ ...features, ...linkedSize(size) })}>{size} vehicle · {size === "Light" ? "Normal Range" : size === "Medium" ? "Elevated / Borderline" : "Critical Impact"}</Button>
      ))}
    </div>
  )
}

function nextFeatures(features: Record<string, number>, column: RangeColumn, tier: { id: string; value: number }): Record<string, number> {
  if (column.name === "peak_horizontal_g") return { ...features, ...linkedPeak(tier.value) }
  if (column.name === "gps_drop_kmh") return { ...features, ...speedForDrop(tier.id), gps_drop_kmh: tier.value }
  if (column.name === "gps_start_kmh") return { ...features, ...linkedSpeed(tier.value, features.gps_end_kmh ?? tier.value) }
  if (column.name === "gps_end_kmh") return { ...features, ...linkedSpeed(features.gps_start_kmh ?? tier.value, tier.value) }
  return { ...features, [column.name]: tier.value }
}

function bodyOf(place: { lat: number; lng: number; locality: string }, features: Record<string, number>, gyro: number, preset: RangePreset, schema: SchemaRanges, force: boolean): SimulateBody {
  const labels = Object.fromEntries(schema.columns.map((column) => [column.name, activeTier(column, features[column.name] ?? 0).name]))
  labels[schema.gyro.name] = activeTier(schema.gyro, gyro).name
  const size = features.size_heavy === 1 ? "Heavy" : features.size_light === 1 ? "Light" : "Medium"
  return {
    locality: place.locality,
    latitude: place.lat,
    longitude: place.lng,
    peak_g: features.peak_horizontal_g,
    speed_before: features.gps_start_kmh,
    speed_after: features.gps_end_kmh,
    vehicle_size: size,
    scenario_preset: preset.scenario_preset,
    features,
    gyro_deg_s: gyro,
    tier_labels: labels,
    force_broadcast: force,
  }
}

function readGps(move: (lat: number, lng: number) => void, setError: (value: string) => void) {
  if (!navigator.geolocation) {
    setError("This browser has no GPS API.")
    return
  }
  navigator.geolocation.getCurrentPosition(
    (position) => move(position.coords.latitude, position.coords.longitude),
    () => setError("Browser GPS is unavailable. Drop a pin inside Bengaluru."),
  )
}
