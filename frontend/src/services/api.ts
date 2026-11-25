import type { ExampleRow, FeatureField, PredictionResponse, StatusPayload } from "./types"

const API_BASE = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000"
const timeoutMs = 20000

export class ApiError extends Error {
  status: number
  constructor(status: number, detail: string) {
    super(detail)
    this.status = status
  }
}

export function apiBase(): string {
  return API_BASE
}

export async function getStatus(): Promise<StatusPayload> {
  return request<StatusPayload>("/mlops/status")
}

export async function getSchema(): Promise<{ features: FeatureField[] }> {
  return request("/mlops/schema")
}

export async function getExamples(): Promise<{ rows: ExampleRow[] }> {
  return request("/mlops/examples")
}

export async function predictFeatures(features: Record<string, number>): Promise<PredictionResponse> {
  return request("/predict", { method: "POST", body: JSON.stringify({ features }) })
}

export async function predictWindow(body: {
  gsensor: number[][]
  gps_speed: number[]
  gyro: number[][]
  gyro_is_hd: boolean
  vehiclesize: string | null
}): Promise<PredictionResponse> {
  return request("/predict/window", { method: "POST", body: JSON.stringify(body) })
}

export async function setProduction(version: string): Promise<{ production: string }> {
  return request("/mlops/registry/production", { method: "POST", body: JSON.stringify({ version }) })
}

export type ChatReply = {
  synthetic: boolean
  reply: string
  locality: string | null
  label_filter: string | null
  matched_rows: number
  rows: Array<{
    event_id: number
    label_name: string
    locality: string
    latitude: number
    longitude: number
    observed_at: string
    peak_horizontal_g: number
  }>
  density: Array<{ locality: string; crashes: number; area_km2: number; crashes_per_km2: number }>
  emergency: Array<{ service: string; number: string; note: string }>
}

export async function askSynthetic(message: string): Promise<ChatReply> {
  return request("/synthetic/chat", { method: "POST", body: JSON.stringify({ message }) })
}

export type SimEvent = {
  event_id: string
  source?: string
  locality: string
  latitude: number
  longitude: number
  peak_horizontal_g: number
  gps_start_kmh: number
  gps_end_kmh: number
  gps_drop_kmh: number
  vehiclesize?: string
  scenario_preset?: string
  observed_at?: string
  prediction?: string
  probability?: number
  threshold?: number
  model_version?: string
  label_name: string
  alert_triggered?: boolean
  broadcast?: boolean
  hub?: DispatchHub
  synthetic?: boolean
  gyro_deg_s?: number | null
  severity_tier?: string
  sos_requested?: boolean
  tier_labels?: Record<string, string>
  contributions?: Contribution[]
  top_triggers?: Contribution[]
}

export type Contribution = {
  name: string
  label: string
  value: number
  delta: number
  direction: "toward_crash" | "toward_normal"
  sign: "+" | "-"
  tier?: string
}

export type RangeTier = { id: string; name: string; min: number; max: number; value: number }

export type RangeColumn = { name: string; label: string; unit: string; used_by_model: boolean; tiers: RangeTier[]; reason?: string }

export type RangePreset = {
  id: string
  name: string
  scenario_preset: SimulateBody["scenario_preset"]
  vehicle_size: SimulateBody["vehicle_size"]
  peak_g: number
  speed_before: number
  speed_after: number
  gyro_deg_s: number
  features: Record<string, number>
}

export type SchemaRanges = { columns: RangeColumn[]; feature_names: string[]; gyro: RangeColumn; presets: RangePreset[]; synthetic: boolean }

export type DispatchHub = {
  name: string
  zone: string
  lat: number
  lng: number
  traffic_contact: string
  public_numbers: string[]
}

export type LocalityStat = {
  locality: string
  total_events: number
  crash_count: number
  near_miss_count: number
  crash_rate_pct: number
  crashes_per_km2: number
  severity_tier: string
}

export type HelperReply = {
  reply: string
  matched_events: SimEvent[]
  focus_target: { lat: number; lng: number; zoom: number; event_id: string } | null
  hub: DispatchHub | null
  action?: string
  synthetic: boolean
}

export type SimulateBody = {
  locality: string
  latitude: number
  longitude: number
  peak_g: number
  speed_before: number
  speed_after: number
  vehicle_size: "Light" | "Medium" | "Heavy"
  scenario_preset: "crash" | "near_miss" | "normal"
  features?: Record<string, number>
  gyro_deg_s?: number
  tier_labels?: Record<string, string>
  force_broadcast?: boolean
}

export async function getSchemaRanges(): Promise<SchemaRanges> {
  return request("/api/v2/schema-ranges")
}

export async function dispatchAck(eventId: string, action: "ambulance" | "eta" | "secured", etaMin?: number): Promise<{ message: string }> {
  return request("/api/v2/dispatch-ack", { method: "POST", body: JSON.stringify({ event_id: eventId, action, eta_min: etaMin ?? null }) })
}

export async function getDispatchStatus(eventId: string): Promise<{ event_id: string; message: string }> {
  return request(`/api/v2/dispatch-status?event_id=${encodeURIComponent(eventId)}`)
}

export function streamUrl(): string {
  return `${API_BASE}/api/v2/stream`
}

export async function simulateTelemetry(body: SimulateBody): Promise<SimEvent> {
  return request("/api/v2/simulate", { method: "POST", body: JSON.stringify(body) })
}

export async function getSimEvents(label = "all", locality = "all"): Promise<{ events: SimEvent[]; localities: LocalityStat[] }> {
  return request(`/api/v2/events?label=${encodeURIComponent(label)}&locality=${encodeURIComponent(locality)}&include_live=true`)
}

export async function askHelper(message: string): Promise<HelperReply> {
  return request("/api/v2/helper-chat", { method: "POST", body: JSON.stringify({ message }) })
}

export function osrmUrl(hubLat: number, hubLng: number, crashLat: number, crashLng: number): string {
  return `https://router.project-osrm.org/route/v1/driving/${hubLng},${hubLat};${crashLng},${crashLat}?overview=full&geometries=geojson`
}

export async function fetchDrivingRoute(hubLat: number, hubLng: number, crashLat: number, crashLng: number): Promise<OsrmPayload> {
  const url = osrmUrl(hubLat, hubLng, crashLat, crashLng)
  try {
    const response = await fetch(url)
    if (!response.ok) throw new Error("osrm")
    return (await response.json()) as OsrmPayload
  } catch {
    const query = `hub_lat=${hubLat}&hub_lng=${hubLng}&crash_lat=${crashLat}&crash_lng=${crashLng}`
    const proxied = await request<{ route: OsrmPayload }>(`/api/v2/route?${query}`)
    return proxied.route
  }
}

export type OsrmPayload = {
  routes?: Array<{ distance: number; duration: number; geometry: { coordinates: [number, number][] } }>
}

export function driftReportUrl(): string {
  return `${API_BASE}/mlops/reports/drift`
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    signal: AbortSignal.timeout(timeoutMs),
    headers: { "Content-Type": "application/json", ...init.headers },
  })
  if (!response.ok) {
    throw new ApiError(response.status, await detail(response))
  }
  return response.json() as Promise<T>
}

async function detail(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: string }
    return body.detail ?? response.statusText
  } catch {
    return response.statusText
  }
}
