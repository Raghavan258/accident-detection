import type { ExampleRow, FeatureField, HealthPayload, PredictionResponse, StatusPayload } from "./types"

const base = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000"
const timeoutMs = 20000

export class ApiError extends Error {
  status: number
  constructor(status: number, detail: string) {
    super(detail)
    this.status = status
  }
}

export function apiBase(): string {
  return base
}

export async function getStatus(): Promise<StatusPayload> {
  return request<StatusPayload>("/mlops/status")
}

export async function getHealth(): Promise<HealthPayload> {
  return request<HealthPayload>("/mlops/health")
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

export function driftReportUrl(): string {
  return `${base}/mlops/reports/drift`
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(`${base}${path}`, {
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
