import { Badge, Button, Card, Empty, Hint, TextInput } from "@/components/ui"
import { ApiError, predictFeatures, predictWindow } from "@/services/api"
import { useExamples, useSchema } from "@/services/queries"
import type { ExampleRow, PredictionResponse } from "@/services/types"
import { useMutation } from "@tanstack/react-query"
import { useState, type ReactNode } from "react"
import { toast } from "sonner"

export function PredictionsPage() {
  const schema = useSchema()
  const examples = useExamples()
  const [values, setValues] = useState<Record<string, string>>({})
  const [windowText, setWindowText] = useState('{"gsensor":[[0.2,0.1,1.0]],"gps_speed":[40]}')
  const featurePredict = useMutation({
    mutationFn: predictFeatures,
    onError: (error: Error) => toast.error(error instanceof ApiError ? error.message : error.message),
  })
  const windowPredict = useMutation({
    mutationFn: predictWindow,
    onError: (error: Error) => toast.error(error instanceof ApiError ? error.message : error.message),
  })
  if (schema.isLoading) return <Empty>Loading feature schema.</Empty>
  if (schema.isError || !schema.data) return <Empty>Feature schema could not be loaded.</Empty>
  const fields = schema.data.features
  return (
    <div className="grid gap-4">
      <p className="text-[11px] tracking-[0.18em] text-[#99f6e4]">OFFICIAL VZCRASH • PROD v1</p>
      <div className="grid gap-4 md:grid-cols-2">
        <Card title="Event loader">
          <div className="mb-3 grid gap-2">
            {(examples.data?.rows ?? []).map((row) => (
              <Scenario key={row.event_id} row={row} onPick={() => setValues(asText(row.features))} />
            ))}
            <Button onClick={() => setValues({})}>Reset</Button>
          </div>
          <div className="grid gap-3 md:grid-cols-2">
            {fields.map((field) => (
              <label key={field.name} className="text-xs text-[#9aa8bd]">
                <Hint text={`${field.name}. Unit: ${field.unit}. Group: ${field.group}.`}>
                  <span>{field.group} · {field.label}</span>
                </Hint>
                <TextInput className="mt-1" inputMode="decimal" aria-label={field.name} value={values[field.name] ?? ""} onChange={(event) => setValues({ ...values, [field.name]: event.target.value })} />
              </label>
            ))}
          </div>
          <Button className="mt-4 border-[#14b8a6] bg-[#14b8a6] text-[#042f2e]" onClick={() => submitFeatures(fields.map((field) => field.name), values, featurePredict.mutate)}>
            Predict
          </Button>
        </Card>
        <div className="grid content-start gap-4">
          <Result result={featurePredict.data} pending={featurePredict.isPending} values={values} />
          <Card title="Sensor window">
            <p className="mb-2 text-xs text-[#9aa8bd]">JSON body for /predict/window. gsensor samples are [ax, ay, az].</p>
            <textarea className="h-28 w-full rounded-md border border-[rgba(148,163,184,0.12)] bg-[#060911] p-2 font-mono text-xs" value={windowText} onChange={(event) => setWindowText(event.target.value)} />
            <Button className="mt-2" onClick={() => submitWindow(windowText, windowPredict.mutate)}>Predict window</Button>
            <div className="mt-3"><Result result={windowPredict.data} pending={windowPredict.isPending} values={{}} /></div>
          </Card>
        </div>
      </div>
    </div>
  )
}

function Scenario({ row, onPick }: { row: ExampleRow; onPick: () => void }) {
  const peak = row.features.peak_horizontal_g
  const start = row.features.gps_start_kmh
  const end = row.features.gps_end_kmh
  return (
    <button className="rounded-lg border border-[rgba(148,163,184,0.12)] bg-[#060911] p-3 text-left hover:border-[#14b8a6]" onClick={onPick}>
      <span className="block text-sm">{row.title}</span>
      <span className="mt-1 block font-mono text-xs tabular-nums text-[#9aa8bd]">Peak {show(peak)} g · Speed {show(start)} → {show(end)} km/h</span>
    </button>
  )
}

function asText(features: Record<string, number | null>): Record<string, string> {
  return Object.fromEntries(Object.entries(features).map(([key, value]) => [key, value == null ? "" : String(value)]))
}

function show(value: number | null | undefined): string {
  return typeof value === "number" ? value.toFixed(2) : "N/A"
}

function submitFeatures(names: string[], values: Record<string, string>, mutate: (features: Record<string, number>) => void) {
  const features: Record<string, number> = {}
  for (const name of names) {
    if (values[name] == null || values[name].trim() === "") {
      toast.error(`${name} must be a finite number`)
      return
    }
    const parsed = Number(values[name])
    if (!Number.isFinite(parsed)) {
      toast.error(`${name} must be a finite number`)
      return
    }
    features[name] = parsed
  }
  mutate(features)
}

function submitWindow(text: string, mutate: (body: { gsensor: number[][]; gps_speed: number[]; gyro: number[][]; gyro_is_hd: boolean; vehiclesize: string | null }) => void) {
  try {
    const body = JSON.parse(text) as { gsensor?: number[][]; gps_speed?: number[] }
    if (!Array.isArray(body.gsensor) || !Array.isArray(body.gps_speed)) {
      toast.error("gsensor and gps_speed must be arrays")
      return
    }
    mutate({
      gsensor: body.gsensor,
      gps_speed: body.gps_speed,
      gyro: [],
      gyro_is_hd: false,
      vehiclesize: null,
    })
  } catch {
    toast.error("Window JSON is not valid")
  }
}

function Result({ result, pending, values }: { result?: PredictionResponse; pending: boolean; values: Record<string, string> }) {
  if (pending) return <p className="text-sm">Scoring…</p>
  if (!result) return <Empty>No prediction yet.</Empty>
  const crash = result.prediction === "crash"
  const width = Math.min(100, result.probability * 100)
  const mark = Math.min(100, result.threshold * 100)
  return (
    <Card title="Live classification">
      <p className="text-lg font-semibold" style={{ color: crash ? "#ef4444" : "#10b981", textShadow: crash ? "0 0 18px rgba(239,68,68,0.45)" : "0 0 18px rgba(16,185,129,0.35)" }}>
        {crash ? "CRASH DETECTED" : "NON-CRASH"}
      </p>
      <Badge tone={crash ? "bad" : "ok"}>{result.prediction}</Badge>
      <div className="relative mt-4 h-3 rounded bg-[#060911]">
        <div className="h-3 rounded" style={{ width: `${width}%`, background: crash ? "#ef4444" : "#14b8a6" }} />
        <span className="absolute top-[-4px] h-5 w-0.5 bg-white" style={{ left: `${mark}%` }} />
      </div>
      <p className="mt-2 font-mono text-xs tabular-nums text-[#9aa8bd]">0.00 · score <span>{result.probability}</span> · marker <span>{result.threshold}</span> · 1.00</p>
      <div className="mt-3 flex flex-wrap gap-2 text-xs">
        <Pill>Peak Accel: {values.peak_horizontal_g || "—"} g</Pill>
        <Pill>Speed Delta: {values.gps_start_kmh || "—"} → {values.gps_end_kmh || "—"} km/h</Pill>
        <Pill>Model: Prod v{result.model_version}</Pill>
      </div>
    </Card>
  )
}

function Pill({ children }: { children: ReactNode }) {
  return <span className="rounded-full border border-[rgba(148,163,184,0.12)] px-2 py-1 font-mono tabular-nums text-[#e7ecf3]">{children}</span>
}
