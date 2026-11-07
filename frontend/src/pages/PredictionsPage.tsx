import { Badge, Button, Card, Empty, Hint, TextInput } from "@/components/ui"
import { ApiError, predictFeatures, predictWindow } from "@/services/api"
import { useExamples, useSchema } from "@/services/queries"
import type { PredictionResponse } from "@/services/types"
import { useMutation } from "@tanstack/react-query"
import { useState } from "react"
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
    <div className="grid gap-4 lg:grid-cols-[1.2fr_0.8fr]">
      <Card title="Defensible features">
        <div className="mb-3 flex flex-wrap gap-2">
          {(examples.data?.rows ?? []).map((row) => (
            <Button
              key={row.event_id}
              onClick={() => setValues(Object.fromEntries(Object.entries(row.features).map(([key, value]) => [key, value == null ? "" : String(value)])))}
            >
              {row.title}
            </Button>
          ))}
          <Button onClick={() => setValues({})}>Reset</Button>
        </div>
        <div className="grid gap-3 md:grid-cols-2">
          {fields.map((field) => (
            <label key={field.name} className="text-xs text-[#9aa8bd]">
              <Hint text={`${field.name}. Unit: ${field.unit}. Group: ${field.group}.`}>
                <span>{field.label}</span>
              </Hint>
              <TextInput
                className="mt-1"
                inputMode="decimal"
                aria-label={field.name}
                value={values[field.name] ?? ""}
                onChange={(event) => setValues({ ...values, [field.name]: event.target.value })}
              />
            </label>
          ))}
        </div>
        <Button className="mt-4" onClick={() => submitFeatures(fields.map((field) => field.name), values, featurePredict.mutate)}>
          Predict
        </Button>
      </Card>
      <div className="grid gap-4">
        <Result result={featurePredict.data} pending={featurePredict.isPending} />
        <Card title="Sensor window">
          <p className="mb-2 text-xs text-[#9aa8bd]">JSON body for /predict/window. gsensor samples are [ax, ay, az].</p>
          <textarea
            className="h-28 w-full rounded-md border border-[#2a3958] bg-[#0d1524] p-2 text-xs"
            value={windowText}
            onChange={(event) => setWindowText(event.target.value)}
          />
          <Button className="mt-2" onClick={() => submitWindow(windowText, windowPredict.mutate)}>
            Predict window
          </Button>
          <div className="mt-3">
            <Result result={windowPredict.data} pending={windowPredict.isPending} />
          </div>
        </Card>
      </div>
    </div>
  )
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

function Result({ result, pending }: { result?: PredictionResponse; pending: boolean }) {
  if (pending) return <p className="text-sm">Scoring…</p>
  if (!result) return <Empty>No prediction yet.</Empty>
  const crash = result.prediction === "crash"
  return (
    <Card title="Result">
      <Badge tone={crash ? "bad" : "ok"}>{result.prediction}</Badge>
      <dl className="mt-3 grid grid-cols-2 gap-2 text-sm">
        <dt className="text-[#9aa8bd]">Probability</dt>
        <dd>{result.probability}</dd>
        <dt className="text-[#9aa8bd]">Threshold</dt>
        <dd>{result.threshold}</dd>
        <dt className="text-[#9aa8bd]">Version</dt>
        <dd>{result.model_version}</dd>
        <dt className="text-[#9aa8bd]">Latency</dt>
        <dd>{result.latency_ms} ms</dd>
        <dt className="text-[#9aa8bd]">Requested</dt>
        <dd>{new Date().toLocaleTimeString()}</dd>
      </dl>
    </Card>
  )
}
