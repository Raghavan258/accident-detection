import { Badge, Card, Empty } from "@/components/ui"
import { useHealth } from "@/services/queries"

export function HealthPage() {
  const health = useHealth()
  if (health.isLoading) return <Empty>Checking services.</Empty>
  if (health.isError || !health.data) return <Empty>Health check failed. The API did not respond.</Empty>
  const data = health.data
  const rows = [
    ["FastAPI", data.fastapi.ok, "This response was returned by the API."],
    ["MLflow", data.mlflow.ok, data.mlflow.detail ?? `Production alias ${data.mlflow.production ?? "N/A"}`],
    ["SQLite", data.sqlite.ok, data.sqlite.detail ?? `${data.sqlite.sensor_events ?? "N/A"} sensor events at ${data.sqlite.path ?? "N/A"}`],
    ["Reports", data.reports.ok, data.reports.detail ?? `drift html ${String(data.reports.drift_html)} (${data.reports.drift_bytes ?? "N/A"} bytes)`],
    ["Model load", data.model.loaded, data.model.detail ?? `version ${data.model.version ?? "N/A"}`],
    ["Docker", data.docker.operational, data.docker.detail],
    ["Cloud", Boolean(data.cloud.enabled), data.cloud.note ?? `missing ${(data.cloud.missing ?? []).join(", ") || "none"}`],
  ]
  return (
    <div className="grid gap-3">
      <p className="text-xs text-[#9aa8bd]">Checked {data.checked_at}. Docker and cloud are unavailable unless the check says available.</p>
      {rows.map(([label, ok, detail]) => (
        <Card key={String(label)}>
          <div className="flex items-center justify-between">
            <h2 className="text-sm">{label}</h2>
            <Badge tone={ok ? "ok" : "bad"}>{ok ? "available" : "unavailable"}</Badge>
          </div>
          <p className="mt-2 text-sm text-[#9aa8bd]">{detail}</p>
        </Card>
      ))}
    </div>
  )
}
