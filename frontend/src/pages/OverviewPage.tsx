import { RunBars, SplitBars } from "@/components/charts"
import { Badge, Card, Empty, Skeleton } from "@/components/ui"
import { metric } from "@/lib/utils"
import { useStatus } from "@/services/queries"

export function OverviewPage() {
  const status = useStatus()
  if (status.isLoading) return <Skeleton className="h-40" />
  if (status.isError || !status.data) return <Empty>Status could not be loaded.</Empty>
  const data = status.data
  const validation = data.validation ?? {}
  const gate = data.retraining?.gate
  const cards = [
    ["Production model", data.registry.name || "N/A"],
    ["Version", data.registry.production ?? "N/A"],
    ["Validation precision", metric(validation.precision_crash)],
    ["Validation recall", metric(validation.recall_crash)],
    ["Validation F1", metric(validation.f1_crash)],
    ["Validation AP", metric(validation.average_precision_crash)],
    ["Prediction latency", "N/A"],
    ["Total predictions", "N/A"],
  ]
  return (
    <div className="grid gap-4">
      <div className="grid gap-3 md:grid-cols-4">
        {cards.map(([label, value]) => (
          <Card key={label}>
            <p className="text-xs text-[#9aa8bd]">{label}</p>
            <p className="mt-1 text-lg">{value}</p>
          </Card>
        ))}
      </div>
      <div className="grid gap-4 lg:grid-cols-2">
        <Card title="Validation versus test">
          <p className="mb-2 text-xs text-[#9aa8bd]">Test metrics are recorded after the gate. They are not a deployment input.</p>
          <SplitBars validation={validation} test={data.test ?? {}} />
        </Card>
        <Card title="Experiment comparison (validation)">
          <RunBars
            runs={(data.experiments ?? []).map((run) => ({
              name: run.params?.selected_model || run.run_name || run.run_id.slice(0, 8),
              validation: run.validation ?? {},
            }))}
          />
        </Card>
      </div>
      <div className="grid gap-4 lg:grid-cols-3">
        <Card title="Deployment">
          <p>Promoted: {String(data.deployment?.promoted ?? "N/A")}</p>
          <p className="mt-2">Gate: {gate?.passed == null ? "N/A" : gate.passed ? "passed" : "failed"}</p>
          <p className="mt-2 text-xs text-[#9aa8bd]">Used test labels: {String(gate?.used_test_labels ?? "N/A")}</p>
        </Card>
        <Card title="Data quality">
          <Badge tone={schemaTone(data.quality?.passed)}>{schemaLabel(data.quality?.passed)}</Badge>
          <p className="mt-2 text-sm">Integrity suite: {String(data.quality?.deepchecks?.integrity_passed ?? "N/A")}</p>
          <p className="text-sm">Train/test suite: {String(data.quality?.deepchecks?.train_test_passed ?? "N/A")}</p>
        </Card>
        <Card title="Monitoring">
          <Badge tone={data.drift?.drift_detected ? "warn" : "ok"}>
            {data.drift?.drift_detected ? "Drift flagged" : "No drift flag"}
          </Badge>
          <p className="mt-2 text-sm">{(data.drift?.drifted_features ?? []).join(", ") || "No drifted features listed."}</p>
        </Card>
      </div>
      <Card title="Recent predictions">
        <Empty>Prediction requests are not stored. This count stays N/A.</Empty>
      </Card>
    </div>
  )
}

function schemaTone(passed: boolean | undefined): "ok" | "warn" | "idle" {
  if (passed === true) return "ok"
  if (passed === false) return "warn"
  return "idle"
}

function schemaLabel(passed: boolean | undefined): string {
  if (passed === true) return "Schema passed"
  if (passed === false) return "Schema not passed"
  return "Schema report not loaded"
}
