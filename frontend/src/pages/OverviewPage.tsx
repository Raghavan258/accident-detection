import { RunBars } from "@/components/charts"
import { Badge, Card, Empty, Skeleton } from "@/components/ui"
import { metric } from "@/lib/utils"
import { driftReportUrl } from "@/services/api"
import { useStatus } from "@/services/queries"
import type { MetricSlice, StatusPayload } from "@/services/types"
import { ExternalLink } from "lucide-react"

export function OverviewPage() {
  const status = useStatus()
  if (status.isLoading) return <Skeleton className="h-40" />
  if (status.isError || !status.data) return <Empty>Status could not be loaded.</Empty>
  const data = status.data
  const gate = data.retraining?.gate
  return (
    <div className="grid gap-4">
      <SourceLine />
      <Hero data={data} passed={gate?.passed === true} />
      <KpiGrid validation={data.validation ?? {}} test={data.test ?? {}} />
      <div className="grid gap-4 lg:grid-cols-2">
        <Parity validation={data.validation ?? {}} test={data.test ?? {}} />
        <Card title="Experiment comparison (validation)">
          <RunBars runs={(data.experiments ?? []).map((run) => ({ name: run.params?.selected_model || run.run_name || run.run_id.slice(0, 8), validation: run.validation ?? {} }))} />
        </Card>
      </div>
      <Governance data={data} />
    </div>
  )
}

function SourceLine() {
  return <p className="text-[11px] tracking-[0.18em] text-[#99f6e4]">OFFICIAL VZCRASH • PROD v1</p>
}

function Hero({ data, passed }: { data: StatusPayload; passed: boolean }) {
  return (
    <Card>
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="text-xs tracking-[0.16em] text-[#10b981]">{passed ? "MODEL GATE: PASSED" : "MODEL GATE: NOT PASSED"}</p>
          <h1 className="mt-1 text-xl font-semibold">Production Version {data.registry.production ?? "N/A"}</h1>
          <p className="mt-1 font-mono text-sm tabular-nums text-[#9aa8bd]">Frozen Decision Threshold: {thresholdOf(data)} (Locked Pre-Test)</p>
        </div>
        <a className="inline-flex items-center gap-1.5 rounded-md border border-[rgba(148,163,184,0.12)] px-3 py-1.5 text-sm text-[#e7ecf3] hover:border-[#14b8a6]" href={driftReportUrl()} target="_blank" rel="noreferrer">
          Drift HTML <ExternalLink size={14} />
        </a>
      </div>
    </Card>
  )
}

function KpiGrid({ validation, test }: { validation: MetricSlice; test: MetricSlice }) {
  const cards = [
    ["Precision", metric(test.precision_crash), test.precision_crash],
    ["Recall", metric(test.recall_crash), test.recall_crash],
    ["F1", metric(test.f1_crash), test.f1_crash],
    ["Average precision", metric(test.average_precision_crash), test.average_precision_crash],
  ] as const
  return (
    <div className="grid gap-3 md:grid-cols-4">
      {cards.map(([label, value, raw]) => (
        <Card key={label}>
          <p className="text-xs text-[#9aa8bd]">{label}</p>
          <p className="mt-1 font-mono text-2xl tabular-nums">{value}</p>
          <Meter value={typeof raw === "number" ? raw : 0} />
        </Card>
      ))}
      <Card>
        <p className="text-xs text-[#9aa8bd]">Validation precision</p>
        <p className="mt-1 font-mono text-lg tabular-nums">{metric(validation.precision_crash)}</p>
      </Card>
      <Card>
        <p className="text-xs text-[#9aa8bd]">Prediction latency</p>
        <p className="mt-1 font-mono text-lg tabular-nums">N/A</p>
      </Card>
      <Card>
        <p className="text-xs text-[#9aa8bd]">Total predictions</p>
        <p className="mt-1 font-mono text-lg tabular-nums">N/A</p>
      </Card>
    </div>
  )
}

function Meter({ value }: { value: number }) {
  return (
    <div className="mt-3 h-1.5 overflow-hidden rounded bg-[#060911]">
      <div className="h-full rounded bg-[#14b8a6]" style={{ width: `${Math.max(0, Math.min(100, value * 100))}%` }} />
    </div>
  )
}

function Parity({ validation, test }: { validation: MetricSlice; test: MetricSlice }) {
  return (
    <Card title="Validation versus test">
      <p className="mb-3 text-xs text-[#9aa8bd]">Test metrics are recorded after the gate. They are not a deployment input.</p>
      <Compare label="Precision" left={validation.precision_crash} right={test.precision_crash} />
      <Compare label="Recall" left={validation.recall_crash} right={test.recall_crash} />
    </Card>
  )
}

function Compare({ label, left, right }: { label: string; left?: number; right?: number }) {
  return (
    <div className="mb-3">
      <div className="mb-1 flex justify-between font-mono text-[11px] tabular-nums text-[#9aa8bd]">
        <span>{label}</span>
        <span>val {metric(left)} · test {metric(right)}</span>
      </div>
      <div className="grid grid-cols-2 gap-1">
        <div className="h-1.5 rounded bg-[#060911]"><div className="h-full rounded bg-[#f59e0b]" style={{ width: `${bar(left)}%` }} /></div>
        <div className="h-1.5 rounded bg-[#060911]"><div className="h-full rounded bg-[#14b8a6]" style={{ width: `${bar(right)}%` }} /></div>
      </div>
    </div>
  )
}

function Governance({ data }: { data: StatusPayload }) {
  const gate = data.retraining?.gate
  return (
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
        <Badge tone={data.drift?.drift_detected ? "warn" : "ok"}>{data.drift?.drift_detected ? "Drift flagged" : "No drift flag"}</Badge>
        <p className="mt-2 text-sm">{(data.drift?.drifted_features ?? []).join(", ") || "No drifted features listed."}</p>
      </Card>
    </div>
  )
}

function thresholdOf(data: StatusPayload): string {
  const current = data.registry.versions.find((item) => item.version === data.registry.production)
  const value = Number(current?.tags.threshold)
  return Number.isFinite(value) ? value.toFixed(4) : "N/A"
}

function bar(value: number | undefined): number {
  return typeof value === "number" ? Math.max(0, Math.min(100, value * 100)) : 0
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
