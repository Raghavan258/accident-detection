import { RunBars } from "@/components/charts"
import { Card, Empty, TextInput } from "@/components/ui"
import { metric, when } from "@/lib/utils"
import { useStatus } from "@/services/queries"
import type { ExperimentRun } from "@/services/types"
import { useMemo, useState } from "react"

export function ExperimentsPage() {
  const status = useStatus()
  const [query, setQuery] = useState("")
  const [selected, setSelected] = useState<string | null>(null)
  const runs = status.data?.experiments ?? []
  const visible = useMemo(
    () => runs.filter((run) => `${run.run_id} ${run.run_name ?? ""} ${run.status}`.toLowerCase().includes(query.toLowerCase())),
    [runs, query],
  )
  const current = visible.find((run) => run.run_id === selected) ?? visible[0]
  if (status.isError) return <Empty>Experiments could not be loaded.</Empty>
  return (
    <div className="grid gap-4">
      <Card title="Validation comparison">
        <RunBars runs={visible.map((run) => ({ name: run.run_name || run.run_id.slice(0, 8), validation: run.validation }))} />
      </Card>
      <Card title="Runs" action={<TextInput className="w-48" placeholder="Filter" value={query} onChange={(event) => setQuery(event.target.value)} />}>
        {visible.length === 0 ? <Empty>No runs match.</Empty> : null}
        <table className="w-full text-left text-sm">
          <thead className="text-xs text-[#9aa8bd]">
            <tr>
              <th>Run</th>
              <th>Status</th>
              <th>Model</th>
              <th>Validation F1</th>
              <th>Started</th>
            </tr>
          </thead>
          <tbody>
            {visible.map((run) => (
              <tr key={run.run_id} className="cursor-pointer border-t border-[#243049]" onClick={() => setSelected(run.run_id)}>
                <td className="py-2">{run.run_name || run.run_id}</td>
                <td>{run.status}</td>
                <td>{run.params.selected_model ?? "N/A"}</td>
                <td>{metric(run.validation.f1_crash)}</td>
                <td>{when(run.start_time)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
      {current ? <RunDetail run={current} /> : null}
    </div>
  )
}

function RunDetail({ run }: { run: ExperimentRun }) {
  return (
    <Card title="Run detail">
      <p className="text-xs text-[#9aa8bd]">{run.run_id}</p>
      <div className="mt-3 grid gap-4 md:grid-cols-2">
        <MetricList title="Validation" values={run.validation} />
        <MetricList title="Test, scored after the gate" values={run.test} />
      </div>
      <p className="mt-3 text-sm">Threshold {run.params.threshold ?? "N/A"}</p>
      <p className="text-sm">Criterion {run.params.threshold_criterion ?? "N/A"}</p>
    </Card>
  )
}

function MetricList({ title, values }: { title: string; values: ExperimentRun["validation"] }) {
  return (
    <div>
      <h3 className="text-sm">{title}</h3>
      <p>Precision {metric(values.precision_crash)}</p>
      <p>Recall {metric(values.recall_crash)}</p>
      <p>F1 {metric(values.f1_crash)}</p>
      <p>Average precision {metric(values.average_precision_crash)}</p>
      <p>False positives {metric(values.false_positives, 0)}</p>
      <p>False negatives {metric(values.false_negatives, 0)}</p>
    </div>
  )
}
