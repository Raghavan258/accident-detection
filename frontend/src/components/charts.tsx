import { metric } from "@/lib/utils"
import type { Confusion, MetricSlice } from "@/services/types"
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"

const rows = [
  ["precision_crash", "Precision"],
  ["recall_crash", "Recall"],
  ["f1_crash", "F1"],
  ["average_precision_crash", "Average precision"],
] as const

export function SplitBars({ validation, test }: { validation: MetricSlice; test: MetricSlice }) {
  const data = rows.map(([key, label]) => ({
    metric: label,
    Validation: validation[key] ?? null,
    Test: test[key] ?? null,
  }))
  if (data.every((row) => row.Validation == null && row.Test == null)) {
    return <p className="text-sm text-[#9aa8bd]">No validation or test metrics are in the latest run.</p>
  }
  return (
    <div className="h-64">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data}>
          <CartesianGrid stroke="#243049" vertical={false} />
          <XAxis dataKey="metric" stroke="#9aa8bd" fontSize={12} />
          <YAxis stroke="#9aa8bd" fontSize={12} domain={[0, 1]} />
          <Tooltip contentStyle={{ background: "#0d1524", border: "1px solid #243049" }} />
          <Legend />
          <Bar dataKey="Validation" fill="#7aa2f7" />
          <Bar dataKey="Test" fill="#3dd68c" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

export function RunBars({ runs }: { runs: { name: string; validation: MetricSlice }[] }) {
  const data = runs.map((run) => ({
    name: run.name,
    Precision: run.validation.precision_crash ?? null,
    Recall: run.validation.recall_crash ?? null,
    F1: run.validation.f1_crash ?? null,
    AP: run.validation.average_precision_crash ?? null,
  }))
  if (data.length === 0) return <p className="text-sm text-[#9aa8bd]">No experiment runs are recorded.</p>
  return (
    <div className="h-64">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data}>
          <CartesianGrid stroke="#243049" vertical={false} />
          <XAxis dataKey="name" stroke="#9aa8bd" fontSize={12} />
          <YAxis stroke="#9aa8bd" fontSize={12} domain={[0, 1]} />
          <Tooltip contentStyle={{ background: "#0d1524", border: "1px solid #243049" }} />
          <Legend />
          <Bar dataKey="Precision" fill="#7aa2f7" />
          <Bar dataKey="Recall" fill="#3dd68c" />
          <Bar dataKey="F1" fill="#e8b931" />
          <Bar dataKey="AP" fill="#c4b5fd" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

export function ConfusionGrid({ matrix, caption }: { matrix: Confusion | null; caption: string }) {
  if (!matrix) return <p className="text-sm text-[#9aa8bd]">{caption}: N/A</p>
  return (
    <div>
      <p className="mb-2 text-xs text-[#9aa8bd]">{caption}</p>
      <table className="text-sm">
        <tbody>
          {matrix.matrix.map((row, rowIndex) => (
            <tr key={matrix.labels[rowIndex] ?? rowIndex}>
              <th className="pr-3 text-left text-[#9aa8bd]">{matrix.labels[rowIndex] ?? rowIndex}</th>
              {row.map((value, column) => (
                <td key={column} className="border border-[#243049] px-3 py-1">
                  {metric(value, 0)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
