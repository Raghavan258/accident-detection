import type { SimEvent } from "@/services/api"
import { Link } from "react-router-dom"

export function ExplainCard({ result, status }: { result: SimEvent | null; status: string }) {
  if (!result) return <section className="rounded-lg bg-[#111827] p-4 text-sm text-[#9aa8bd]">Choose a pin and a labeled range, then simulate to see why the frozen model decided.</section>
  const crash = result.prediction === "crash"
  const width = Math.min(100, (result.probability ?? 0) * 100)
  const mark = Math.min(100, (result.threshold ?? 0.6152) * 100)
  return (
    <section className="space-y-3 rounded-lg bg-[#111827] p-4">
      <p className="text-2xl font-semibold" style={{ color: crash ? "#ef4444" : "#14b8a6" }}>{result.prediction}</p>
      <p className="text-sm">{result.event_id} · probability {(result.probability ?? 0).toFixed(4)} · {result.severity_tier}</p>
      <div className="relative h-3 rounded bg-[#0b1120]">
        <div className="h-3 rounded" style={{ width: `${width}%`, background: crash ? "#ef4444" : "#14b8a6" }} />
        <span className="absolute top-[-4px] h-5 w-0.5 bg-white" style={{ left: `${mark}%` }} />
      </div>
      <p className="text-xs text-[#9aa8bd]">White marker is the frozen threshold {(result.threshold ?? 0.6152).toFixed(4)}.</p>
      <ContributionTable result={result} />
      <DispatchLine result={result} status={status} />
    </section>
  )
}

function ContributionTable({ result }: { result: SimEvent }) {
  const rows = result.contributions ?? []
  if (rows.length === 0) return <p className="text-sm text-[#9aa8bd]">Column shifts appear when Production Model v1 scores this vector.</p>
  return (
    <div className="space-y-2">
      {rows.map((row) => (
        <div key={row.name} className="grid grid-cols-[1.4fr_0.6fr_auto] items-center gap-2 text-xs">
          <span>{row.label}{row.tier ? ` · ${row.tier}` : ""}</span>
          <span className="text-[#9aa8bd]">{row.value}</span>
          <span style={{ color: row.sign === "+" ? "#ef4444" : "#14b8a6" }}>{row.sign === "+" ? "Toward Crash (+)" : "Toward Normal (-)"}</span>
        </div>
      ))}
    </div>
  )
}

function DispatchLine({ result, status }: { result: SimEvent; status: string }) {
  if (status) return <p className="text-sm text-[#14b8a6]">{status}</p>
  if (!result.broadcast && !result.sos_requested) return <p className="text-sm">Telemetry scored. Probability is under the frozen threshold, so no SOS was sent.</p>
  return (
    <div className="space-y-2 text-sm">
      <p>SOS Alert Dispatched to Helper Console</p>
      <Link className="text-[#14b8a6] underline" to="/helper">Open Helper View →</Link>
    </div>
  )
}
