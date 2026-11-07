import { Badge, Card, Empty } from "@/components/ui"
import { metric } from "@/lib/utils"
import { driftReportUrl } from "@/services/api"
import { useStatus } from "@/services/queries"

export function MonitoringPage() {
  const status = useStatus()
  if (status.isError || !status.data) return <Empty>Monitoring data could not be loaded.</Empty>
  const data = status.data
  const checks = data.quality?.deepchecks
  const event = data.investigation?.extreme_event
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <Card title="Evidently drift">
        <Badge tone={data.drift?.drift_detected ? "warn" : "ok"}>
          {data.drift?.drift_detected == null ? "N/A" : data.drift.drift_detected ? "Drift flagged" : "Not flagged"}
        </Badge>
        <p className="mt-2 text-sm">Mean-shift threshold {metric(data.drift?.share_threshold, 2)}</p>
        <ul className="mt-2 text-sm">
          {(data.drift?.drifted_features ?? []).map((name) => (
            <li key={name}>{name}</li>
          ))}
        </ul>
        {(data.drift?.drifted_features ?? []).length === 0 ? <Empty>No drifted feature list.</Empty> : null}
        <a className="mt-3 inline-block text-sm underline" href={driftReportUrl()} target="_blank" rel="noreferrer">
          Open drift.html
        </a>
      </Card>
      <Card title="Deepchecks">
        <p className="text-sm">Available {String(checks?.available ?? "N/A")}</p>
        <p className="text-sm">Train/test suite {String(checks?.train_test_passed ?? "N/A")}</p>
        <p className="text-sm">Data integrity {String(checks?.integrity_passed ?? "N/A")}</p>
        <p className="mt-2 text-sm">Failed integrity conditions: {(checks?.integrity_not_passed ?? []).join(", ") || "none listed"}</p>
        <p className="mt-2 text-sm">Failed train/test conditions: {(checks?.train_test_not_passed ?? []).join(", ") || "none listed"}</p>
      </Card>
      <Card title="Recorded investigation">
        <p className="text-xs text-[#9aa8bd]">
          Source {data.investigation?.source ?? "N/A"}. Measured {data.investigation?.measured_on ?? "N/A"}. Recomputed on this request:{" "}
          {String(data.investigation?.recomputed_on_request ?? "N/A")}.
        </p>
        {event ? (
          <p className="mt-2 text-sm">
            Validation event {event.event_id} ({event.label_name}) peak_horizontal_g {metric(event.peak_horizontal_g, 2)} g. {event.note}
          </p>
        ) : (
          <Empty>No recorded extreme event.</Empty>
        )}
        <p className="mt-2 text-sm">
          Median peak, train {metric(data.investigation?.peak_horizontal_g_median_train, 3)} g, validation{" "}
          {metric(data.investigation?.peak_horizontal_g_median_validation, 3)} g.
        </p>
        <p className="mt-2 text-sm">{data.investigation?.correlation_note ?? ""}</p>
      </Card>
    </div>
  )
}
