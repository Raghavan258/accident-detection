import { ConfusionGrid, SplitBars } from "@/components/charts"
import { Card, Empty } from "@/components/ui"
import { metric } from "@/lib/utils"
import { useStatus } from "@/services/queries"
import type { MetricSlice } from "@/services/types"

export function PerformancePage() {
  const status = useStatus()
  if (status.isError || !status.data) return <Empty>Performance data could not be loaded.</Empty>
  const run = status.data.experiments[0]
  const gate = status.data.retraining?.gate
  return (
    <div className="grid gap-4">
      <Card title="Production candidate">
        <p className="text-sm">Selected model {run?.params.selected_model ?? "N/A"}</p>
        <p className="text-sm">Threshold {run?.params.threshold ?? "N/A"}</p>
        <p className="mt-2 text-xs text-[#9aa8bd]">
          The tracked retrain was compared with the frozen Phase 4 test result and matched it. This page does not treat that match as an improvement. Test numbers below were scored after the validation gate.
        </p>
      </Card>
      <div className="grid gap-4 lg:grid-cols-2">
        <Slice title="Validation" values={status.data.validation} />
        <Slice title="Test" values={status.data.test} />
      </div>
      <Card title="Comparison">
        <SplitBars validation={status.data.validation ?? {}} test={status.data.test ?? {}} />
      </Card>
      <div className="grid gap-4 lg:grid-cols-2">
        <ConfusionGrid matrix={run?.validation_confusion_matrix ?? null} caption="Validation confusion" />
        <ConfusionGrid matrix={run?.test_confusion_matrix ?? null} caption="Test confusion" />
      </div>
      <Card title="Gate">
        <p className="text-sm">Passed {String(gate?.passed ?? "N/A")}</p>
        <p className="text-sm">Minimum validation AP {metric(gate?.min_validation_average_precision, 2)}</p>
        <p className="text-sm">Minimum validation recall {metric(gate?.min_validation_recall, 2)}</p>
        <p className="text-sm">Used test labels {String(gate?.used_test_labels ?? "N/A")}</p>
      </Card>
    </div>
  )
}

function Slice({ title, values }: { title: string; values: MetricSlice }) {
  return (
    <Card title={title}>
      <p>Precision {metric(values?.precision_crash)}</p>
      <p>Recall {metric(values?.recall_crash)}</p>
      <p>F1 {metric(values?.f1_crash)}</p>
      <p>Average precision {metric(values?.average_precision_crash)}</p>
      <p>False positives {metric(values?.false_positives, 0)}</p>
      <p>False negatives {metric(values?.false_negatives, 0)}</p>
    </Card>
  )
}
