import { Button } from "@/components/ui"
import type { RangeColumn, RangeTier } from "@/services/api"
import { activeTier } from "@/sim/featureMath"

export function ColumnPills({ column, value, onPick }: { column: RangeColumn; value: number; onPick: (tier: RangeTier) => void }) {
  const current = activeTier(column, value)
  return (
    <div className="space-y-2 rounded-lg bg-[#0b1120] p-3">
      <div className="flex items-center justify-between text-sm">
        <span>{column.label}</span>
        <span className="text-[#9aa8bd]">{value.toFixed(2)} {column.unit}</span>
      </div>
      <div className="flex flex-wrap gap-2">
        {column.tiers.map((tier) => (
          <Button key={tier.id} className={current.id === tier.id ? tone(tier.id) : ""} onClick={() => onPick(tier)}>
            {tier.name} ({tier.min}–{tier.max} {column.unit})
          </Button>
        ))}
      </div>
      <input className="w-full accent-[#14b8a6]" type="range" min={column.tiers[0].min} max={column.tiers[column.tiers.length - 1].max} step={column.unit === "g" ? 0.1 : 1} value={clamp(value, column)} onChange={(event) => onPick({ ...current, value: Number(event.target.value) })} />
    </div>
  )
}

function tone(id: string): string {
  if (id === "critical") return "border-[#ef4444] text-[#ef4444]"
  if (id === "elevated") return "border-[#f59e0b] text-[#f59e0b]"
  return "border-[#14b8a6] text-[#14b8a6]"
}

function clamp(value: number, column: RangeColumn): number {
  const max = column.tiers[column.tiers.length - 1].max
  const min = column.tiers[0].min
  return Math.min(max, Math.max(min, value))
}
