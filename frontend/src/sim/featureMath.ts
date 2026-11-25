import type { RangeColumn, RangePreset, RangeTier } from "@/services/api"

const SPEED = { normal: [40, 38], elevated: [45, 18], critical: [48, 0] } as const

export function linkedPeak(peak: number): Record<string, number> {
  return {
    peak_horizontal_g: peak,
    peak_magnitude_g: peak + 0.08,
    min_ax_g: -0.9 * peak,
    max_ax_g: 0.28 * peak,
    std_horizontal_g: Math.max(0.02, 0.12 * peak),
  }
}

export function linkedSpeed(start: number, end: number): Record<string, number> {
  return { gps_start_kmh: start, gps_end_kmh: end, gps_min_kmh: Math.min(start, end), gps_drop_kmh: start - end }
}

export function linkedSize(size: "Light" | "Medium" | "Heavy"): Record<string, number> {
  return { size_light: size === "Light" ? 1 : 0, size_medium: size === "Medium" ? 1 : 0, size_heavy: size === "Heavy" ? 1 : 0 }
}

export function speedForDrop(tierId: string): Record<string, number> {
  const pair = SPEED[tierId as keyof typeof SPEED] ?? SPEED.critical
  return linkedSpeed(pair[0], pair[1])
}

export function activeTier(column: RangeColumn, value: number): RangeTier {
  const match = [...column.tiers].reverse().find((tier) => value >= tier.min && value <= tier.max)
  return match ?? column.tiers[column.tiers.length - 1]
}

export function applyPreset(preset: RangePreset): { features: Record<string, number>; gyro: number } {
  return { features: preset.features, gyro: preset.gyro_deg_s }
}
