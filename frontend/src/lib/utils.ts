import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function metric(value: number | null | undefined, digits = 4): string {
  if (typeof value !== "number" || Number.isNaN(value)) return "N/A"
  return value.toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: 0 })
}

export function when(value: number | string | null | undefined): string {
  if (value == null || value === "") return "N/A"
  const date = typeof value === "number" ? new Date(value) : new Date(value)
  if (Number.isNaN(date.getTime())) return "N/A"
  return date.toLocaleString()
}
