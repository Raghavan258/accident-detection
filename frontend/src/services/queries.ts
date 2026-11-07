import { useQuery } from "@tanstack/react-query"
import { getExamples, getHealth, getSchema, getStatus } from "./api"

export function useStatus() {
  return useQuery({ queryKey: ["status"], queryFn: getStatus })
}

export function useHealth() {
  return useQuery({ queryKey: ["health"], queryFn: getHealth })
}

export function useSchema() {
  return useQuery({ queryKey: ["schema"], queryFn: getSchema })
}

export function useExamples() {
  return useQuery({ queryKey: ["examples"], queryFn: getExamples })
}
