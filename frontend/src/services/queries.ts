import { useQuery } from "@tanstack/react-query"
import { getExamples, getSchema, getStatus } from "./api"

export function useStatus() {
  return useQuery({ queryKey: ["status"], queryFn: getStatus })
}

export function useSchema() {
  return useQuery({ queryKey: ["schema"], queryFn: getSchema })
}

export function useExamples() {
  return useQuery({ queryKey: ["examples"], queryFn: getExamples })
}
