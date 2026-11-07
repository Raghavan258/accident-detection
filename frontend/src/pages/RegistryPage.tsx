import { Badge, Button, Card, ConfirmDialog, Empty } from "@/components/ui"
import { metric, when } from "@/lib/utils"
import { ApiError, setProduction } from "@/services/api"
import { useStatus } from "@/services/queries"
import { useMutation, useQueryClient } from "@tanstack/react-query"
import { useState } from "react"
import { toast } from "sonner"

export function RegistryPage() {
  const status = useStatus()
  const client = useQueryClient()
  const [pending, setPending] = useState<string | null>(null)
  const promote = useMutation({
    mutationFn: setProduction,
    onSuccess: async () => {
      toast.success("Production alias updated")
      setPending(null)
      await client.invalidateQueries({ queryKey: ["status"] })
    },
    onError: (error: Error) => {
      toast.error(error instanceof ApiError ? error.message : error.message)
      setPending(null)
    },
  })
  if (status.isError || !status.data) return <Empty>Registry could not be loaded.</Empty>
  const registry = status.data.registry
  return (
    <Card title={registry.name || "Registry"}>
      <p className="mb-3 text-sm text-[#9aa8bd]">
        Production {registry.production ?? "N/A"} · Staging {registry.staging ?? "N/A"}
      </p>
      {registry.versions.length === 0 ? <Empty>No registered versions.</Empty> : null}
      <table className="w-full text-left text-sm">
        <thead className="text-xs text-[#9aa8bd]">
          <tr>
            <th>Version</th>
            <th>Gate</th>
            <th>Threshold</th>
            <th>Created</th>
            <th>Alias</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {registry.versions.map((item) => {
            const passed = item.tags.gate_passed === "true"
            return (
              <tr key={item.version} className="border-t border-[#243049]">
                <td className="py-2">{item.version}</td>
                <td>{passed ? <Badge tone="ok">passed</Badge> : <Badge tone="bad">not passed</Badge>}</td>
                <td>{metric(Number(item.tags.threshold), 6)}</td>
                <td>{when(item.creation_timestamp)}</td>
                <td>{item.version === registry.production ? "production" : item.version === registry.staging ? "staging" : "none"}</td>
                <td>
                  <Button disabled={!passed || item.version === registry.production} onClick={() => setPending(item.version)}>
                    Set production
                  </Button>
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
      <ConfirmDialog
        open={pending != null}
        title="Move the production alias"
        body={`Point production at version ${pending}. Versions that failed the validation gate are rejected by the API.`}
        confirmLabel="Move alias"
        onCancel={() => setPending(null)}
        onConfirm={() => { if (pending) promote.mutate(pending) }}
      />
    </Card>
  )
}
