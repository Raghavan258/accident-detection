import { driftReportUrl } from "@/services/api"
import { useQueryClient } from "@tanstack/react-query"
import {
  Activity,
  ClipboardCheck,
  FlaskConical,
  LayoutDashboard,
  Menu,
  PanelsTopLeft,
  RefreshCw,
  Radar,
  Stethoscope,
} from "lucide-react"
import { NavLink, Outlet } from "react-router-dom"
import { Badge, Button } from "./ui"
import { when } from "@/lib/utils"
import { useHealth, useStatus } from "@/services/queries"
import { useState } from "react"

const links = [
  ["/overview", "Overview", LayoutDashboard],
  ["/predictions", "Predictions", Stethoscope],
  ["/registry", "Model Registry", PanelsTopLeft],
  ["/experiments", "Experiments", FlaskConical],
  ["/monitoring", "Data Monitoring", Radar],
  ["/performance", "Model Performance", ClipboardCheck],
  ["/health", "System Health", Activity],
] as const

export function AppShell() {
  const status = useStatus()
  const health = useHealth()
  const client = useQueryClient()
  const [open, setOpen] = useState(true)
  const connected = health.data?.fastapi.ok && health.data.mlflow.ok
  return (
    <div className="flex min-h-screen">
      <aside className={`${open ? "w-56" : "w-14"} shrink-0 border-r border-[#243049] bg-[#0e1626]`}>
        <button className="flex h-14 items-center px-4 text-sm" onClick={() => setOpen(!open)} aria-label="Toggle sidebar">
          <Menu size={16} />
          {open ? <span className="ml-3">Accident Detection</span> : null}
        </button>
        <nav className="flex flex-col gap-1 px-2">
          {links.map(([path, label, Icon]) => (
            <NavLink
              key={path}
              to={path}
              className={({ isActive }) =>
                `flex items-center gap-3 rounded-md px-2 py-2 text-sm ${isActive ? "bg-[#1c2942] text-white" : "text-[#9aa8bd]"}`
              }
            >
              <Icon size={16} />
              {open ? <span>{label}</span> : null}
            </NavLink>
          ))}
        </nav>
      </aside>
      <div className="min-w-0 flex-1">
        <header className="flex h-14 items-center justify-between border-b border-[#243049] px-4">
          <div>
            <p className="text-sm font-medium">Accident Detection MLOps</p>
            <p className="text-xs text-[#9aa8bd]">Production version {status.data?.registry.production ?? "N/A"}</p>
          </div>
          <div className="flex items-center gap-3 text-xs text-[#9aa8bd]">
            <Badge tone={connected ? "ok" : "bad"}>{connected ? "Backend connected" : "Backend unavailable"}</Badge>
            <span>Updated {when(status.data?.generated_at)}</span>
            <Button onClick={() => void client.invalidateQueries()} aria-label="Refresh">
              <RefreshCw size={14} />
            </Button>
            <a className="underline" href={driftReportUrl()} target="_blank" rel="noreferrer">
              Drift HTML
            </a>
          </div>
        </header>
        <main className="p-4">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
