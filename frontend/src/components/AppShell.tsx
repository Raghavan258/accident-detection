import { useLiveAlerts } from "@/components/LiveAlerts"
import { driftReportUrl } from "@/services/api"
import { useQueryClient } from "@tanstack/react-query"
import {
  Car,
  ClipboardCheck,
  FlaskConical,
  LayoutDashboard,
  Menu,
  MessageSquare,
  PanelsTopLeft,
  Radar,
  RefreshCw,
  Siren,
  Stethoscope,
} from "lucide-react"
import { NavLink, Outlet } from "react-router-dom"
import { Button } from "./ui"
import { when } from "@/lib/utils"
import { useStatus } from "@/services/queries"
import { useState } from "react"

const links = [
  ["/overview", "Overview", LayoutDashboard],
  ["/predictions", "Predictions", Stethoscope],
  ["/user", "User Simulator", Car],
  ["/helper", "Helper Dispatch", Siren],
  ["/registry", "Model Registry", PanelsTopLeft],
  ["/experiments", "Experiments", FlaskConical],
  ["/monitoring", "Data Monitoring", Radar],
  ["/performance", "Model Performance", ClipboardCheck],
  ["/chat", "Helper chat", MessageSquare],
] as const

export function AppShell() {
  const status = useStatus()
  const client = useQueryClient()
  const [open, setOpen] = useState(true)
  const alerts = useLiveAlerts()
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
              {path === "/helper" && alerts.unread > 0 ? <span className="ml-auto rounded bg-[#ef4444] px-1.5 text-[10px] text-white">{alerts.unread}</span> : null}
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
            <span>Updated {when(status.data?.generated_at)}</span>
            <Button onClick={() => void client.invalidateQueries()} aria-label="Refresh">
              <RefreshCw size={14} />
            </Button>
            <a className="underline" href={driftReportUrl()} target="_blank" rel="noreferrer">
              Drift HTML
            </a>
          </div>
        </header>
        <PrimaryTabs unread={alerts.unread} />
        <main className="p-4">
          <Outlet />
        </main>
      </div>
    </div>
  )
}

function PrimaryTabs({ unread }: { unread: number }) {
  const tabs = [
    ["/overview", "Overview"],
    ["/predictions", "Predictions"],
    ["/user", "User Simulator"],
    ["/helper", "Helper Dispatch"],
  ] as const
  return (
    <div className="flex gap-2 border-b border-[#243049] px-4 py-2">
      {tabs.map(([path, label]) => (
        <NavLink key={path} to={path} className={({ isActive }) => `rounded px-3 py-1 text-sm ${isActive ? "bg-[#14b8a6] text-[#042f2e]" : "text-[#9aa8bd]"}`}>
          {label}
          {path === "/helper" && unread > 0 ? <span className="ml-2 rounded bg-[#ef4444] px-1.5 text-[10px] text-white">{unread}</span> : null}
        </NavLink>
      ))}
    </div>
  )
}
