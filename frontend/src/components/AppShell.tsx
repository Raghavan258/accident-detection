import { useLiveAlerts } from "@/components/LiveAlerts"
import { cn } from "@/lib/utils"
import { useStatus } from "@/services/queries"
import { Activity, Car, LayoutDashboard, Siren, Stethoscope } from "lucide-react"
import { NavLink, Outlet } from "react-router-dom"

const tabs = [
  ["/overview", "Overview", LayoutDashboard],
  ["/predictions", "Predictions", Stethoscope],
  ["/user", "User Simulator", Car],
  ["/helper", "Helper Dispatch", Siren],
] as const

const more = [
  ["/registry", "Registry"],
  ["/experiments", "Experiments"],
  ["/monitoring", "Monitoring"],
  ["/performance", "Performance"],
  ["/chat", "Helper chat"],
] as const

export function AppShell() {
  const status = useStatus()
  const alerts = useLiveAlerts()
  const threshold = frozenLabel(status.data?.registry)
  return (
    <div className="min-h-screen bg-[#060911] text-[#e7ecf3]">
      <header className="sticky top-0 z-40 border-b border-[rgba(148,163,184,0.12)] bg-[#0f1623]/90 backdrop-blur-md">
        <div className="flex h-16 items-center gap-4 px-4">
          <Brand />
          <span className="hidden rounded-full border border-[rgba(148,163,184,0.12)] px-2 py-1 font-mono text-[10px] tabular-nums tracking-wide text-[#99f6e4] lg:inline">
            PROD v1 • THRESHOLD {threshold}
          </span>
          <nav className="ml-auto flex items-center gap-1">
            {tabs.map(([path, label, Icon]) => (
              <Tab key={path} path={path} label={label} icon={Icon} unread={path === "/helper" ? alerts.unread : 0} />
            ))}
          </nav>
          <Status api={status.isSuccess} sse={alerts.connected} />
        </div>
        <MoreLinks />
      </header>
      <main className="p-4">
        <Outlet />
      </main>
    </div>
  )
}

function Brand() {
  return (
    <div className="flex items-center gap-2">
      <span className="grid h-8 w-8 place-items-center rounded-md border border-[#14b8a6]/40 bg-[#14b8a6]/10 text-[#14b8a6]">
        <Activity size={16} />
      </span>
      <p className="whitespace-nowrap text-sm font-semibold tracking-wide">VZCrash Telemetry & Dispatch</p>
    </div>
  )
}

function Tab({ path, label, icon: Icon, unread }: { path: string; label: string; icon: typeof Car; unread: number }) {
  return (
    <NavLink
      to={path}
      className={({ isActive }) =>
        cn(
          "relative flex items-center gap-1.5 rounded-full px-3 py-1.5 text-xs",
          isActive ? "bg-[#14b8a6]/15 text-[#14b8a6] shadow-[inset_0_-2px_0_#14b8a6]" : "text-[#9aa8bd] hover:text-white",
        )
      }
    >
      <Icon size={14} />
      <span className="hidden md:inline">{label}</span>
      {unread > 0 ? <span className="live-badge rounded-full bg-[#ef4444] px-1.5 font-mono text-[10px] text-white">{unread}</span> : null}
    </NavLink>
  )
}

function Status({ api, sse }: { api: boolean; sse: boolean }) {
  return (
    <div className="hidden items-center gap-2 text-[10px] tracking-wide sm:flex">
      <Dot on={api} label={api ? "API Connected" : "API Offline"} />
      <Dot on={sse} label={sse ? "SSE Live" : "SSE Connecting"} />
    </div>
  )
}

function Dot({ on, label }: { on: boolean; label: string }) {
  return (
    <span className="inline-flex items-center gap-1 text-[#9aa8bd]">
      <span className={cn("h-1.5 w-1.5 rounded-full", on ? "bg-[#10b981]" : "bg-[#f59e0b]")} />
      {label}
    </span>
  )
}

function MoreLinks() {
  return (
    <div className="flex gap-3 border-t border-[rgba(148,163,184,0.08)] px-4 py-1 text-[10px] text-[#9aa8bd]">
      {more.map(([path, label]) => (
        <NavLink key={path} to={path} className={({ isActive }) => (isActive ? "text-[#14b8a6]" : "hover:text-white")}>
          {label}
        </NavLink>
      ))}
    </div>
  )
}

function frozenLabel(registry: { production: string | null; versions: { version: string; tags: Record<string, string> }[] } | undefined): string {
  const current = registry?.versions.find((item) => item.version === registry.production)
  const value = Number(current?.tags.threshold)
  return Number.isFinite(value) ? value.toFixed(4) : "0.6152"
}
