import { AppShell } from "@/components/AppShell"
import { ExperimentsPage } from "@/pages/ExperimentsPage"
import { HealthPage } from "@/pages/HealthPage"
import { MonitoringPage } from "@/pages/MonitoringPage"
import { OverviewPage } from "@/pages/OverviewPage"
import { PerformancePage } from "@/pages/PerformancePage"
import { PredictionsPage } from "@/pages/PredictionsPage"
import { RegistryPage } from "@/pages/RegistryPage"
import { Navigate, Route, Routes } from "react-router-dom"

export default function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route index element={<Navigate to="/overview" replace />} />
        <Route path="/overview" element={<OverviewPage />} />
        <Route path="/predictions" element={<PredictionsPage />} />
        <Route path="/registry" element={<RegistryPage />} />
        <Route path="/experiments" element={<ExperimentsPage />} />
        <Route path="/monitoring" element={<MonitoringPage />} />
        <Route path="/performance" element={<PerformancePage />} />
        <Route path="/health" element={<HealthPage />} />
      </Route>
    </Routes>
  )
}
