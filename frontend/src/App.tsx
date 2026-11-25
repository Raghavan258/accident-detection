import { AppShell } from "@/components/AppShell"
import { LiveAlertProvider } from "@/components/LiveAlerts"
import { ChatPage } from "@/pages/ChatPage"
import { HelperPage } from "@/pages/HelperPage"
import { ExperimentsPage } from "@/pages/ExperimentsPage"
import { MonitoringPage } from "@/pages/MonitoringPage"
import { OverviewPage } from "@/pages/OverviewPage"
import { PerformancePage } from "@/pages/PerformancePage"
import { PredictionsPage } from "@/pages/PredictionsPage"
import { RegistryPage } from "@/pages/RegistryPage"
import { UserPage } from "@/pages/UserPage"
import { Navigate, Route, Routes } from "react-router-dom"

export default function App() {
  return (
    <LiveAlertProvider>
      <Routes>
        <Route element={<AppShell />}>
          <Route index element={<Navigate to="/overview" replace />} />
          <Route path="/overview" element={<OverviewPage />} />
          <Route path="/predictions" element={<PredictionsPage />} />
          <Route path="/user" element={<UserPage />} />
          <Route path="/helper" element={<HelperPage />} />
          <Route path="/registry" element={<RegistryPage />} />
          <Route path="/experiments" element={<ExperimentsPage />} />
          <Route path="/monitoring" element={<MonitoringPage />} />
          <Route path="/performance" element={<PerformancePage />} />
          <Route path="/health" element={<Navigate to="/overview" replace />} />
          <Route path="/system-health" element={<Navigate to="/overview" replace />} />
          <Route path="/chat" element={<ChatPage />} />
        </Route>
      </Routes>
    </LiveAlertProvider>
  )
}
