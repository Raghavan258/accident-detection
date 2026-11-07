import { renderAt } from "@/test/render"
import { health, status } from "@/test/fixtures"
import { screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { beforeEach, describe, expect, it, vi } from "vitest"

beforeEach(() => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo, init?: RequestInit) => {
      const url = String(input)
      if (url.endsWith("/mlops/status")) return json(status)
      if (url.endsWith("/mlops/health")) return json(health)
      if (url.endsWith("/mlops/schema")) {
        return json({ features: [{ name: "peak_horizontal_g", label: "Peak horizontal acceleration", unit: "g", group: "acceleration" }] })
      }
      if (url.endsWith("/mlops/examples")) return json({ rows: [] })
      if (url.endsWith("/predict") && init?.method === "POST") return json({ prediction: "non_crash", crash: 0, probability: 0.0134, threshold: 0.61524, model_name: "iveras-crash-detector", model_version: "1", feature_names: ["peak_horizontal_g"], latency_ms: 1.6 })
      return json({ detail: "missing" }, 422)
    }),
  )
})

describe("mlops frontend", () => {
  it("separates validation metrics from test metrics", async () => {
    renderAt("/overview")
    expect(await screen.findByText("Validation precision")).toBeInTheDocument()
    expect(screen.getByText("0.8169").closest("section")?.textContent).toContain("Validation precision")
    expect(screen.getByText("Total predictions").closest("section")?.textContent).toContain("N/A")
    expect(screen.getByText(/not a deployment input/i)).toBeInTheDocument()
  })

  it("shows the registry gate and blocks an unvalidated alias in the form", async () => {
    renderAt("/registry")
    expect(await screen.findByText("iveras-crash-detector")).toBeInTheDocument()
    expect(screen.getByText("not passed")).toBeInTheDocument()
    const buttons = screen.getAllByRole("button", { name: "Set production" })
    expect(buttons[1]).toBeDisabled()
  })

  it("submits a feature prediction and surfaces API errors", async () => {
    const user = userEvent.setup()
    renderAt("/predictions")
    await user.click(await screen.findByRole("button", { name: "Predict" }))
    expect(await screen.findByText(/must be a finite number/i)).toBeInTheDocument()
    await user.type(screen.getByLabelText("peak_horizontal_g"), "0.2")
    await user.click(screen.getByRole("button", { name: "Predict" }))
    expect(await screen.findByText("non_crash")).toBeInTheDocument()
    expect(screen.getByText("0.0134")).toBeInTheDocument()
  })

  it("shows a failed prediction response", async () => {
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo) => {
      const url = String(input)
      if (url.endsWith("/mlops/status")) return json(status)
      if (url.endsWith("/mlops/health")) return json(health)
      if (url.endsWith("/mlops/schema")) return json({ features: [{ name: "peak_horizontal_g", label: "Peak", unit: "g", group: "acceleration" }] })
      if (url.endsWith("/mlops/examples")) return json({ rows: [] })
      return json({ detail: "no production model is registered" }, 409)
    }))
    const user = userEvent.setup()
    renderAt("/predictions")
    await user.type(await screen.findByLabelText("peak_horizontal_g"), "1")
    await user.click(screen.getByRole("button", { name: "Predict" }))
    await waitFor(() => expect(screen.getByText(/no production model is registered/i)).toBeInTheDocument())
  })
})

function json(body: unknown, statusCode = 200) {
  return Promise.resolve({ ok: statusCode < 400, status: statusCode, statusText: "error", json: async () => body })
}
