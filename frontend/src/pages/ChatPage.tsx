import { askSynthetic, type ChatReply } from "@/services/api"
import { Badge, Button, Card, Empty } from "@/components/ui"
import { useState } from "react"

const prompts = ["crashes in Koramangala", "non-crash in Whitefield", "accident density", "ambulance and police"]

type Turn = { role: "you" | "helper"; text: string; body?: ChatReply }

export function ChatPage() {
  const [draft, setDraft] = useState("")
  const [pending, setPending] = useState(false)
  const [turns, setTurns] = useState<Turn[]>([{ role: "helper", text: opening() }])

  async function send(text: string) {
    const message = text.trim()
    if (!message || pending) return
    setTurns((current) => [...current, { role: "you", text: message }])
    setDraft("")
    setPending(true)
    try {
      const body = await askSynthetic(message)
      setTurns((current) => [...current, { role: "helper", text: body.reply, body }])
    } catch (error) {
      const detail = error instanceof Error ? error.message : "The helper did not respond."
      setTurns((current) => [...current, { role: "helper", text: detail }])
    } finally {
      setPending(false)
    }
  }

  return (
    <div className="grid gap-4 lg:grid-cols-[1.1fr_0.9fr]">
      <Card title="Helper chat">
        <p className="mb-3 text-xs text-[#9aa8bd]">Filters the synthetic Bengaluru sample. It does not call a live dispatch service.</p>
        <div className="mb-3 flex flex-wrap gap-2">
          {prompts.map((prompt) => (
            <Button key={prompt} onClick={() => void send(prompt)}>
              {prompt}
            </Button>
          ))}
        </div>
        <div className="grid max-h-[28rem] gap-2 overflow-auto">
          {turns.map((turn, index) => (
            <p key={`${turn.role}-${index}`} className="text-sm">
              <span className="text-[#9aa8bd]">{turn.role === "you" ? "You" : "Helper"}. </span>
              {turn.text}
            </p>
          ))}
        </div>
        <form
          className="mt-3 flex gap-2"
          onSubmit={(event) => {
            event.preventDefault()
            void send(draft)
          }}
        >
          <input
            className="min-w-0 flex-1 rounded-md border border-[#2a3958] bg-[#0d1524] px-2 py-1.5 text-sm"
            value={draft}
            aria-label="Message"
            onChange={(event) => setDraft(event.target.value)}
          />
          <Button type="submit" disabled={pending}>
            Send
          </Button>
        </form>
      </Card>
      <Result turn={[...turns].reverse().find((turn) => turn.body)} />
    </div>
  )
}

function opening(): string {
  return "Ask for crashes or non-crashes in a locality, accident density, or police and ambulance numbers. The pins are synthetic."
}

function Result({ turn }: { turn: Turn | undefined }) {
  if (!turn?.body) return <Empty>Ask a question to list events, density, and the public emergency numbers.</Empty>
  const body = turn.body
  return (
    <div className="grid gap-4">
      <Card title="Public emergency numbers">
        {body.emergency.map((item) => (
          <p key={item.service} className="text-sm">
            {item.service}: {item.number}. <span className="text-[#9aa8bd]">{item.note}</span>
          </p>
        ))}
      </Card>
      <Card title="Crash density">
        {body.density.map((item) => (
          <p key={item.locality} className="text-sm">
            {item.locality}: {item.crashes} crashes, {item.crashes_per_km2} per km²
          </p>
        ))}
      </Card>
      <Card title={`Matching events (${body.matched_rows})`}>
        {body.rows.length === 0 ? <Empty>No synthetic events matched.</Empty> : null}
        {body.rows.map((row) => (
          <p key={row.event_id} className="text-sm">
            <Badge tone={row.label_name === "crash" ? "bad" : "ok"}>{row.label_name}</Badge> {row.locality}{" "}
            {row.latitude.toFixed(4)}, {row.longitude.toFixed(4)} at {row.observed_at}
          </p>
        ))}
      </Card>
    </div>
  )
}
