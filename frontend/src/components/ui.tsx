import * as Dialog from "@radix-ui/react-dialog"
import * as Tooltip from "@radix-ui/react-tooltip"
import { cn } from "@/lib/utils"
import type { ButtonHTMLAttributes, InputHTMLAttributes, ReactNode } from "react"

export function Button({ className, ...props }: ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      className={cn(
        "inline-flex items-center justify-center rounded-md border border-[#2a3958] bg-[#162033] px-3 py-1.5 text-sm text-[#e7ecf3] hover:bg-[#1c2942] disabled:cursor-not-allowed disabled:opacity-40",
        className,
      )}
      {...props}
    />
  )
}

export function Card({ title, children, action }: { title?: string; children: ReactNode; action?: ReactNode }) {
  return (
    <section className="rounded-lg border border-[#243049] bg-[#121a2b] p-4">
      {title ? (
        <div className="mb-3 flex items-center justify-between gap-3">
          <h2 className="text-sm font-medium text-[#d5deea]">{title}</h2>
          {action}
        </div>
      ) : null}
      {children}
    </section>
  )
}

export function Badge({ tone, children }: { tone: "ok" | "warn" | "bad" | "info" | "idle"; children: ReactNode }) {
  const colors = {
    ok: "border-emerald-700/60 text-emerald-300",
    warn: "border-amber-700/60 text-amber-300",
    bad: "border-rose-800/70 text-rose-300",
    info: "border-sky-800/70 text-sky-300",
    idle: "border-[#2a3958] text-[#9aa8bd]",
  }
  return <span className={cn("rounded border px-1.5 py-0.5 text-xs", colors[tone])}>{children}</span>
}

export function TextInput(props: InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      {...props}
      className={cn(
        "w-full rounded-md border border-[#2a3958] bg-[#0d1524] px-2 py-1.5 text-sm text-[#e7ecf3] outline-none focus:border-sky-700",
        props.className,
      )}
    />
  )
}

export function Hint({ text, children }: { text: string; children: ReactNode }) {
  return (
    <Tooltip.Root>
      <Tooltip.Trigger asChild>{children}</Tooltip.Trigger>
      <Tooltip.Portal>
        <Tooltip.Content className="z-50 max-w-xs rounded border border-[#2a3958] bg-[#0d1524] px-2 py-1 text-xs text-[#d5deea]">
          {text}
        </Tooltip.Content>
      </Tooltip.Portal>
    </Tooltip.Root>
  )
}

export function ConfirmDialog({
  open,
  title,
  body,
  confirmLabel,
  onCancel,
  onConfirm,
}: {
  open: boolean
  title: string
  body: string
  confirmLabel: string
  onCancel: () => void
  onConfirm: () => void
}) {
  return (
    <Dialog.Root open={open} onOpenChange={(next) => { if (!next) onCancel() }}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 bg-black/60" />
        <Dialog.Content className="fixed left-1/2 top-1/3 w-[28rem] -translate-x-1/2 rounded-lg border border-[#2a3958] bg-[#121a2b] p-4">
          <Dialog.Title className="text-base">{title}</Dialog.Title>
          <Dialog.Description className="mt-2 text-sm text-[#9aa8bd]">{body}</Dialog.Description>
          <div className="mt-4 flex justify-end gap-2">
            <Button onClick={onCancel}>Cancel</Button>
            <Button onClick={onConfirm}>{confirmLabel}</Button>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  )
}

export function Skeleton({ className }: { className?: string }) {
  return <div className={cn("animate-pulse rounded bg-[#1c2942]", className)} />
}

export function Empty({ children }: { children: ReactNode }) {
  return <p className="text-sm text-[#9aa8bd]">{children}</p>
}
