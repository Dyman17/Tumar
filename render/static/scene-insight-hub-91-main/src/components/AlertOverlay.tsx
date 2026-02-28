import { AlertTriangle, X } from "lucide-react";
import { useState } from "react";

interface AlertEvent {
  id: number;
  time: string;
  message: string;
  severity: "low" | "medium" | "high";
}

interface AlertOverlayProps {
  events: AlertEvent[];
}

export default function AlertOverlay({ events }: AlertOverlayProps) {
  const [dismissed, setDismissed] = useState<Set<number>>(new Set());

  const visible = events.filter((e) => !dismissed.has(e.id)).slice(-3);
  if (visible.length === 0) return null;

  const severityStyles = {
    low: "border-foreground/10 bg-card/80",
    medium: "border-foreground/20 bg-card/80",
    high: "border-destructive/40 bg-card/80 glow-alert",
  };

  return (
    <div className="fixed bottom-6 right-6 z-50 flex flex-col gap-2 max-w-sm">
      {visible.map((event) => (
        <div
          key={event.id}
          className={`
            rounded-lg border p-3 backdrop-blur-md flex items-start gap-3
            animate-fade-in ${severityStyles[event.severity]}
          `}
        >
          <AlertTriangle className="w-4 h-4 mt-0.5 text-foreground shrink-0" />
          <div className="flex-1 min-w-0">
            <p className="text-[10px] font-mono text-muted-foreground">{event.time}</p>
            <p className="text-xs font-mono text-foreground">{event.message}</p>
          </div>
          <button
            onClick={() => setDismissed((s) => new Set(s).add(event.id))}
            className="text-muted-foreground hover:text-foreground transition-colors"
          >
            <X className="w-3 h-3" />
          </button>
        </div>
      ))}
    </div>
  );
}
