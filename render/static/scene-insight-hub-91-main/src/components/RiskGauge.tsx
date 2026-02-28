interface RiskGaugeProps {
  score: number;
}

function getRiskLevel(score: number) {
  if (score < 0.4) return { label: "LOW", colorClass: "text-foreground text-glow", glowClass: "glow-white", ringColor: "stroke-foreground" };
  if (score < 0.7) return { label: "MEDIUM", colorClass: "text-foreground text-glow", glowClass: "glow-white", ringColor: "stroke-muted-foreground" };
  return { label: "ALERT", colorClass: "text-destructive text-glow-alert pulse-glow", glowClass: "glow-alert", ringColor: "stroke-destructive" };
}

export default function RiskGauge({ score }: RiskGaugeProps) {
  const { label, colorClass, glowClass, ringColor } = getRiskLevel(score);
  const circumference = 2 * Math.PI * 54;
  const offset = circumference - score * circumference;

  return (
    <div className="flex flex-col items-center gap-3">
      <h2 className="text-xs font-mono uppercase tracking-[0.3em] text-muted-foreground">Risk Score</h2>
      <div className={`relative w-36 h-36 rounded-full ${glowClass}`}>
        <svg className="w-full h-full -rotate-90" viewBox="0 0 120 120">
          <circle cx="60" cy="60" r="54" fill="none" className="stroke-muted" strokeWidth="4" />
          <circle
            cx="60" cy="60" r="54"
            fill="none"
            className={`${ringColor} transition-all duration-700`}
            strokeWidth="4"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className={`text-3xl font-mono font-bold ${colorClass}`}>
            {score.toFixed(2)}
          </span>
        </div>
      </div>
      <span className={`text-sm font-mono font-semibold tracking-widest ${colorClass}`}>
        {label}
      </span>
    </div>
  );
}
