import { Car, User, Truck, Thermometer } from "lucide-react";

interface SceneStatsProps {
  vehicles: number;
  persons: number;
  convoyFlag: boolean;
  thermalFlag: boolean;
}

export default function SceneStats({ vehicles, persons, convoyFlag, thermalFlag }: SceneStatsProps) {
  const stats = [
    { icon: Car, label: "Vehicles", value: vehicles, active: vehicles > 0 },
    { icon: User, label: "Persons", value: persons, active: persons > 0 },
    { icon: Truck, label: "Convoy", value: convoyFlag ? "YES" : "NO", active: convoyFlag },
    { icon: Thermometer, label: "Thermal", value: thermalFlag ? "YES" : "NO", active: thermalFlag },
  ];

  return (
    <div className="grid grid-cols-2 gap-3">
      {stats.map(({ icon: Icon, label, value, active }) => (
        <div
          key={label}
          className={`
            rounded-lg border bg-card/60 backdrop-blur-sm p-4 flex items-center gap-3 transition-all duration-300
            ${active ? "border-foreground/20 glow-white" : "border-border"}
          `}
        >
          <Icon
            className={`w-5 h-5 ${active ? "text-foreground" : "text-muted-foreground"}`}
          />
          <div>
            <p className="text-[10px] font-mono uppercase tracking-widest text-muted-foreground">{label}</p>
            <p className={`text-lg font-mono font-bold ${active ? "text-foreground" : "text-muted-foreground"}`}>
              {value}
            </p>
          </div>
        </div>
      ))}
    </div>
  );
}
