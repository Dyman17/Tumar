import { useState, useEffect, useRef } from "react";
import { Car, User, Crosshair } from "lucide-react";

export interface MapObject {
  id: string;
  type: "vehicle" | "person";
  x: number; // 0-1 normalized
  y: number; // 0-1 normalized
  angle?: number; // degrees
  risk?: number; // 0-1
}

interface RoomMapProps {
  objects: MapObject[];
  width?: number;
  height?: number;
}

const GRID_COLS = 8;
const GRID_ROWS = 6;

export default function RoomMap({ objects, width = 800, height = 500 }: RoomMapProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [hovered, setHovered] = useState<string | null>(null);

  return (
    <div className="flex flex-col gap-2">
      <h2 className="text-xs font-mono uppercase tracking-[0.3em] text-muted-foreground flex items-center gap-2">
        <Crosshair className="w-3.5 h-3.5" />
        Room Map — Live Positions
      </h2>

      <div
        ref={containerRef}
        className="relative w-full rounded-lg border border-border bg-card/40 backdrop-blur-sm overflow-hidden"
        style={{ aspectRatio: `${width}/${height}` }}
      >
        {/* Grid lines */}
        <svg className="absolute inset-0 w-full h-full" preserveAspectRatio="none">
          {/* Vertical grid */}
          {Array.from({ length: GRID_COLS + 1 }).map((_, i) => (
            <line
              key={`v-${i}`}
              x1={`${(i / GRID_COLS) * 100}%`}
              y1="0"
              x2={`${(i / GRID_COLS) * 100}%`}
              y2="100%"
              stroke="hsl(0 0% 100%)"
              strokeOpacity={i === 0 || i === GRID_COLS ? 0.08 : 0.03}
              strokeWidth="1"
            />
          ))}
          {/* Horizontal grid */}
          {Array.from({ length: GRID_ROWS + 1 }).map((_, i) => (
            <line
              key={`h-${i}`}
              x1="0"
              y1={`${(i / GRID_ROWS) * 100}%`}
              x2="100%"
              y2={`${(i / GRID_ROWS) * 100}%`}
              stroke="hsl(0 0% 100%)"
              strokeOpacity={i === 0 || i === GRID_ROWS ? 0.08 : 0.03}
              strokeWidth="1"
            />
          ))}
        </svg>

        {/* Axis labels */}
        <div className="absolute bottom-1 left-0 right-0 flex justify-between px-2">
          {Array.from({ length: GRID_COLS + 1 }).map((_, i) => (
            <span key={i} className="text-[8px] font-mono text-muted-foreground/40">
              {((i / GRID_COLS) * width).toFixed(0)}
            </span>
          ))}
        </div>
        <div className="absolute top-0 bottom-0 left-1 flex flex-col justify-between py-2">
          {Array.from({ length: GRID_ROWS + 1 }).map((_, i) => (
            <span key={i} className="text-[8px] font-mono text-muted-foreground/40">
              {((i / GRID_ROWS) * height).toFixed(0)}
            </span>
          ))}
        </div>

        {/* Objects */}
        {objects.map((obj) => {
          const isHovered = hovered === obj.id;
          const isHighRisk = (obj.risk ?? 0) > 0.7;

          return (
            <div
              key={obj.id}
              className="absolute flex flex-col items-center transition-all duration-500 ease-out"
              style={{
                left: `${obj.x * 100}%`,
                top: `${obj.y * 100}%`,
                transform: "translate(-50%, -50%)",
                zIndex: isHovered ? 20 : 10,
              }}
              onMouseEnter={() => setHovered(obj.id)}
              onMouseLeave={() => setHovered(null)}
            >
              {/* Ping ring */}
              <div
                className={`absolute w-8 h-8 rounded-full border transition-all duration-300 ${
                  isHighRisk
                    ? "border-destructive/40 animate-ping"
                    : "border-foreground/10"
                }`}
              />

              {/* Direction indicator */}
              {obj.angle !== undefined && (
                <div
                  className="absolute w-10 h-10"
                  style={{ transform: `rotate(${obj.angle}deg)` }}
                >
                  <div className="absolute top-0 left-1/2 -translate-x-1/2 w-0.5 h-3 bg-gradient-to-t from-transparent to-foreground/30 rounded-full" />
                </div>
              )}

              {/* Icon */}
              <div
                className={`
                  relative w-6 h-6 rounded-full flex items-center justify-center
                  transition-all duration-300
                  ${isHighRisk ? "bg-destructive/20 glow-alert" : "bg-foreground/10 glow-white"}
                  ${isHovered ? "scale-125" : ""}
                `}
              >
                {obj.type === "vehicle" ? (
                  <Car className={`w-3.5 h-3.5 ${isHighRisk ? "text-destructive" : "text-foreground"}`} />
                ) : (
                  <User className={`w-3.5 h-3.5 ${isHighRisk ? "text-destructive" : "text-foreground"}`} />
                )}
              </div>

              {/* Tooltip */}
              {isHovered && (
                <div className="absolute -top-10 bg-card/90 border border-border backdrop-blur-md rounded px-2 py-1 whitespace-nowrap animate-fade-in">
                  <span className="text-[10px] font-mono text-foreground">
                    {obj.type} #{obj.id} — ({(obj.x * width).toFixed(0)}, {(obj.y * height).toFixed(0)})
                    {obj.risk !== undefined && ` • risk ${obj.risk.toFixed(2)}`}
                  </span>
                </div>
              )}
            </div>
          );
        })}

        {/* Empty state */}
        {objects.length === 0 && (
          <div className="absolute inset-0 flex items-center justify-center">
            <p className="text-xs font-mono text-muted-foreground/50 tracking-widest uppercase">
              No objects detected
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
