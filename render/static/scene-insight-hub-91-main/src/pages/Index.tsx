import { useState, useEffect } from "react";
import ArrowsBackground from "@/components/ArrowsBackground";
import VideoStream from "@/components/VideoStream";
import RiskGauge from "@/components/RiskGauge";
import SceneStats from "@/components/SceneStats";
import AlertOverlay from "@/components/AlertOverlay";
import RoomMap, { type MapObject } from "@/components/RoomMap";
import { Orbit, Settings } from "lucide-react";

// Config: set your FastAPI base URL here
const API_BASE = ""; // e.g. "http://192.168.1.100:8000"
const POLL_INTERVAL = 1000;

function generateMockObjects(): MapObject[] {
  const count = 2 + Math.floor(Math.random() * 4);
  return Array.from({ length: count }, (_, i) => ({
    id: String(i + 1),
    type: i < Math.ceil(count * 0.6) ? "vehicle" as const : "person" as const,
    x: 0.1 + Math.random() * 0.8,
    y: 0.1 + Math.random() * 0.8,
    angle: Math.random() * 360,
    risk: Math.random() * 0.9,
  }));
}

function useSceneData() {
  const [data, setData] = useState({
    riskScore: 0.32,
    vehicles: 2,
    persons: 0,
    convoyFlag: false,
    thermalFlag: false,
  });
  const [objects, setObjects] = useState<MapObject[]>(generateMockObjects);
  const [alerts, setAlerts] = useState<
    { id: number; time: string; message: string; severity: "low" | "medium" | "high" }[]
  >([]);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    let id = 0;
    let cancelled = false;

    const fetchData = async () => {
      if (!API_BASE) {
        // Mock mode
        const risk = Math.min(1, Math.max(0, 0.3 + Math.random() * 0.55));
        const vehicles = Math.floor(Math.random() * 6);
        const persons = Math.floor(Math.random() * 3);
        const convoy = vehicles >= 4;

        setData({ riskScore: risk, vehicles, persons, convoyFlag: convoy, thermalFlag: Math.random() > 0.85 });
        setObjects(generateMockObjects());

        if (risk > 0.7) {
          id++;
          const now = new Date();
          setAlerts((prev) => [
            ...prev.slice(-5),
            {
              id,
              time: now.toLocaleTimeString(),
              message: convoy
                ? `Convoy detected: ${vehicles} vehicles`
                : `High risk: ${risk.toFixed(2)} — ${persons} person(s)`,
              severity: risk > 0.85 ? "high" : "medium",
            },
          ]);
        }
        return;
      }

      // Real FastAPI mode
      try {
        const res = await fetch(`${API_BASE}/risk`);
        if (!res.ok) throw new Error("API error");
        const json = await res.json();

        setConnected(true);
        setData({
          riskScore: json.risk_score ?? 0,
          vehicles: json.vehicles ?? 0,
          persons: json.persons ?? 0,
          convoyFlag: json.convoy_flag ?? false,
          thermalFlag: json.thermal_flag ?? false,
        });

        if (json.objects) {
          setObjects(
            json.objects.map((o: any) => ({
              id: String(o.id),
              type: o.type === "person" ? "person" : "vehicle",
              x: o.x,
              y: o.y,
              angle: o.angle,
              risk: o.risk,
            }))
          );
        }

        if ((json.risk_score ?? 0) > 0.7) {
          id++;
          const now = new Date();
          setAlerts((prev) => [
            ...prev.slice(-5),
            {
              id,
              time: now.toLocaleTimeString(),
              message: `Risk ${json.risk_score.toFixed(2)} — ${json.vehicles} veh, ${json.persons} pers`,
              severity: json.risk_score > 0.85 ? "high" : "medium",
            },
          ]);
        }
      } catch {
        setConnected(false);
      }
    };

    const interval = setInterval(() => {
      if (!cancelled) fetchData();
    }, POLL_INTERVAL);
    fetchData();

    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  return { data, objects, alerts, connected, isLive: !!API_BASE };
}

export default function Index() {
  const { data, objects, alerts, connected, isLive } = useSceneData();

  return (
    <div className="relative min-h-screen">
      <ArrowsBackground />

      <main className="relative z-10 max-w-4xl mx-auto px-4 py-8 flex flex-col gap-6">
        {/* Header */}
        <header className="flex items-center gap-3">
          <Orbit className="w-6 h-6 text-foreground" />
          <h1 className="text-lg font-mono font-bold tracking-wider gradient-cosmic-text uppercase">
            Sentinel Dashboard
          </h1>
          <div className="ml-auto flex items-center gap-3">
            <span className="flex items-center gap-1.5">
              <span className={`w-1.5 h-1.5 rounded-full ${isLive && connected ? "bg-foreground pulse-glow" : isLive ? "bg-destructive" : "bg-muted-foreground"}`} />
              <span className="text-[10px] font-mono text-muted-foreground tracking-widest uppercase">
                {isLive ? (connected ? "Connected" : "Disconnected") : "Mock Data"}
              </span>
            </span>
          </div>
        </header>

        {/* Video + Risk row */}
        <div className="grid grid-cols-1 md:grid-cols-[1fr_200px] gap-6 items-start">
          <VideoStream src={isLive ? `${API_BASE}/video` : undefined} />
          <RiskGauge score={data.riskScore} />
        </div>

        {/* Room Map */}
        <RoomMap objects={objects} />

        {/* Scene Stats */}
        <SceneStats
          vehicles={data.vehicles}
          persons={data.persons}
          convoyFlag={data.convoyFlag}
          thermalFlag={data.thermalFlag}
        />
      </main>

      <AlertOverlay events={alerts} />
    </div>
  );
}
