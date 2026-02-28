import { SafeQunarAPI, useSafeQunarAPI } from "@/hooks/useSafeQunarAPI";
import ArrowsBackground from "@/components/ArrowsBackground";
import VideoStream from "@/components/VideoStream";
import RiskGauge from "@/components/RiskGauge";
import SceneStats from "@/components/SceneStats";
import AlertOverlay from "@/components/AlertOverlay";
import RoomMap, { type MapObject } from "@/components/RoomMap";
import { Orbit, Settings, Activity, Users, Car, Thermometer } from "lucide-react";

export default function SafeQunarIndex() {
  const { data, objects, alerts, connected, analytics, isLive } = useSafeQunarAPI();

  return (
    <div className="relative min-h-screen">
      <ArrowsBackground />

      <main className="relative z-10 max-w-6xl mx-auto px-4 py-8 flex flex-col gap-6">
        {/* Header */}
        <header className="flex items-center gap-3">
          <Orbit className="w-6 h-6 text-foreground" />
          <h1 className="text-lg font-mono font-bold tracking-wider gradient-cosmic-text uppercase">
            Tumar AI Monitor
          </h1>
          <div className="ml-auto flex items-center gap-4">
            <span className="flex items-center gap-1.5">
              <span className={`w-1.5 h-1.5 rounded-full ${connected ? "bg-foreground pulse-glow" : "bg-destructive"}`} />
              <span className="text-[10px] font-mono text-muted-foreground tracking-widest uppercase">
                {connected ? "Connected" : "Disconnected"}
              </span>
            </span>
            {analytics && (
              <span className="text-[10px] font-mono text-muted-foreground">
                FPS: {data.fps?.toFixed(1) || "0.0"}
              </span>
            )}
          </div>
        </header>

        {/* Video + Risk row */}
        <div className="grid grid-cols-1 lg:grid-cols-[2fr_1fr] gap-6 items-start">
          <VideoStream src={data.video_frame ? `data:image/jpeg;base64,${data.video_frame}` : undefined} />
          <div className="flex flex-col gap-6">
            <RiskGauge score={data.riskScore} />
            
            {/* Quick Stats */}
            <div className="bg-background/80 backdrop-blur-sm border border-border rounded-lg p-4">
              <h3 className="text-sm font-medium mb-3 flex items-center gap-2">
                <Activity className="w-4 h-4" />
                Live Stats
              </h3>
              <div className="grid grid-cols-2 gap-3 text-xs">
                <div className="flex items-center gap-2">
                  <Car className="w-3 h-3 text-blue-500" />
                  <span>{data.vehicles} vehicles</span>
                </div>
                <div className="flex items-center gap-2">
                  <Users className="w-3 h-3 text-green-500" />
                  <span>{data.persons} persons</span>
                </div>
                <div className="flex items-center gap-2">
                  <Thermometer className="w-3 h-3 text-orange-500" />
                  <span>{data.thermalFlag ? "Active" : "Normal"}</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="w-3 h-3 rounded-full bg-purple-500"></span>
                  <span>{data.avgSpeed?.toFixed(1) || "0.0"} m/s</span>
                </div>
              </div>
            </div>
          </div>
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

        {/* Analytics Panel */}
        {analytics && (
          <div className="bg-background/80 backdrop-blur-sm border border-border rounded-lg p-6">
            <h3 className="text-lg font-medium mb-4">System Analytics</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-500">
                  {Math.floor(analytics.uptime / 3600)}h {Math.floor((analytics.uptime % 3600) / 60)}m
                </div>
                <div className="text-xs text-muted-foreground">Uptime</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-green-500">
                  {analytics.total_frames}
                </div>
                <div className="text-xs text-muted-foreground">Total Frames</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-red-500">
                  {analytics.alerts_count}
                </div>
                <div className="text-xs text-muted-foreground">Alerts</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-purple-500">
                  {analytics.risk_stats?.average?.toFixed(3) || "0.000"}
                </div>
                <div className="text-xs text-muted-foreground">Avg Risk</div>
              </div>
            </div>
          </div>
        )}
      </main>

      <AlertOverlay events={alerts} />
    </div>
  );
}
