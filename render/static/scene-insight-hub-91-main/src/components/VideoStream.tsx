import { Video } from "lucide-react";

interface VideoStreamProps {
  src?: string;
}

export default function VideoStream({ src }: VideoStreamProps) {
  const isLive = !!src;

  return (
    <div className="relative w-full aspect-video rounded-lg overflow-hidden border border-border bg-card/60 backdrop-blur-sm scanlines">
      {isLive ? (
        <img
          src={src}
          alt="Live camera feed"
          className="w-full h-full object-cover"
          style={{ filter: "brightness(0.9)" }}
        />
      ) : (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-muted/30">
          <Video className="w-10 h-10 text-muted-foreground animate-float" />
          <p className="text-xs font-mono text-muted-foreground tracking-widest uppercase">
            No Video Feed
          </p>
          <p className="text-[10px] font-mono text-muted-foreground/60">
            Connect ESP32-CAM MJPEG stream
          </p>
        </div>
      )}
      {/* Live indicator */}
      <div className="absolute top-3 left-3 flex items-center gap-2">
        <span className={`w-2 h-2 rounded-full ${isLive ? "bg-destructive pulse-glow" : "bg-muted-foreground"}`} />
        <span className="text-[10px] font-mono uppercase tracking-widest text-foreground/70">
          {isLive ? "Live" : "Offline"}
        </span>
      </div>
    </div>
  );
}
