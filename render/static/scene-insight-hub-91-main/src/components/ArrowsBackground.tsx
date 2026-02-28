import { useEffect, useRef, useState, useCallback } from "react";

interface Arrow {
  x: number;
  y: number;
  size: number;
  baseAngle: number;
  delay: number;
}

const ARROW_COUNT = 18;

function generateArrows(): Arrow[] {
  return Array.from({ length: ARROW_COUNT }, () => ({
    x: Math.random() * 100,
    y: Math.random() * 100,
    size: 20 + Math.random() * 24,
    baseAngle: Math.random() * 360,
    delay: Math.random() * 2,
  }));
}

export default function ArrowsBackground() {
  const [arrows] = useState(generateArrows);
  const [mouse, setMouse] = useState<{ x: number; y: number } | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const isMobile = typeof window !== "undefined" && window.innerWidth < 768;

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (containerRef.current) {
      const rect = containerRef.current.getBoundingClientRect();
      setMouse({
        x: ((e.clientX - rect.left) / rect.width) * 100,
        y: ((e.clientY - rect.top) / rect.height) * 100,
      });
    }
  }, []);

  useEffect(() => {
    if (isMobile) return;
    window.addEventListener("mousemove", handleMouseMove);
    return () => window.removeEventListener("mousemove", handleMouseMove);
  }, [handleMouseMove, isMobile]);

  const getAngle = (arrow: Arrow) => {
    if (isMobile) {
      const cx = 50, cy = 50;
      return Math.atan2(cy - arrow.y, cx - arrow.x) * (180 / Math.PI);
    }
    if (!mouse) return arrow.baseAngle;
    return Math.atan2(mouse.y - arrow.y, mouse.x - arrow.x) * (180 / Math.PI);
  };

  return (
    <div ref={containerRef} className="fixed inset-0 overflow-hidden pointer-events-none z-0 stars">
      {arrows.map((arrow, i) => {
        const angle = getAngle(arrow);
        return (
          <svg
            key={i}
            className="absolute"
            style={{
              left: `${arrow.x}%`,
              top: `${arrow.y}%`,
              width: arrow.size,
              height: arrow.size,
              transform: `translate(-50%, -50%) rotate(${angle}deg)`,
              transition: "transform 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94)",
              opacity: 0.08 + (i % 3) * 0.04,
            }}
            viewBox="0 0 24 24"
            fill="none"
          >
            <path
              d="M5 12h14M13 5l6 7-6 7"
              stroke="hsl(0 0% 100%)"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        );
      })}
    </div>
  );
}
