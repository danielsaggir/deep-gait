import { useEffect, useRef } from "react";
import type { PoseFrame } from "../../types/analysis";
import { nearestPoseFrame, videoContentRect } from "../../utils/skeleton";

type Props = {
  video: HTMLVideoElement | null;
  frames: PoseFrame[];
  edges: Array<[number, number]>;
  enabled: boolean;
};

export function SkeletonOverlay({ video, frames, edges, enabled }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !video || !enabled) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    let raf = 0;

    const draw = () => {
      const w = video.clientWidth;
      const h = video.clientHeight;
      const dpr = window.devicePixelRatio || 1;
      const pw = Math.round(w * dpr);
      const ph = Math.round(h * dpr);

      if (canvas.width !== pw || canvas.height !== ph) {
        canvas.width = pw;
        canvas.height = ph;
      }

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);

      const lineWidth = Math.max(2, w / 180);
      const jointRadius = Math.max(3, w / 150);

      const frame = nearestPoseFrame(frames, video.currentTime);
      if (frame && frame.detected !== false) {
        const { ox, oy, dw, dh } = videoContentRect(video);
        ctx.strokeStyle = "rgba(77, 232, 255, 0.9)";
        ctx.fillStyle = "rgba(77, 232, 255, 0.98)";
        ctx.lineWidth = lineWidth;
        ctx.lineCap = "round";
        ctx.shadowColor = "rgba(77, 232, 255, 0.55)";
        ctx.shadowBlur = Math.max(4, w / 120);

        for (const [a, b] of edges) {
          const ja = frame.joints[a];
          const jb = frame.joints[b];
          if (!ja || !jb) continue;
          if ((ja.confidence ?? 1) <= 0.02 && (jb.confidence ?? 1) <= 0.02) continue;
          ctx.beginPath();
          ctx.moveTo(ox + ja.x * dw, oy + ja.y * dh);
          ctx.lineTo(ox + jb.x * dw, oy + jb.y * dh);
          ctx.stroke();
        }
        for (const joint of frame.joints) {
          if ((joint.confidence ?? 1) <= 0.02) continue;
          ctx.beginPath();
          ctx.arc(ox + joint.x * dw, oy + joint.y * dh, jointRadius, 0, Math.PI * 2);
          ctx.fill();
        }
      }
      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [video, frames, edges, enabled]);

  if (!enabled) return null;
  return <canvas className="skeleton-canvas" ref={canvasRef} />;
}
