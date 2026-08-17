import { useEffect, useRef } from "react";
import type { PoseFrame } from "../../types/analysis";
import { readTheme, rgba } from "../../utils/theme";
import { nearestPoseFrame, videoContentRect } from "../../utils/skeleton";

type Props = {
  video: HTMLVideoElement | null;
  frames: PoseFrame[];
  edges: Array<[number, number]>;
  enabled: boolean;
  /** Which subject this overlay belongs to, so it matches the charts. */
  tone: "a" | "b";
};

export function SkeletonOverlay({ video, frames, edges, enabled, tone }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !video || !enabled) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    let raf = 0;
    let lastFrame: PoseFrame | null | undefined;
    let lastW = -1;
    let lastH = -1;

    const draw = () => {
      raf = requestAnimationFrame(draw);
      if (document.hidden) return;

      const w = video.clientWidth;
      const h = video.clientHeight;
      const frame = nearestPoseFrame(frames, video.currentTime);

      // A paused clip sits on the same pose indefinitely. Repainting identical
      // content every frame is the single most expensive thing this component
      // can do, and it used to keep doing it for as long as a result was open.
      if (frame === lastFrame && w === lastW && h === lastH) return;
      lastFrame = frame;
      lastW = w;
      lastH = h;

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

      if (frame && frame.detected !== false) {
        const { ox, oy, dw, dh } = videoContentRect(video);
        const theme = readTheme();
        const subject = tone === "a" ? theme.subjectA : theme.subjectB;
        ctx.lineCap = "round";

        const strokeBones = () => {
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
        };

        // A wide translucent pass reads as a glow for a fraction of the cost of
        // shadowBlur, which would otherwise rasterise a blur per bone per frame.
        ctx.strokeStyle = rgba(subject, 0.2);
        ctx.lineWidth = lineWidth * 3.2;
        strokeBones();

        ctx.strokeStyle = rgba(subject, 0.92);
        ctx.lineWidth = lineWidth;
        strokeBones();

        ctx.fillStyle = rgba(theme.star, 0.95);
        for (const joint of frame.joints) {
          if ((joint.confidence ?? 1) <= 0.02) continue;
          ctx.beginPath();
          ctx.arc(ox + joint.x * dw, oy + joint.y * dh, jointRadius, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [video, frames, edges, enabled, tone]);

  if (!enabled) return null;
  return <canvas className="skeleton-canvas" ref={canvasRef} />;
}
