import type { PoseFrame } from "../types/analysis";

export function nearestPoseFrame(frames: PoseFrame[], time: number): PoseFrame | null {
  if (!frames.length) return null;
  let best = frames[0];
  let bestDist = Math.abs(best.timestamp - time);
  for (let i = 1; i < frames.length; i += 1) {
    const dist = Math.abs(frames[i].timestamp - time);
    if (dist < bestDist) {
      best = frames[i];
      bestDist = dist;
    }
  }
  return best;
}

export function videoContentRect(video: HTMLVideoElement): {
  ox: number;
  oy: number;
  dw: number;
  dh: number;
} {
  const rw = video.videoWidth || 1;
  const rh = video.videoHeight || 1;
  const cw = video.clientWidth;
  const ch = video.clientHeight;
  const scale = Math.min(cw / rw, ch / rh);
  const dw = rw * scale;
  const dh = rh * scale;
  return { ox: (cw - dw) / 2, oy: (ch - dh) / 2, dw, dh };
}
