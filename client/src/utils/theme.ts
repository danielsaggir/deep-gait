/** Read design tokens from :root so canvas layers stay in sync with the CSS palette. */
export type ThemeColors = {
  accent: string;
  accentBright: string;
  indigo: string;
  star: string;
  match: string;
  bg0: string;
  subjectA: string;
  subjectB: string;
};

let cached: ThemeColors | null = null;

/**
 * getComputedStyle forces a style recalc, so animation loops must never call
 * this per frame. The palette is static at runtime; read it once and reuse.
 */
export function readTheme(): ThemeColors {
  if (cached) return cached;
  const root = getComputedStyle(document.documentElement);
  const v = (name: string, fallback: string) => root.getPropertyValue(name).trim() || fallback;
  cached = {
    accent: v("--accent", "#4d9bff"),
    accentBright: v("--accent-bright", "#8ec5ff"),
    indigo: v("--indigo", "#4338ca"),
    star: v("--star", "#cfe3ff"),
    match: v("--match", "#3de8c0"),
    bg0: v("--bg-0", "#04060d"),
    subjectA: v("--subject-a", "#38bdf8"),
    subjectB: v("--subject-b", "#c084fc"),
  };
  return cached;
}

function hexRgb(hex: string): [number, number, number] {
  const h = hex.replace("#", "");
  const full = h.length === 3 ? h.split("").map((c) => c + c).join("") : h;
  const n = Number.parseInt(full, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

export function rgba(hex: string, alpha: number): string {
  if (!hex.startsWith("#")) return `rgba(77, 155, 255, ${alpha})`;
  const [r, g, b] = hexRgb(hex);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}
