import { useEffect, useRef } from "react";
import { readTheme, rgba } from "../../utils/theme";

const GRID = 48;
const FRAME_MS = 1000 / 30;
const TRAIL = 150;
const TRAIL_STEPS = 12;
const NODE_FADE = 0.045;

type Axis = "h" | "v";
type Beam = { axis: Axis; lane: number; pos: number; dir: 1 | -1; speed: number; tint: number };
type Lit = { x: number; y: number; life: number; tint: number };

type Props = { charged?: boolean };

/**
 * A beam grid. Precise dots mark every intersection, and a small number of
 * light beams travel the lanes between them, igniting each node they cross.
 *
 * The restraint is deliberate: a background carries more visual weight than
 * anything else on the page, so it earns its keep by suggesting structure and
 * then getting out of the way of the content sitting on top of it.
 */
export function AmbientCanvas({ charged = false }: Props) {
  const boardRef = useRef<HTMLCanvasElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const chargedRef = useRef(charged);
  chargedRef.current = charged;

  useEffect(() => {
    const canvas = canvasRef.current;
    const board = boardRef.current;
    if (!canvas || !board) return;
    const ctx = canvas.getContext("2d", { alpha: true });
    const bctx = board.getContext("2d", { alpha: true });
    if (!ctx || !bctx) return;

    const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const theme = readTheme();

    const dotColor = rgba(theme.accent, 0.38);
    const dotMajor = rgba(theme.accentBright, 0.62);
    const lineColor = rgba(theme.accent, 0.1);

    // Three tints keep the field from reading as a single flat colour.
    const tints = [theme.accentBright, theme.accent, theme.match];
    const trailRamp = tints.map((tint) =>
      Array.from({ length: TRAIL_STEPS }, (_, i) =>
        rgba(tint, Math.pow(1 - i / TRAIL_STEPS, 2) * 0.8)
      )
    );
    const headColor = tints.map((tint) => rgba(tint, 1));
    const haloColor = tints.map((tint) => rgba(tint, 0.3));
    const nodeRamp = tints.map((tint) => Array.from({ length: 10 }, (_, i) => rgba(tint, i / 9)));

    let raf = 0;
    let w = 0;
    let h = 0;
    let cols = 0;
    let rows = 0;
    let beams: Beam[] = [];
    let lit: Lit[] = [];

    const paintBoard = () => {
      bctx.clearRect(0, 0, w, h);

      bctx.lineWidth = 1;
      bctx.strokeStyle = lineColor;
      for (let c = 0; c <= cols; c += 4) {
        bctx.beginPath();
        bctx.moveTo(c * GRID + 0.5, 0);
        bctx.lineTo(c * GRID + 0.5, h);
        bctx.stroke();
      }
      for (let r = 0; r <= rows; r += 4) {
        bctx.beginPath();
        bctx.moveTo(0, r * GRID + 0.5);
        bctx.lineTo(w, r * GRID + 0.5);
        bctx.stroke();
      }

      for (let c = 0; c <= cols; c++) {
        for (let r = 0; r <= rows; r++) {
          const major = c % 4 === 0 && r % 4 === 0;
          bctx.fillStyle = major ? dotMajor : dotColor;
          const s = major ? 3 : 1.8;
          bctx.fillRect(c * GRID - s / 2, r * GRID - s / 2, s, s);
        }
      }
    };

    const resize = () => {
      // The background is soft by design, so it does not need retina density.
      const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
      w = window.innerWidth;
      h = window.innerHeight;
      cols = Math.ceil(w / GRID);
      rows = Math.ceil(h / GRID);

      for (const c of [canvas, board]) {
        c.width = Math.floor(w * dpr);
        c.height = Math.floor(h * dpr);
        c.style.width = `${w}px`;
        c.style.height = `${h}px`;
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      bctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      const count = w < 900 ? 10 : 22;
      beams = Array.from({ length: count }, () => spawn());
      lit = [];
      paintBoard();
    };

    const spawn = (): Beam => {
      const axis: Axis = Math.random() < 0.5 ? "h" : "v";
      const dir = Math.random() < 0.5 ? 1 : -1;
      const span = axis === "h" ? w : h;
      return {
        axis,
        lane: Math.floor(Math.random() * (axis === "h" ? rows : cols) + 0.5),
        pos: dir === 1 ? -TRAIL - Math.random() * span : span + TRAIL + Math.random() * span,
        dir,
        speed: 1.6 + Math.random() * 2.8,
        tint: Math.random() < 0.14 ? 2 : Math.random() < 0.5 ? 0 : 1,
      };
    };

    resize();
    let resizeTimer = 0;
    const onResize = () => {
      window.clearTimeout(resizeTimer);
      resizeTimer = window.setTimeout(resize, 180);
    };
    window.addEventListener("resize", onResize);

    // The dot grid is static, so reduced motion keeps it and drops only beams.
    if (still) {
      return () => {
        window.clearTimeout(resizeTimer);
        window.removeEventListener("resize", onResize);
      };
    }

    let running = true;
    let last = 0;

    const draw = (now: number) => {
      raf = requestAnimationFrame(draw);
      if (!running || now - last < FRAME_MS) return;
      last = now;

      const speed = chargedRef.current ? 2.2 : 1;
      ctx.clearRect(0, 0, w, h);
      ctx.lineCap = "round";

      for (let i = 0; i < beams.length; i++) {
        const b = beams[i];
        const before = b.pos;
        b.pos += b.speed * b.dir * speed;

        const span = b.axis === "h" ? w : h;
        if ((b.dir === 1 && b.pos > span + TRAIL) || (b.dir === -1 && b.pos < -TRAIL)) {
          beams[i] = spawn();
          continue;
        }

        const cross = b.lane * GRID;
        const ramp = trailRamp[b.tint];

        ctx.lineWidth = 1.8;
        for (let s = 0; s < TRAIL_STEPS; s++) {
          const from = b.pos - b.dir * (s / TRAIL_STEPS) * TRAIL;
          const to = b.pos - b.dir * ((s + 1) / TRAIL_STEPS) * TRAIL;
          ctx.strokeStyle = ramp[s];
          ctx.beginPath();
          if (b.axis === "h") {
            ctx.moveTo(from, cross);
            ctx.lineTo(to, cross);
          } else {
            ctx.moveTo(cross, from);
            ctx.lineTo(cross, to);
          }
          ctx.stroke();
        }

        const hx = b.axis === "h" ? b.pos : cross;
        const hy = b.axis === "h" ? cross : b.pos;
        ctx.fillStyle = haloColor[b.tint];
        ctx.fillRect(hx - 5, hy - 5, 10, 10);
        ctx.fillStyle = headColor[b.tint];
        ctx.fillRect(hx - 2, hy - 2, 4, 4);

        // Every intersection the head passed this frame lights up and decays.
        const a0 = Math.min(before, b.pos);
        const a1 = Math.max(before, b.pos);
        const first = Math.ceil(a0 / GRID);
        const lastCell = Math.floor(a1 / GRID);
        for (let k = first; k <= lastCell && lit.length < 220; k++) {
          lit.push({
            x: b.axis === "h" ? k * GRID : cross,
            y: b.axis === "h" ? cross : k * GRID,
            life: 1,
            tint: b.tint,
          });
        }
      }

      for (let i = lit.length - 1; i >= 0; i--) {
        const n = lit[i];
        n.life -= NODE_FADE * speed;
        if (n.life <= 0) {
          lit.splice(i, 1);
          continue;
        }
        const step = Math.min(9, Math.floor(n.life * 9));
        ctx.fillStyle = nodeRamp[n.tint][step];
        const s = 2 + n.life * 3;
        ctx.fillRect(n.x - s / 2, n.y - s / 2, s, s);
      }
    };

    const onVis = () => {
      running = !document.hidden;
      last = 0;
    };
    document.addEventListener("visibilitychange", onVis);
    raf = requestAnimationFrame(draw);

    return () => {
      running = false;
      cancelAnimationFrame(raf);
      window.clearTimeout(resizeTimer);
      window.removeEventListener("resize", onResize);
      document.removeEventListener("visibilitychange", onVis);
    };
  }, []);

  return (
    <>
      <canvas ref={boardRef} className="ambient-board" aria-hidden="true" />
      <canvas ref={canvasRef} className="ambient-canvas" aria-hidden="true" />
    </>
  );
}
