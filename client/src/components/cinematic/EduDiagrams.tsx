import type { CSSProperties } from "react";
import { trimPadding } from "../../utils/series";

const W = 440;
const H = 200;

export function FrameStrip() {
  return (
    <svg viewBox="0 0 440 200" className="edu-diagram" aria-hidden="true">
      {[0, 1, 2, 3, 4].map((i) => (
        <g key={i} style={{ "--i": i } as CSSProperties} className="edu-frame">
          <rect x={20 + i * 84} y="52" width="72" height="88" rx="3" />
          <circle cx={56 + i * 84} cy="82" r="9" />
          <path d={`M ${56 + i * 84} 92 L ${56 + i * 84} 120`} />
        </g>
      ))}
      <path className="edu-arrow" d="M 20 168 L 420 168" />
      <text className="edu-axis-label" x="20" y="188">
        t = 0
      </text>
      <text className="edu-axis-label" x="420" y="188" textAnchor="end">
        64 frames
      </text>
    </svg>
  );
}

export function JointCloud() {
  const joints: Array<[number, number]> = [
    [220, 26], [200, 46], [240, 46], [182, 56], [258, 56],
    [166, 88], [274, 88], [144, 124], [296, 124], [132, 156], [308, 156],
    [190, 128], [250, 128], [182, 166], [258, 166], [176, 192], [264, 192],
  ];
  const bones: Array<[number, number]> = [
    [0, 1], [0, 2], [1, 3], [2, 4], [5, 6], [5, 7], [7, 9],
    [6, 8], [8, 10], [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
  ];
  return (
    <svg viewBox="0 0 440 210" className="edu-diagram" aria-hidden="true">
      {bones.map(([a, b], i) => (
        <line
          key={i}
          className="edu-bone"
          x1={joints[a][0]}
          y1={joints[a][1]}
          x2={joints[b][0]}
          y2={joints[b][1]}
          style={{ "--i": i } as CSSProperties}
        />
      ))}
      {joints.map(([x, y], i) => (
        <circle key={i} className="edu-joint" cx={x} cy={y} r="4.5" style={{ "--i": i } as CSSProperties} />
      ))}
    </svg>
  );
}

export function CenteringDiagram() {
  return (
    <svg viewBox="0 0 440 200" className="edu-diagram" aria-hidden="true">
      <line className="edu-axis" x1="220" y1="16" x2="220" y2="184" />
      <line className="edu-axis" x1="40" y1="100" x2="400" y2="100" />
      <circle className="edu-orbit" cx="220" cy="100" r="48" pathLength={1000} />
      <circle className="edu-orbit slow" cx="220" cy="100" r="76" pathLength={1000} />
      <circle className="edu-origin" cx="220" cy="100" r="6" />
      <text className="edu-axis-label" x="230" y="94">
        pelvis
      </text>
    </svg>
  );
}

/** Real per-frame velocity for both subjects: own height, shared time axis. */
export function MotionTrace({ a, b }: { a: number[]; b: number[] }) {
  // Each clip gets its own vertical range; a shared one buries the quieter
  // walker on the baseline and reads as a missing series. Time stays shared, or
  // a short clip would be stretched into a slower-looking cadence.
  const clean = (raw: number[]) => {
    const vals = trimPadding(raw).filter((n) => Number.isFinite(n));
    if (vals.length < 2 || Math.max(...vals) === Math.min(...vals)) return null;
    return vals;
  };

  const va = clean(a);
  const vb = clean(b);
  if (!va && !vb) return null;

  const span = Math.max(2, va?.length ?? 0, vb?.length ?? 0) - 1;

  const line = (vals: number[]) => {
    const max = Math.max(...vals);
    const min = Math.min(...vals);
    const range = max - min;
    return vals
      .map((v, i) => {
        const x = 16 + (i / span) * (W - 32);
        const y = H - 26 - ((v - min) / range) * (H - 44);
        return `${i === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
      })
      .join(" ");
  };

  const da = va ? line(va) : "";
  const db = vb ? line(vb) : "";

  return (
    <svg viewBox="0 0 440 200" className="edu-diagram" aria-hidden="true">
      <line className="edu-grid" x1="16" y1="26" x2="424" y2="26" />
      <line className="edu-grid" x1="16" y1="100" x2="424" y2="100" />
      <line className="edu-grid" x1="16" y1="174" x2="424" y2="174" />
      {da ? <path className="edu-series-a" d={da} pathLength={1000} /> : null}
      {db ? <path className="edu-series-b" d={db} pathLength={1000} /> : null}

      <rect className="edu-key-swatch-a" x="16" y="10" width="16" height="3" rx="1.5" />
      <text className="edu-axis-label edu-key-a" x="38" y="16">
        Video A
      </text>
      <rect className="edu-key-swatch-b" x="104" y="10" width="16" height="3" rx="1.5" />
      <text className="edu-axis-label edu-key-b" x="126" y="16">
        Video B
      </text>

      <text className="edu-axis-label" x="16" y="194">
        joint velocity per frame · shared time axis, each clip scaled to its own height
      </text>
    </svg>
  );
}

export function ChannelBars({ values }: { values: Array<[string, number]> }) {
  const max = Math.max(...values.map(([, v]) => v), 0.0001);
  const bw = 52;
  const gap = 20;
  const start = (440 - (values.length * bw + (values.length - 1) * gap)) / 2;
  return (
    <svg viewBox="0 0 440 200" className="edu-diagram" aria-hidden="true">
      {values.map(([label, v], i) => {
        const h = Math.max(5, (v / max) * 130);
        const x = start + i * (bw + gap);
        return (
          <g key={label} style={{ "--i": i } as CSSProperties} className="edu-bar">
            <rect x={x} y={162 - h} width={bw} height={h} rx="2" />
            <text x={x + bw / 2} y="182" textAnchor="middle">
              {label}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

/** 17×17 joint adjacency the graph convolution actually operates over. */
export function AdjacencyMatrix({ edges, joints }: { edges: Array<[number, number]>; joints: number }) {
  const n = Math.max(joints, 1);
  const cell = 150 / n;
  const ox = (440 - 150) / 2;
  const set = new Set<string>();
  for (let i = 0; i < n; i++) set.add(`${i}-${i}`);
  for (const [u, v] of edges) {
    set.add(`${u}-${v}`);
    set.add(`${v}-${u}`);
  }

  return (
    <svg viewBox="0 0 440 200" className="edu-diagram" aria-hidden="true">
      {Array.from({ length: n }).map((_, r) =>
        Array.from({ length: n }).map((__, c) => {
          const on = set.has(`${r}-${c}`);
          return (
            <rect
              key={`${r}-${c}`}
              x={ox + c * cell}
              y={24 + r * cell}
              width={cell - 0.6}
              height={cell - 0.6}
              className={on ? "edu-adj on" : "edu-adj"}
              style={{ "--i": r * n + c } as CSSProperties}
            />
          );
        })
      )}
      <text className="edu-axis-label" x={ox} y="192">
        {n} × {n} adjacency · {edges.length} bones
      </text>
    </svg>
  );
}

export function EncodingDiagram() {
  return (
    <svg viewBox="0 0 440 200" className="edu-diagram" aria-hidden="true">
      {[0, 1, 2].map((layer) =>
        [0, 1, 2, 3].map((n) => (
          <circle
            key={`${layer}-${n}`}
            className="edu-node"
            cx={70 + layer * 78}
            cy={45 + n * 38}
            r="6"
            style={{ "--i": layer * 4 + n } as CSSProperties}
          />
        ))
      )}
      {[0, 1].map((layer) =>
        [0, 1, 2, 3].map((from) =>
          [0, 1, 2, 3].map((to) => (
            <line
              key={`${layer}-${from}-${to}`}
              className="edu-edge"
              x1={70 + layer * 78}
              y1={45 + from * 38}
              x2={148 + layer * 78}
              y2={45 + to * 38}
            />
          ))
        )
      )}
      {Array.from({ length: 16 }).map((_, i) => (
        <rect
          key={i}
          className="edu-vector"
          x={330}
          y={30 + i * 9}
          width="72"
          height="6"
          style={{ "--i": i } as CSSProperties}
        />
      ))}
      <text className="edu-axis-label" x="330" y="192">
        128-D
      </text>
    </svg>
  );
}

/** Both signatures mirrored around a shared axis — A up, B down. */
export function EmbeddingFingerprint({ a, b }: { a: number[]; b: number[] }) {
  if (!a.length || !b.length) return null;
  const n = Math.min(a.length, b.length, 128);
  const scale = Math.max(...a.slice(0, n).map(Math.abs), ...b.slice(0, n).map(Math.abs), 1e-6);
  const bw = (W - 32) / n;
  return (
    <svg viewBox="0 0 440 200" className="edu-diagram" aria-hidden="true">
      <line className="edu-grid" x1="16" y1="100" x2="424" y2="100" />
      {Array.from({ length: n }).map((_, i) => {
        const ha = (Math.abs(a[i]) / scale) * 76;
        const hb = (Math.abs(b[i]) / scale) * 76;
        return (
          <g key={i}>
            <rect className="edu-fp-a" x={16 + i * bw} y={100 - ha} width={bw * 0.8} height={ha} />
            <rect className="edu-fp-b" x={16 + i * bw} y={100} width={bw * 0.8} height={hb} />
          </g>
        );
      })}
      <text className="edu-axis-label edu-key-a" x="16" y="22">
        Video A
      </text>
      <text className="edu-axis-label edu-key-b" x="16" y="194">
        Video B
      </text>
    </svg>
  );
}

/** Per-dimension absolute difference: where the two signatures disagree. */
export function DimensionDelta({ a, b }: { a: number[]; b: number[] }) {
  if (!a.length || !b.length) return null;
  const n = Math.min(a.length, b.length, 128);
  const diffs = Array.from({ length: n }, (_, i) => Math.abs(a[i] - b[i]));
  const max = Math.max(...diffs, 1e-6);
  const mean = diffs.reduce((s, d) => s + d, 0) / n;
  const bw = (W - 32) / n;
  const meanY = 168 - (mean / max) * 130;

  return (
    <svg viewBox="0 0 440 200" className="edu-diagram" aria-hidden="true">
      {diffs.map((d, i) => {
        const h = (d / max) * 130;
        return (
          <rect
            key={i}
            className={d > mean * 1.8 ? "edu-delta hot" : "edu-delta"}
            x={16 + i * bw}
            y={168 - h}
            width={bw * 0.8}
            height={h}
          />
        );
      })}
      <line className="edu-threshold" x1="16" y1={meanY} x2="424" y2={meanY} />

      {/* The key sits above the plot: anchored to the mean line it landed on
          top of the bars and became unreadable. */}
      <rect className="edu-key-swatch-mean" x="16" y="11" width="16" height="2" />
      <text className="edu-axis-label" x="38" y="16">
        mean Δ {mean.toFixed(3)}
      </text>
      <rect className="edu-delta hot" x="150" y="8" width="8" height="8" />
      <text className="edu-axis-label" x="164" y="16">
        widest gaps
      </text>

      <text className="edu-axis-label" x="16" y="192">
        per-dimension difference
      </text>
    </svg>
  );
}

export function CosineDiagram({ cosine }: { cosine: number }) {
  const angle = Math.acos(Math.max(-1, Math.min(1, cosine)));
  const len = 150;
  const cx = 120;
  const cy = 165;
  return (
    <svg viewBox="0 0 440 200" className="edu-diagram" aria-hidden="true">
      <line className="edu-vector-a" x1={cx} y1={cy} x2={cx + len} y2={cy} />
      <line
        className="edu-vector-b"
        x1={cx}
        y1={cy}
        x2={cx + len * Math.cos(angle)}
        y2={cy - len * Math.sin(angle)}
      />
      <path
        className="edu-angle"
        d={`M ${cx + 48} ${cy} A 48 48 0 0 0 ${cx + 48 * Math.cos(angle)} ${cy - 48 * Math.sin(angle)}`}
      />
      <text className="edu-figure" x={cx + 96} y={cy - 54}>
        cos θ = {cosine.toFixed(3)}
      </text>
      <text className="edu-axis-label" x={cx + len + 6} y={cy + 4}>
        A
      </text>
    </svg>
  );
}

/**
 * The logistic mapping from similarity to probability, with this pair's
 * position marked against the decision threshold.
 */
export function DecisionCurve({
  cosine,
  probability,
  threshold,
}: {
  cosine: number;
  probability: number;
  threshold: number;
}) {
  const x0 = 40;
  const x1 = 410;
  const y0 = 170;
  const y1 = 24;
  const toX = (s: number) => x0 + ((s + 1) / 2) * (x1 - x0);
  const toY = (p: number) => y0 - p * (y0 - y1);

  const curve = Array.from({ length: 60 }, (_, i) => {
    const s = -1 + (i / 59) * 2;
    const p = 1 / (1 + Math.exp(-8 * (s - 0.35)));
    return `${i === 0 ? "M" : "L"} ${toX(s).toFixed(1)} ${toY(p).toFixed(1)}`;
  }).join(" ");

  return (
    <svg viewBox="0 0 440 200" className="edu-diagram" aria-hidden="true">
      <line className="edu-axis" x1={x0} y1={y0} x2={x1} y2={y0} />
      <line className="edu-axis" x1={x0} y1={y0} x2={x0} y2={y1} />
      <line className="edu-grid" x1={x0} y1={toY(threshold)} x2={x1} y2={toY(threshold)} />
      <path className="edu-curve is-illustrative" d={curve} pathLength={1000} />
      <line className="edu-drop" x1={toX(cosine)} y1={y0} x2={toX(cosine)} y2={toY(probability)} />
      <circle className="edu-point" cx={toX(cosine)} cy={toY(probability)} r="5" />

      {/* The real classifier is an MLP over the difference, the product and the
          cosine, so no curve drawn against cosine alone can be its true
          response. The shape is labelled as a sketch and the marker as measured
          so the two are not read as lying on each other. */}
      <text className="edu-axis-label" x={x0} y="14">
        ⌁ typical response shape · ● your pair, measured
      </text>

      <text className="edu-axis-label" x={x0 - 6} y={toY(threshold) + 4} textAnchor="end">
        {(threshold * 100).toFixed(0)}%
      </text>
      <text className="edu-axis-label" x={x0} y="192">
        cosine similarity −1 → 1
      </text>
      <text className="edu-figure" x={toX(cosine) + 10} y={toY(probability) - 10}>
        {(probability * 100).toFixed(1)}%
      </text>
    </svg>
  );
}

export function DecisionDiagram({
  probability,
  threshold,
}: {
  probability: number;
  threshold: number;
}) {
  const x = 20;
  const w = 400;
  return (
    <svg viewBox="0 0 440 200" className="edu-diagram" aria-hidden="true">
      <rect className="edu-track" x={x} y="88" width={w} height="24" rx="4" />
      <rect
        className="edu-fill"
        x={x}
        y="88"
        width={w * probability}
        height="24"
        rx="4"
        style={{ "--target": `${w * probability}px` } as CSSProperties}
      />
      <line className="edu-threshold" x1={x + w * threshold} y1="70" x2={x + w * threshold} y2="130" />
      <text className="edu-axis-label" x={x} y="64">
        0%
      </text>
      <text className="edu-axis-label" x={x + w} y="64" textAnchor="end">
        100%
      </text>
      <text className="edu-axis-label" x={x + w * threshold} y="148" textAnchor="middle">
        threshold
      </text>
      <text className="edu-figure" x={x + w * probability} y="82" textAnchor="middle">
        {(probability * 100).toFixed(1)}%
      </text>
    </svg>
  );
}
