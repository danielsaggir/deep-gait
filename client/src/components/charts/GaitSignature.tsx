import { trimPadding } from "../../utils/series";

type Props = {
  a: number[];
  b: number[];
  /** Optional second pair drawn faintly behind, e.g. lower-body motion. */
  subA?: number[];
  subB?: number[];
};

// The chart sits in a wide, short slot, so the viewBox is cut to match: a
// squarer one letterboxes against its max-height and wastes the width it has.
const W = 960;
const H = 230;
const PAD = { l: 44, r: 16, t: 16, b: 30 };
const IW = W - PAD.l - PAD.r;
const IH = H - PAD.t - PAD.b;

type Scaled = { points: Array<[number, number]>; peak: number };

/**
 * Drops padding, non-finite samples and series with no variation at all, which
 * would otherwise draw as a flat line pinned to the baseline — the exact
 * artefact that reads as missing data.
 */
function clean(values: number[]): number[] | null {
  const v = trimPadding(values).filter((n) => Number.isFinite(n));
  if (v.length < 2) return null;
  return Math.max(...v) === Math.min(...v) ? null : v;
}

/**
 * Height is scaled per clip: raw velocity magnitude depends on cadence, framing
 * and camera distance, so a shared vertical axis flattens the quieter clip onto
 * the baseline. Time is scaled across both, because a short clip stretched to
 * the full width would show peaks further apart than they really are — and peak
 * spacing is precisely what the caption asks the reader to compare.
 */
function scale(v: number[], span: number): Scaled {
  const max = Math.max(...v);
  const min = Math.min(...v);
  const range = max - min;
  return {
    peak: max,
    points: v.map((x, i) => [
      PAD.l + (i / span) * IW,
      PAD.t + IH - ((x - min) / range) * IH,
    ]),
  };
}

const toLine = (p: Array<[number, number]>) =>
  p.map(([x, y], i) => `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`).join(" ");

const toArea = (p: Array<[number, number]>) =>
  `${toLine(p)} L${p[p.length - 1][0].toFixed(1)},${PAD.t + IH} L${p[0][0].toFixed(1)},${PAD.t + IH} Z`;

export function GaitSignature({ a, b, subA, subB }: Props) {
  const ca = clean(a);
  const cb = clean(b);
  const cla = subA ? clean(subA) : null;
  const clb = subB ? clean(subB) : null;

  const span =
    Math.max(2, ca?.length ?? 0, cb?.length ?? 0, cla?.length ?? 0, clb?.length ?? 0) - 1;

  const sa = ca && scale(ca, span);
  const sb = cb && scale(cb, span);
  const la = cla && scale(cla, span);
  const lb = clb && scale(clb, span);

  return (
    <figure className="chart">
      <figcaption>
        <h3>Gait signature over time</h3>
        <p>
          Per-frame joint velocity, each clip scaled to its own range. Compare the rhythm and the
          spacing of the peaks rather than the heights.
        </p>
      </figcaption>

      <div className="chart-legend">
        <span className="legend-a">Video A{sa ? ` · peak ${sa.peak.toFixed(2)}` : ""}</span>
        <span className="legend-b">Video B{sb ? ` · peak ${sb.peak.toFixed(2)}` : ""}</span>
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} className="chart-svg" role="img" aria-label="Gait signature">
        {[0, 0.25, 0.5, 0.75, 1].map((f) => (
          <line
            key={f}
            className="chart-grid"
            x1={PAD.l}
            x2={W - PAD.r}
            y1={PAD.t + IH * f}
            y2={PAD.t + IH * f}
          />
        ))}
        <line
          className="chart-axis"
          x1={PAD.l}
          y1={PAD.t}
          x2={PAD.l}
          y2={PAD.t + IH}
        />

        <text className="chart-axis-label" x={PAD.l - 8} y={PAD.t + 4} textAnchor="end">
          peak
        </text>
        <text className="chart-axis-label" x={PAD.l - 8} y={PAD.t + IH + 4} textAnchor="end">
          rest
        </text>
        <text className="chart-axis-label" x={PAD.l} y={H - 10}>
          frame 0
        </text>
        <text className="chart-axis-label" x={W - PAD.r} y={H - 10} textAnchor="end">
          end of window
        </text>

        {la ? <path className="series-sub series-sub-a" d={toLine(la.points)} /> : null}
        {lb ? <path className="series-sub series-sub-b" d={toLine(lb.points)} /> : null}

        {sa ? <path className="series-area-a" d={toArea(sa.points)} /> : null}
        {sb ? <path className="series-area-b" d={toArea(sb.points)} /> : null}
        {sa ? <path className="series-a" d={toLine(sa.points)} /> : null}
        {sb ? <path className="series-b" d={toLine(sb.points)} /> : null}

        {!sa && !sb ? (
          <text className="chart-empty" x={W / 2} y={H / 2} textAnchor="middle">
            No motion signal returned
          </text>
        ) : null}
      </svg>
    </figure>
  );
}
