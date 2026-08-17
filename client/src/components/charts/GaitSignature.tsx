type Props = {
  a: number[];
  b: number[];
};

export function GaitSignature({ a, b }: Props) {
  const w = 360;
  const h = 120;
  const pad = { l: 28, r: 8, t: 12, b: 24 };
  const innerW = w - pad.l - pad.r;
  const innerH = h - pad.t - pad.b;
  const max = Math.max(0.0001, ...a, ...b);

  const path = (values: number[]) => {
    if (!values.length) return "";
    return values
      .map((v, i) => {
        const x = pad.l + (i / Math.max(values.length - 1, 1)) * innerW;
        const y = pad.t + innerH - (v / max) * innerH;
        return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");
  };

  return (
    <div className="chart chart-tall">
      <h3>TEMPORAL GAIT SIGNATURE</h3>
      <div className="chart-legend">
        <span className="legend-a">SUBJECT A</span>
        <span className="legend-b">SUBJECT B</span>
      </div>
      <svg viewBox={`0 0 ${w} ${h}`} width="100%" height={h} className="chart-svg">
        <line
          x1={pad.l}
          y1={pad.t + innerH}
          x2={w - pad.r}
          y2={pad.t + innerH}
          className="chart-axis"
        />
        <line x1={pad.l} y1={pad.t} x2={pad.l} y2={pad.t + innerH} className="chart-axis" />
        <text x={pad.l - 4} y={pad.t + 4} className="chart-axis-label" textAnchor="end">
          MAX
        </text>
        <text x={pad.l - 4} y={pad.t + innerH} className="chart-axis-label" textAnchor="end">
          0
        </text>
        <text x={pad.l} y={h - 4} className="chart-axis-label">
          0
        </text>
        <text x={w - pad.r} y={h - 4} className="chart-axis-label" textAnchor="end">
          64
        </text>
        <path d={path(a)} fill="none" className="series-a" />
        <path d={path(b)} fill="none" className="series-b" />
      </svg>
    </div>
  );
}
