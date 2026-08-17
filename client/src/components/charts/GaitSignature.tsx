type Props = {
  a: number[];
  b: number[];
};

export function GaitSignature({ a, b }: Props) {
  const w = 320;
  const h = 88;
  const max = Math.max(0.0001, ...a, ...b);
  const path = (values: number[]) => {
    if (!values.length) return "";
    return values
      .map((v, i) => {
        const x = (i / Math.max(values.length - 1, 1)) * w;
        const y = h - (v / max) * (h - 8) - 4;
        return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");
  };
  return (
    <div className="chart">
      <h3>TEMPORAL GAIT SIGNATURE</h3>
      <svg viewBox={`0 0 ${w} ${h}`} width="100%" height="88">
        <path d={path(a)} fill="none" stroke="#4de8ff" strokeWidth="1.5" />
        <path d={path(b)} fill="none" stroke="#6ea8ff" strokeWidth="1.5" />
      </svg>
    </div>
  );
}
