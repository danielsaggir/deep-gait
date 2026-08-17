type Props = {
  values: {
    position: number;
    angles: number;
    proportions: number;
    velocity: number;
    acceleration: number;
  };
};

export function FeatureComposition({ values }: Props) {
  const entries = [
    ["POSITION", values.position],
    ["ANGLES", values.angles],
    ["PROPORTIONS", values.proportions],
    ["VELOCITY", values.velocity],
    ["ACCELERATION", values.acceleration],
  ] as const;
  const max = Math.max(0.0001, ...entries.map(([, v]) => v));
  return (
    <div className="chart">
      <h3>FEATURE COMPOSITION</h3>
      {entries.map(([label, value]) => (
        <div key={label} style={{ display: "grid", gridTemplateColumns: "110px 1fr", gap: 8, marginBottom: 6 }}>
          <span style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--muted)" }}>{label}</span>
          <div style={{ background: "rgba(77,232,255,0.08)", height: 8 }}>
            <div style={{ width: `${(value / max) * 100}%`, height: "100%", background: "#4de8ff" }} />
          </div>
        </div>
      ))}
    </div>
  );
}
