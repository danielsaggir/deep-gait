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
    <div className="chart chart-tall">
      <h3>FEATURE COMPOSITION</h3>
      <div className="feature-bars">
        {entries.map(([label, value]) => (
          <div className="feature-row" key={label}>
            <span className="feature-label">{label}</span>
            <div className="feature-track">
              <div className="feature-fill" style={{ width: `${(value / max) * 100}%` }} />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
