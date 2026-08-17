import type { CSSProperties } from "react";

type Props = {
  values: {
    position: number;
    angles: number;
    proportions: number;
    velocity: number;
    acceleration: number;
  };
};

const COPY: Record<string, string> = {
  Position: "where the joints sit",
  Angles: "how limbs are bent",
  Proportions: "limb length ratios",
  Velocity: "how fast joints move",
  Acceleration: "how sharply speed changes",
};

export function FeatureComposition({ values }: Props) {
  const entries = [
    ["Position", values.position],
    ["Angles", values.angles],
    ["Proportions", values.proportions],
    ["Velocity", values.velocity],
    ["Acceleration", values.acceleration],
  ] as const;
  // Position is raw normalised coordinates and routinely runs an order of
  // magnitude above the rest, which flattens every other bar to a sliver. The
  // groups are different units anyway, so the bars are a rough sense of scale
  // and the printed numbers carry the real values.
  const max = Math.max(0.0001, ...entries.map(([, v]) => v));
  const width = (v: number) => Math.max(2, Math.sqrt(Math.max(v, 0) / max) * 100);

  return (
    <figure className="chart">
      <figcaption>
        <h3>Feature composition</h3>
        <p>
          Average magnitude of each input channel group feeding the network. The groups carry
          different units, so bars use a square-root scale and the numbers are the true values.
        </p>
      </figcaption>

      <div className="feature-bars">
        {entries.map(([label, value]) => (
          <div className="feature-row" key={label}>
            <div className="feature-head">
              <span className="feature-label">{label}</span>
              <span className="feature-value">{value.toFixed(3)}</span>
            </div>
            <div className="feature-track">
              <div
                className="feature-fill"
                style={{ "--fill": `${width(value)}%` } as CSSProperties}
              />
            </div>
            <span className="feature-copy">{COPY[label]}</span>
          </div>
        ))}
      </div>
    </figure>
  );
}
