import type { CSSProperties } from "react";

type Props = {
  a: number[];
  b: number[];
};

export function EmbeddingCompare({ a, b }: Props) {
  const cells = Math.min(a.length, b.length, 128);
  const rowA = a.slice(0, cells);
  const rowB = b.slice(0, cells);

  // The embedding is L2-normalised over 128 dimensions, so components cluster
  // around ±0.09 rather than filling [-1, 1]. Mapping the nominal range would
  // squeeze every cell into the middle of the ramp and both strips would read
  // as flat grey, so the ramp is fitted to what the vectors actually contain.
  const extent = Math.max(1e-6, ...rowA.map(Math.abs), ...rowB.map(Math.abs));
  const style = (v: number) => ({ "--intensity": (v / extent + 1) / 2 }) as CSSProperties;

  const deltas = rowA.map((v, i) => Math.abs(v - rowB[i]));
  const maxDelta = Math.max(0.0001, ...deltas);
  const agreement = deltas.filter((d) => d < maxDelta * 0.25).length;

  return (
    <figure className="chart">
      <figcaption>
        <h3>Embedding comparison · 128 dimensions</h3>
        <p>
          Each clip collapses to one 128-number fingerprint — Video A in blue, Video B in violet.
          The bottom row is the per-dimension gap, so short bars are dimensions the two agree on.
        </p>
      </figcaption>

      <div className="embedding-grid" style={{ "--cells": cells } as CSSProperties}>
        <span className="embedding-row-label embedding-row-a">A</span>
        <div className="embedding-strip embedding-strip-a">
          {rowA.map((v, i) => (
            <div key={i} className="embedding-cell" style={style(v)} />
          ))}
        </div>

        <span className="embedding-row-label embedding-row-b">B</span>
        <div className="embedding-strip embedding-strip-b">
          {rowB.map((v, i) => (
            <div key={i} className="embedding-cell" style={style(v)} />
          ))}
        </div>

        <span className="embedding-row-label embedding-row-label-delta">Δ</span>
        <div className="embedding-strip embedding-strip-delta">
          {deltas.map((d, i) => (
            <div
              key={i}
              className="embedding-delta"
              style={{ "--gap": d / maxDelta } as CSSProperties}
            />
          ))}
        </div>
      </div>

      <p className="chart-note">
        <strong>{agreement}</strong> of {cells} dimensions land within a quarter of the largest gap.
      </p>
    </figure>
  );
}
