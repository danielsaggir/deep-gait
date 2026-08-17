type Props = {
  a: number[];
  b: number[];
};

export function EmbeddingCompare({ a, b }: Props) {
  const cells = Math.min(a.length, b.length, 128);

  return (
    <div className="chart chart-tall">
      <h3>EMBEDDING COMPARISON · 128-D</h3>
      <div className="embedding-grid">
        <span className="embedding-row-label">A</span>
        <div className="embedding-strip">
          {a.slice(0, cells).map((v, i) => {
            const n = (v + 1) / 2;
            const c = Math.round(80 + n * 175);
            return (
              <div
                key={i}
                className="embedding-cell"
                style={{ background: `rgb(20, ${c}, ${Math.round(c * 1.1)})` }}
              />
            );
          })}
        </div>
        <span className="embedding-row-label">B</span>
        <div className="embedding-strip">
          {b.slice(0, cells).map((v, i) => {
            const n = (v + 1) / 2;
            const c = Math.round(80 + n * 175);
            return (
              <div
                key={i}
                className="embedding-cell"
                style={{ background: `rgb(20, ${c}, ${Math.round(c * 1.1)})` }}
              />
            );
          })}
        </div>
      </div>
    </div>
  );
}
