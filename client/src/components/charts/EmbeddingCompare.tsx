type Props = {
  a: number[];
  b: number[];
};

export function EmbeddingCompare({ a, b }: Props) {
  const cells = Math.min(a.length, b.length, 128);
  return (
    <div className="chart">
      <h3>EMBEDDING COMPARISON · 128-D</h3>
      <div style={{ display: "grid", gap: 6 }}>
        {[a, b].map((vec, row) => (
          <div key={row} style={{ display: "grid", gridTemplateColumns: `repeat(${cells}, 1fr)`, height: 18 }}>
            {vec.slice(0, cells).map((v, i) => {
              const n = (v + 1) / 2;
              const c = Math.round(80 + n * 175);
              return <div key={i} style={{ background: `rgb(20, ${c}, ${Math.round(c * 1.1)})` }} />;
            })}
          </div>
        ))}
      </div>
    </div>
  );
}
