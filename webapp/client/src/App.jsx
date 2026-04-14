import { useEffect, useState } from "react";

const api = (path) => `/api${path}`;

export default function App() {
  const [suspects, setSuspects] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [matchResult, setMatchResult] = useState(null);

  const loadSuspects = () => {
    setError(null);
    fetch(api("/suspects"))
      .then((r) => r.json())
      .then(setSuspects)
      .catch((e) => setError(String(e)));
  };

  useEffect(() => {
    loadSuspects();
  }, []);

  const onRegister = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    const fd = new FormData(e.target);
    try {
      const res = await fetch(api("/suspects"), { method: "POST", body: fd });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || res.statusText);
      e.target.reset();
      loadSuspects();
    } catch (err) {
      setError(String(err.message || err));
    } finally {
      setLoading(false);
    }
  };

  const onMatch = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setMatchResult(null);
    const fd = new FormData(e.target);
    try {
      const res = await fetch(api("/match"), { method: "POST", body: fd });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || res.statusText);
      setMatchResult(data);
    } catch (err) {
      setError(String(err.message || err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <main>
      <h1>DeepGait</h1>
      <p style={{ color: "#9aa0a6", marginTop: "-0.5rem" }}>
        Upload security footage and compare 128-D gait signatures against enrolled
        suspects.
      </p>

      {error && <p className="error">{error}</p>}

      <section>
        <h2>Enroll suspect</h2>
        <form onSubmit={onRegister}>
          <label htmlFor="name">Name</label>
          <input id="name" name="name" type="text" placeholder="Subject name" required />
          <label htmlFor="reg-video">Security video</label>
          <input id="reg-video" name="video" type="file" accept="video/*" required />
          <button type="submit" disabled={loading}>
            Extract signature & save
          </button>
        </form>
      </section>

      <section>
        <h2>Compare query video</h2>
        <form onSubmit={onMatch}>
          <label htmlFor="match-video">Query video</label>
          <input id="match-video" name="video" type="file" accept="video/*" required />
          <button type="submit" disabled={loading}>
            Run match
          </button>
        </form>
        {matchResult && (
          <div style={{ marginTop: "1rem" }}>
            <h3 style={{ fontSize: "1rem" }}>Ranked (cosine similarity)</h3>
            <table>
              <thead>
                <tr>
                  <th>#</th>
                  <th>Name</th>
                  <th>Similarity</th>
                </tr>
              </thead>
              <tbody>
                {matchResult.ranked?.map((row, i) => (
                  <tr key={String(row.suspectId)}>
                    <td>{i + 1}</td>
                    <td>{row.name}</td>
                    <td>{row.similarity.toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <section>
        <h2>Stored suspects</h2>
        <button type="button" className="secondary" onClick={loadSuspects}>
          Refresh
        </button>
        <ul>
          {suspects.map((s) => (
            <li key={s._id}>
              <strong>{s.name}</strong> — signature dim {s.signature?.length ?? 0} —{" "}
              <span style={{ color: "#9aa0a6", fontSize: "0.85rem" }}>{s.video_path}</span>
            </li>
          ))}
        </ul>
        {suspects.length === 0 && (
          <p style={{ color: "#9aa0a6" }}>No suspects yet. Enroll one above.</p>
        )}
      </section>
    </main>
  );
}
