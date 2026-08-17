import { useEffect, useState } from "react";
import { audio } from "../../audio/engine";
import { fetchHealth } from "../../services/api";

type Props = {
  muted: boolean;
  onToggleMute: () => void;
};

function audioLabel(state: ReturnType<typeof audio.getState>): string {
  if (state === "unsupported") return "N/A";
  if (state === "running") return "READY";
  if (state === "suspended") return "LOCKED";
  return state.toUpperCase();
}

export function Header({ muted, onToggleMute }: Props) {
  const [health, setHealth] = useState({
    status: "…",
    modelAvailable: false,
    device: "—",
  });
  const [audioState, setAudioState] = useState(audio.getState());

  useEffect(() => {
    fetchHealth()
      .then((h) =>
        setHealth({
          status: h.status.toUpperCase(),
          modelAvailable: h.modelAvailable,
          device: h.device.toUpperCase(),
        })
      )
      .catch(() => setHealth({ status: "OFFLINE", modelAvailable: false, device: "—" }));
  }, []);

  useEffect(() => {
    const tick = () => setAudioState(audio.getState());
    tick();
    const id = window.setInterval(tick, 800);
    return () => window.clearInterval(id);
  }, [muted]);

  const online = health.status === "OK" && health.modelAvailable;
  const audioReady = audioState === "running";

  return (
    <header className="header">
      <div className="brand">
        <span className="brand-mark" aria-hidden="true">
          <svg viewBox="0 0 40 40">
            <circle cx="20" cy="20" r="17" />
            <circle cx="20" cy="20" r="11" className="dashed" />
            <circle cx="20" cy="20" r="3.5" className="filled" />
          </svg>
        </span>
        <span className="brand-text">
          <h1>DEEPGAIT</h1>
          <span>GAIT BIOMETRIC VERIFICATION SYSTEM</span>
        </span>
      </div>

      <div className="sys-status">
        <div className="sys-item">
          <span className={`status-led ${online ? "is-on" : "is-warn"}`} />
          <label>SYSTEM</label>
          <b>{health.status}</b>
        </div>
        <div className="sys-item">
          <label>MODEL</label>
          <b>{health.modelAvailable ? "READY" : "MISSING"}</b>
        </div>
        <div className="sys-item">
          <label>ENGINE</label>
          <b>ST-GCN</b>
        </div>
        <div className="sys-item">
          <label>COMPUTE</label>
          <b>{health.device}</b>
        </div>
        <div className="sys-item">
          <span className={`status-led ${audioReady && !muted ? "is-on" : muted ? "" : "is-warn"}`} />
          <label>AUDIO</label>
          <b>{muted ? "MUTED" : audioLabel(audioState)}</b>
        </div>
      </div>

      <div className="header-actions">
        <button type="button" className={muted ? "ghost" : "ghost active"} onClick={onToggleMute}>
          {muted ? "AUDIO MUTED" : "AUDIO ON"}
        </button>
      </div>
    </header>
  );
}
