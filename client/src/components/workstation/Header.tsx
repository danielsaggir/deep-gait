import { useEffect, useState } from "react";
import { audio } from "../../audio/engine";
import { fetchHealth } from "../../services/api";
import { GaitMark } from "../brand/GaitMark";

type Props = {
  muted: boolean;
  onToggleMute: () => void;
};

export function Header({ muted, onToggleMute }: Props) {
  const [health, setHealth] = useState({
    status: "Checking",
    modelAvailable: false,
    device: "—",
  });

  useEffect(() => {
    fetchHealth()
      .then((h) =>
        setHealth({
          status: h.status.toLowerCase() === "ok" ? "Online" : h.status,
          modelAvailable: h.modelAvailable,
          device: h.device.toUpperCase(),
        })
      )
      .catch(() => setHealth({ status: "Offline", modelAvailable: false, device: "—" }));
  }, []);

  const online = health.status === "Online" && health.modelAvailable;

  return (
    <header className="header">
      <div className="brand">
        <span className="brand-mark">
          <GaitMark id="header" />
        </span>
        <span className="brand-text">
          <h1>DeepGait</h1>
          <span>Skeleton-based gait verification</span>
        </span>
      </div>

      <div className="sys-status">
        <span className="sys-item">
          <span className={`status-led ${online ? "is-on" : "is-warn"}`} aria-hidden="true" />
          {online ? "Model ready" : health.status}
        </span>
        <span className="sys-item sys-item-muted">Siamese ST-GCN</span>
        <span className="sys-item sys-item-muted">{health.device}</span>
      </div>

      <div className="header-actions">
        <button
          type="button"
          className="btn btn-quiet"
          onClick={() => {
            audio.resume();
            onToggleMute();
          }}
          aria-pressed={!muted}
        >
          <span className={`btn-dot ${muted ? "" : "is-live"}`} aria-hidden="true" />
          {muted ? "Sound off" : "Sound on"}
        </button>
      </div>
    </header>
  );
}
