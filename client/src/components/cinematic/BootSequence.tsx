import { useEffect, useState, type CSSProperties } from "react";
import { GaitMark } from "../brand/GaitMark";

type Props = {
  onComplete: () => void;
};

const LINES = ["Loading Siamese ST-GCN", "Starting pose estimator", "Ready"];

const LINE_MS = 300;
const HOLD_MS = 320;
const EXIT_MS = 620;

/** Ignition sequence played once on load. Click or press any key to skip. */
export function BootSequence({ onComplete }: Props) {
  const [visible, setVisible] = useState(0);
  const [exiting, setExiting] = useState(false);

  useEffect(() => {
    const timers: number[] = [];

    LINES.forEach((_, i) => {
      timers.push(window.setTimeout(() => setVisible(i + 1), 320 + i * LINE_MS));
    });

    timers.push(
      window.setTimeout(() => setExiting(true), 320 + LINES.length * LINE_MS + HOLD_MS)
    );
    timers.push(
      window.setTimeout(onComplete, 320 + LINES.length * LINE_MS + HOLD_MS + EXIT_MS)
    );

    return () => timers.forEach(window.clearTimeout);
  }, [onComplete]);

  useEffect(() => {
    const skip = () => {
      setExiting(true);
      window.setTimeout(onComplete, 420);
    };
    window.addEventListener("keydown", skip, { once: true });
    return () => window.removeEventListener("keydown", skip);
  }, [onComplete]);

  return (
    <div
      className={`boot ${exiting ? "is-exiting" : ""}`}
      onClick={() => {
        setExiting(true);
        window.setTimeout(onComplete, 420);
      }}
    >
      <div className="boot-aurora" aria-hidden="true" />

      <div className="boot-core">
        <svg viewBox="0 0 240 240" aria-hidden="true">
          <defs>
            <linearGradient id="bootGrad" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="var(--accent)" />
              <stop offset="100%" stopColor="var(--indigo)" />
            </linearGradient>
          </defs>
          <circle className="boot-ring r1" cx="120" cy="120" r="110" pathLength={1000} stroke="url(#bootGrad)" />
          <circle className="boot-ring r2" cx="120" cy="120" r="88" pathLength={1000} stroke="url(#bootGrad)" />
          <circle className="boot-ring r3" cx="120" cy="120" r="64" pathLength={1000} stroke="url(#bootGrad)" />
          <circle className="boot-ring r4" cx="120" cy="120" r="40" pathLength={1000} stroke="url(#bootGrad)" />
          <circle className="boot-spark" cx="120" cy="120" r="3" />
        </svg>
        <span className="boot-mark">
          <GaitMark id="boot" />
        </span>
        <div className="boot-wordmark">DeepGait</div>
      </div>

      <ol className="boot-lines">
        {LINES.map((line, i) => (
          <li
            key={line}
            className={i < visible ? "is-shown" : ""}
            style={{ "--i": i } as CSSProperties}
          >
            <span className="boot-tick" />
            {line}
          </li>
        ))}
      </ol>

      <div className="boot-hint">Click or press any key to skip</div>
    </div>
  );
}
