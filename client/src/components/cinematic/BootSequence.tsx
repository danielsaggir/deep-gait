import { useEffect, useState, type CSSProperties } from "react";

type Props = {
  onComplete: () => void;
};

const LINES = [
  "POWERING BIOMETRIC CORE",
  "MOUNTING ST-GCN WEIGHTS",
  "CALIBRATING POSE ENGINE",
  "GAIT VERIFICATION SUITE ONLINE",
];

const LINE_MS = 460;
const HOLD_MS = 620;
const EXIT_MS = 900;

/** Ignition sequence played once on load. Click or press any key to skip. */
export function BootSequence({ onComplete }: Props) {
  const [visible, setVisible] = useState(0);
  const [exiting, setExiting] = useState(false);

  useEffect(() => {
    const timers: number[] = [];

    LINES.forEach((_, i) => {
      timers.push(window.setTimeout(() => setVisible(i + 1), 700 + i * LINE_MS));
    });

    timers.push(
      window.setTimeout(() => setExiting(true), 700 + LINES.length * LINE_MS + HOLD_MS)
    );
    timers.push(
      window.setTimeout(onComplete, 700 + LINES.length * LINE_MS + HOLD_MS + EXIT_MS)
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
      <div className="boot-letterbox top" />
      <div className="boot-letterbox bottom" />

      <div className="boot-core">
        <svg viewBox="0 0 240 240" aria-hidden="true">
          <circle className="boot-ring r1" cx="120" cy="120" r="110" pathLength={1000} />
          <circle className="boot-ring r2" cx="120" cy="120" r="88" pathLength={1000} />
          <circle className="boot-ring r3" cx="120" cy="120" r="64" pathLength={1000} />
          <circle className="boot-ring r4" cx="120" cy="120" r="40" pathLength={1000} />
          <circle className="boot-spark" cx="120" cy="120" r="3" />
        </svg>
        <div className="boot-wordmark">DEEPGAIT</div>
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

      <div className="boot-hint">PRESS ANY KEY TO SKIP</div>
    </div>
  );
}
