import { TickRing } from "./TickRing";

type State = "idle" | "analyzing" | "match" | "different" | "error";

type Props = {
  state: State;
  /** Real classifier probability in [0,1]. Null until the model returns. */
  probability: number | null;
  threshold: number;
};

const ARC_RADIUS = 70;
const ARC_CIRCUMFERENCE = 2 * Math.PI * ARC_RADIUS;

export function ReactorCore({ state, probability, threshold }: Props) {
  const shown = probability ?? 0;
  const dashOffset = ARC_CIRCUMFERENCE * (1 - shown);
  const thresholdAngle = threshold * 360;

  return (
    <div className={`reactor reactor-${state}`}>
      <svg viewBox="0 0 200 200" className="reactor-svg" aria-hidden="true">
        <defs>
          <radialGradient id="coreGlow">
            <stop offset="0%" stopColor="currentColor" stopOpacity="0.5" />
            <stop offset="55%" stopColor="currentColor" stopOpacity="0.1" />
            <stop offset="100%" stopColor="currentColor" stopOpacity="0" />
          </radialGradient>
        </defs>

        <circle cx="100" cy="100" r="62" fill="url(#coreGlow)" className="reactor-glow" />

        <TickRing radius={94} count={120} length={4} opacity={0.25} spin={110} majorEvery={10} />
        <circle cx="100" cy="100" r="86" className="ring hairline" />
        <TickRing radius={82} count={48} length={5} opacity={0.38} spin={-58} majorEvery={12} />

        <circle cx="100" cy="100" r={ARC_RADIUS} className="ring track" />
        <circle
          cx="100"
          cy="100"
          r={ARC_RADIUS}
          className="ring value-arc"
          strokeDasharray={ARC_CIRCUMFERENCE}
          strokeDashoffset={dashOffset}
          transform="rotate(-90 100 100)"
        />
        <line
          x1="100"
          y1={100 - ARC_RADIUS - 7}
          x2="100"
          y2={100 - ARC_RADIUS + 7}
          className="threshold-mark"
          transform={`rotate(${thresholdAngle} 100 100)`}
        />

        <circle cx="100" cy="100" r="58" className="ring hairline" />
        <g className="spokes">
          {[0, 45, 90, 135, 180, 225, 270, 315].map((a) => (
            <line key={a} x1="100" y1="42" x2="100" y2="50" transform={`rotate(${a} 100 100)`} />
          ))}
        </g>

        <circle cx="100" cy="100" r="46" className="ring arc-segment segment-a" />
        <circle cx="100" cy="100" r="38" className="ring arc-segment segment-b" />
        <circle cx="100" cy="100" r="28" className="ring hairline" />
        <circle cx="100" cy="100" r="2.5" className="core-pin" />
      </svg>

      {state === "analyzing" ? <div className="reactor-sweep" aria-hidden="true" /> : null}

      <div className="reactor-readout">
        {probability === null ? (
          <>
            <span className="readout-idle">
              {state === "analyzing"
                ? "PROCESSING"
                : state === "error"
                  ? "INTERRUPTED"
                  : "STANDBY"}
            </span>
            <span className="readout-sub">DEEPGAIT CORE</span>
          </>
        ) : (
          <>
            <span className="readout-label">MATCH PROBABILITY</span>
            <span className="readout-value">{(shown * 100).toFixed(1)}%</span>
            <span className="readout-sub">THRESHOLD {(threshold * 100).toFixed(0)}%</span>
          </>
        )}
      </div>
    </div>
  );
}
