type Props = {
  /** Subject slots that hold footage, so their line reads as charged. */
  chargedA: boolean;
  chargedB: boolean;
  /** Packets only travel while the model is actually working. */
  flowing: boolean;
};

const PACKET_DELAYS = ["0s", "0.35s", "0.7s", "1.05s", "1.4s", "1.75s"];

/**
 * Conduits running from each subject panel into the analysis core. The panels
 * paint over the ends, so the lines appear to emerge from behind the footage
 * and disappear into the reactor.
 */
export function FlowConduit({ chargedA, chargedB, flowing }: Props) {
  return (
    <svg
      className={`conduit ${flowing ? "is-flowing" : ""}`}
      viewBox="0 0 1000 300"
      preserveAspectRatio="none"
      aria-hidden="true"
    >
      <defs>
        {/* Normalised length so one dash pattern works for both curves. */}
        <path id="conduit-a" pathLength={1000} d="M 110 96 C 250 96, 300 150, 440 150" />
        <path id="conduit-b" pathLength={1000} d="M 890 96 C 750 96, 700 150, 560 150" />
      </defs>

      {(
        [
          ["a", chargedA],
          ["b", chargedB],
        ] as const
      ).map(([side, charged]) => (
        <g key={side} className={`conduit-group ${charged ? "is-charged" : ""}`}>
          <use href={`#conduit-${side}`} className="conduit-track" />
          {flowing
            ? PACKET_DELAYS.map((delay) => (
                <use
                  key={delay}
                  href={`#conduit-${side}`}
                  className="conduit-packet"
                  style={{ animationDelay: delay }}
                />
              ))
            : null}
        </g>
      ))}
    </svg>
  );
}
