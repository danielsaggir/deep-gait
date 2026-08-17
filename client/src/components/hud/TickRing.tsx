type Props = {
  radius: number;
  count: number;
  length: number;
  width?: number;
  opacity?: number;
  /** Seconds per full revolution. Negative spins counter-clockwise. */
  spin?: number;
  /** Emphasise every nth tick. */
  majorEvery?: number;
};

/** Concentric ring of radial ticks, the core building block of the HUD dials. */
export function TickRing({
  radius,
  count,
  length,
  width = 1,
  opacity = 0.5,
  spin = 0,
  majorEvery = 0,
}: Props) {
  const ticks = [];
  for (let i = 0; i < count; i += 1) {
    const angle = (i / count) * 360;
    const major = majorEvery > 0 && i % majorEvery === 0;
    const len = major ? length * 2 : length;
    ticks.push(
      <line
        key={i}
        x1={100}
        y1={100 - radius}
        x2={100}
        y2={100 - radius + len}
        strokeWidth={major ? width * 1.6 : width}
        opacity={major ? Math.min(1, opacity * 1.8) : opacity}
        transform={`rotate(${angle} 100 100)`}
      />
    );
  }

  const style = spin
    ? {
        animation: `hud-spin ${Math.abs(spin)}s linear infinite`,
        animationDirection: spin < 0 ? ("reverse" as const) : ("normal" as const),
      }
    : undefined;

  return (
    <g className="tick-ring" style={style} stroke="currentColor">
      {ticks}
    </g>
  );
}
