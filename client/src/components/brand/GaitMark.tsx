type Props = {
  /** Unique suffix so multiple marks on one page keep their own gradients. */
  id?: string;
  className?: string;
};

/** Joint positions for a single mid-stride frame, side view. */
const JOINTS: Array<[number, number]> = [
  [20, 13], // shoulder
  [20, 22], // hip
  [26.5, 17], // front elbow
  [25, 22.5], // front hand
  [13.5, 17], // rear elbow
  [12.5, 22.5], // rear hand
  [26.5, 28], // front knee
  [29, 35], // front foot
  [14, 29], // rear knee
  [11, 35], // rear foot
];

const BONES = "M20 13 L20 22 M20 13 L26.5 17 L25 22.5 M20 13 L13.5 17 L12.5 22.5 M20 22 L26.5 28 L29 35 M20 22 L14 29 L11 35";

/**
 * The mark is the product: a skeleton rendered as a graph, caught mid-stride.
 * Nodes are joints, strokes are bones — the same structure the adjacency matrix
 * encodes and the network convolves over.
 */
export function GaitMark({ id = "brand", className }: Props) {
  const grad = `${id}-grad`;

  return (
    <svg viewBox="0 0 40 40" className={className} aria-hidden="true">
      <defs>
        <linearGradient id={grad} x1="0%" y1="100%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="var(--indigo)" />
          <stop offset="55%" stopColor="var(--accent)" />
          <stop offset="100%" stopColor="var(--accent-bright)" />
        </linearGradient>
      </defs>

      {/* Ground line and stride arc: the temporal half of the signature. */}
      <path
        d="M6 36.5 H34"
        stroke="var(--accent)"
        strokeOpacity="0.28"
        strokeWidth="1"
        strokeLinecap="round"
        fill="none"
      />
      <path
        d="M11 35 Q20 26 29 35"
        stroke="var(--accent)"
        strokeOpacity="0.35"
        strokeWidth="1"
        strokeDasharray="2 2.5"
        strokeLinecap="round"
        fill="none"
      />

      <circle cx="20" cy="7.5" r="3.4" fill="none" stroke={`url(#${grad})`} strokeWidth="2" />
      <path
        d={BONES}
        fill="none"
        stroke={`url(#${grad})`}
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />

      {JOINTS.map(([x, y]) => (
        <circle key={`${x}-${y}`} cx={x} cy={y} r="1.5" fill="var(--accent-bright)" />
      ))}
    </svg>
  );
}
