import { STAGES, type Phase } from "../../types/analysis";

type Props = {
  stageIndex: number;
  phase: Phase;
};

function statusFor(index: number, stageIndex: number, phase: Phase): string {
  if (phase === "ERROR" && index === Math.max(stageIndex, 0)) return "failed";
  if (phase === "RESULT" || index < stageIndex) return "complete";
  if (phase === "ANALYZING" && index === stageIndex) return "processing";
  return "waiting";
}

/** Horizontal pipeline rail. Always present, so the console never jumps height. */
export function StageRail({ stageIndex, phase }: Props) {
  const done = phase === "RESULT" ? STAGES.length : Math.max(stageIndex, 0);
  const progress = phase === "READY" || stageIndex < 0 ? 0 : (done / STAGES.length) * 100;

  return (
    <div className={`rail is-${phase.toLowerCase()}`}>
      <div className="rail-progress">
        <span style={{ width: `${progress}%` }} />
      </div>
      <ol>
        {STAGES.map((name, i) => (
          <li key={name} className={`rail-step is-${statusFor(i, stageIndex, phase)}`}>
            <span className="rail-node" aria-hidden="true" />
            <span className="rail-name">{name}</span>
          </li>
        ))}
      </ol>
    </div>
  );
}
