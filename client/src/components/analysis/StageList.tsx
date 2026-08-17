import type { CSSProperties } from "react";
import { STAGES, type Phase } from "../../types/analysis";

type Props = {
  stageIndex: number;
  phase: Phase;
};

function statusFor(index: number, stageIndex: number, phase: Phase): string {
  if (phase === "ERROR" && index === Math.max(stageIndex, 0)) return "FAILED";
  if (phase === "RESULT" || index < stageIndex) return "COMPLETE";
  if (phase === "ANALYZING" && index === stageIndex) return "PROCESSING";
  return "WAITING";
}

export function StageList({ stageIndex, phase }: Props) {
  return (
    <ol className="stages">
      {STAGES.map((name, i) => {
        const status = statusFor(i, stageIndex, phase);
        return (
          <li
            className={`stage is-${status.toLowerCase()}`}
            key={name}
            style={{ "--i": i } as CSSProperties}
          >
            <span className="stage-index">{String(i + 1).padStart(2, "0")}</span>
            <span className="stage-node" />
            <span className="stage-name">{name}</span>
            <span className="stage-status">{status}</span>
          </li>
        );
      })}
    </ol>
  );
}
