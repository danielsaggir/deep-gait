import { ReactorCore } from "../hud/ReactorCore";
import type { AnalysisResult, ApiError, Phase } from "../../types/analysis";
import { StageList } from "./StageList";

type Props = {
  phase: Phase;
  canAnalyze: boolean;
  analyzing: boolean;
  stageIndex: number;
  analysis: AnalysisResult | null;
  error: ApiError | null;
  onAnalyze: () => void;
  onReset: () => void;
};

export function AnalysisCore({
  phase,
  canAnalyze,
  analyzing,
  stageIndex,
  analysis,
  error,
  onAnalyze,
  onReset,
}: Props) {
  const match = analysis?.result.verdict === "LIKELY_MATCH";
  const reactorState = analyzing
    ? "analyzing"
    : error
      ? "error"
      : analysis
        ? match
          ? "match"
          : "different"
        : "idle";

  return (
    <section className={`core is-${reactorState}`}>
      <div className="core-head">
        <span>DEEPGAIT ANALYSIS CORE</span>
        <span className="core-head-id">STGCN // SIAMESE</span>
      </div>

      <ReactorCore
        state={reactorState}
        probability={analysis ? analysis.result.samePersonProbability : null}
        threshold={analysis ? analysis.result.threshold : 0.5}
      />

      {analysis ? (
        <div className={`verdict ${match ? "is-match" : "is-diff"}`}>
          <span className="verdict-mark" />
          {match ? "LIKELY MATCH" : "LIKELY DIFFERENT"}
        </div>
      ) : null}

      {error ? (
        <div className="core-error" role="alert">
          <div className="core-error-title">ANALYSIS INTERRUPTED</div>
          <div className="core-error-code">
            {error.code.replaceAll("_", " ")}
            {error.subject ? ` // SUBJECT ${error.subject}` : ""}
          </div>
          <p>{error.message}</p>
        </div>
      ) : null}

      {phase === "RESULT" || phase === "ERROR" ? (
        <button type="button" className="primary" onClick={onReset}>
          NEW ANALYSIS
        </button>
      ) : (
        <button
          type="button"
          className="primary"
          disabled={!canAnalyze || analyzing}
          onClick={onAnalyze}
        >
          INITIATE ANALYSIS
        </button>
      )}

      <StageList stageIndex={stageIndex} phase={phase} />
    </section>
  );
}
