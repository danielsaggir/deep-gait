import { ReactorCore } from "../hud/ReactorCore";
import type { AnalysisResult, ApiError, Phase, SubjectSlot } from "../../types/analysis";
import { StageRail } from "./StageRail";

type Props = {
  phase: Phase;
  subjectA: SubjectSlot;
  subjectB: SubjectSlot;
  stageIndex: number;
  analysis: AnalysisResult | null;
  error: ApiError | null;
  onAnalyze: () => void;
  onReset: () => void;
  onDebrief: () => void;
};

function readableCode(code: string): string {
  const s = code.toLowerCase().replace(/_/g, " ");
  return s.charAt(0).toUpperCase() + s.slice(1);
}

function clipSummary(a: SubjectSlot, b: SubjectSlot): string {
  const secs = (s: SubjectSlot) => (s.metadata ? `${s.metadata.duration.toFixed(1)}s` : "—");
  return `${secs(a)} and ${secs(b)} of footage loaded. Both clips will be sampled to a 64-frame window and passed through the twin encoders together.`;
}

export function Console({
  phase,
  subjectA,
  subjectB,
  stageIndex,
  analysis,
  error,
  onAnalyze,
  onReset,
  onDebrief,
}: Props) {
  const analyzing = phase === "ANALYZING";
  const isResult = phase === "RESULT" && analysis !== null;
  const isError = phase === "ERROR";
  const ready = phase === "READY_TO_ANALYZE";
  const loaded = [subjectA.file, subjectB.file].filter(Boolean).length;

  const reactorState = analyzing
    ? "analyzing"
    : isError
      ? "error"
      : isResult
        ? analysis.result.verdict === "LIKELY_MATCH"
          ? "match"
          : "different"
        : "idle";

  const tone = isResult
    ? analysis.result.verdict === "LIKELY_MATCH"
      ? "is-match"
      : "is-diff"
    : isError
      ? "is-error"
      : "";

  // One short sentence per phase. Making the whole console a live region meant
  // every stage tick and the keyed remount of the body were re-announced, which
  // buries the one thing a screen reader user is waiting for. Errors are left
  // to the role="alert" below so they are not read out twice.
  const status = analyzing
    ? "Comparing gait."
    : isResult
      ? `${
          analysis.result.verdict === "LIKELY_MATCH"
            ? "Likely the same person"
            : "Likely different people"
        }. Match probability ${(analysis.result.samePersonProbability * 100).toFixed(1)} percent.`
      : ready
        ? "Both clips loaded. Ready to compare."
        : "";

  return (
    <section className={`console ${tone}`}>
      <p className="sr-only" role="status">
        {status}
      </p>

      <div className="console-core">
        <ReactorCore
          state={reactorState}
          probability={isResult ? analysis.result.samePersonProbability : null}
          threshold={analysis?.result.threshold ?? 0.5}
        />
      </div>

      <div className="console-body" key={phase}>
        {isResult ? (
          <>
            <h2>
              {analysis.result.verdict === "LIKELY_MATCH"
                ? "Likely the same person"
                : "Likely different people"}
            </h2>
            <p>
              The verifier puts the probability that both clips show the same walker at{" "}
              <strong>{(analysis.result.samePersonProbability * 100).toFixed(1)}%</strong>,{" "}
              {analysis.result.verdict === "LIKELY_MATCH" ? "above" : "below"} the{" "}
              {(analysis.result.threshold * 100).toFixed(0)}% decision threshold. On subjects it had
              never seen during training the model was right about 85% of the time, so read this as
              evidence rather than an identification.
            </p>
            <div className="console-actions">
              <button type="button" className="btn btn-primary btn-lg" onClick={onReset}>
                New comparison
              </button>
              <button type="button" className="btn btn-secondary" onClick={onDebrief}>
                Walk me through it
              </button>
            </div>
          </>
        ) : isError ? (
          <>
            <h2>Analysis stopped</h2>
            <p role="alert">
              <span className="console-code">
                {readableCode(error?.code ?? "Unknown")}
                {error?.subject ? ` · Video ${error.subject}` : ""}
              </span>
              {error?.message}
            </p>
            <div className="console-actions">
              <button type="button" className="btn btn-primary btn-lg" onClick={onReset}>
                Start over
              </button>
            </div>
          </>
        ) : analyzing ? (
          <>
            <h2>Comparing gait</h2>
            <p>
              Both clips are running through identical encoders that share the same weights, so each
              one is measured by exactly the same yardstick before the two signatures meet.
            </p>
          </>
        ) : (
          <>
            <h2>{ready ? "Ready to compare" : "Load two clips"}</h2>
            <p>
              {ready
                ? clipSummary(subjectA, subjectB)
                : `${loaded} of 2 loaded. DeepGait answers one question — is this the same person walking in both clips? Give it a few seconds of walking with the whole body in frame, ideally filmed from the side.`}
            </p>
            <div className="console-actions">
              <button
                type="button"
                className="btn btn-primary btn-lg"
                disabled={!ready}
                onClick={onAnalyze}
              >
                Run comparison
              </button>
            </div>
          </>
        )}
      </div>

      <StageRail stageIndex={stageIndex} phase={phase} />
    </section>
  );
}
