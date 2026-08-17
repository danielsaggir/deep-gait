import { useCallback, useEffect, useReducer, useRef } from "react";
import { audio } from "../../audio/engine";
import { workstationReducer, initialState } from "../../hooks/workstationReducer";
import { runAnalysis } from "../../services/api";
import type { VideoMetadata } from "../../types/analysis";
import { AnalysisCore } from "../analysis/AnalysisCore";
import { FeatureComposition } from "../charts/FeatureComposition";
import { EmbeddingCompare } from "../charts/EmbeddingCompare";
import { GaitSignature } from "../charts/GaitSignature";
import { HudFrame } from "../hud/HudFrame";
import { SubjectPanel } from "../video/SubjectPanel";
import { Header } from "./Header";

const STAGE_MS = 420;

export function Workstation() {
  const mutedRef = useRef(sessionStorage.getItem("deepgait-muted") === "1");
  const [state, dispatch] = useReducer(workstationReducer, initialState(mutedRef.current));
  const stageTimer = useRef<number | null>(null);

  useEffect(() => {
    sessionStorage.setItem("deepgait-muted", state.muted ? "1" : "0");
  }, [state.muted]);

  const play = useCallback(
    (fn: () => void) => {
      if (!state.muted) fn();
    },
    [state.muted]
  );

  const setSubject = (slot: "A" | "B", file: File, objectUrl: string, metadata: VideoMetadata) => {
    play(() => audio.accepted());
    dispatch({ type: "SET_SUBJECT", slot, file, objectUrl, metadata });
  };

  const analyze = async () => {
    if (!state.subjectA.file || !state.subjectB.file) return;
    audio.resume();
    play(() => audio.analyze());
    dispatch({ type: "START_ANALYSIS" });
    let stage = 0;
    stageTimer.current = window.setInterval(() => {
      stage = Math.min(stage + 1, 6);
      dispatch({ type: "SET_STAGE", index: stage });
      if (stage >= 6 && stageTimer.current) {
        window.clearInterval(stageTimer.current);
        stageTimer.current = null;
      }
    }, STAGE_MS);

    try {
      const result = await runAnalysis(state.subjectA.file, state.subjectB.file);
      const wait = Math.max(0, STAGE_MS * 7 - STAGE_MS);
      await new Promise((r) => setTimeout(r, Math.min(wait, 800)));
      dispatch({ type: "ANALYSIS_SUCCESS", analysis: result });
      play(() => (result.result.verdict === "LIKELY_MATCH" ? audio.success() : audio.different()));
    } catch (err) {
      const error =
        err && typeof err === "object" && "code" in err
          ? (err as { code: string; message: string; subject?: string })
          : { code: "REQUEST_FAILURE", message: "Analysis request failed." };
      dispatch({ type: "ANALYSIS_ERROR", error });
      play(() => audio.failure());
    } finally {
      if (stageTimer.current) {
        window.clearInterval(stageTimer.current);
        stageTimer.current = null;
      }
    }
  };

  const analysis = state.analysis;

  return (
    <div className="app-root">
      <div className="hud-bg" />
      <div className="workstation">
        <Header
          muted={state.muted}
          onToggleMute={() => {
            dispatch({ type: "TOGGLE_MUTE" });
          }}
        />
        <div className="stage-grid">
          <SubjectPanel
            label="SUBJECT A"
            slot={state.subjectA}
            poseFrames={analysis?.subjectA.poseFrames}
            edges={analysis?.subjectA.skeletonEdges}
            overlayEnabled={state.overlayEnabled}
            onSelect={(file, url, meta) => setSubject("A", file, url, meta)}
            onClear={() => dispatch({ type: "CLEAR_SUBJECT", slot: "A" })}
          />
          <AnalysisCore
            phase={state.phase}
            canAnalyze={state.phase === "READY_TO_ANALYZE"}
            analyzing={state.phase === "ANALYZING"}
            stageIndex={state.stageIndex}
            analysis={analysis}
            error={state.error}
            onAnalyze={() => void analyze()}
            onReset={() => dispatch({ type: "RESET" })}
          />
          <SubjectPanel
            label="SUBJECT B"
            slot={state.subjectB}
            poseFrames={analysis?.subjectB.poseFrames}
            edges={analysis?.subjectB.skeletonEdges}
            overlayEnabled={state.overlayEnabled}
            onSelect={(file, url, meta) => setSubject("B", file, url, meta)}
            onClear={() => dispatch({ type: "CLEAR_SUBJECT", slot: "B" })}
          />
        </div>
        <HudFrame className="bottom">
          {analysis ? (
            <>
              <div className="bottom-head">
                <span>ANALYSIS TELEMETRY</span>
                <button
                  type="button"
                  className={state.overlayEnabled ? "ghost active" : "ghost"}
                  onClick={() => dispatch({ type: "TOGGLE_OVERLAY" })}
                >
                  {state.overlayEnabled ? "SKELETON ON" : "SKELETON OFF"}
                </button>
              </div>
              <div className="metrics">
                <div className="metric">
                  <label>EMBEDDING COSINE</label>
                  <b>{analysis.result.cosineSimilarity.toFixed(3)}</b>
                </div>
                <div className="metric">
                  <label>DECISION THRESHOLD</label>
                  <b>{analysis.result.threshold.toFixed(3)}</b>
                </div>
                <div className="metric">
                  <label>POSE COVERAGE</label>
                  <b>
                    {(analysis.subjectA.poseQuality.coverage * 100).toFixed(0)}% /{" "}
                    {(analysis.subjectB.poseQuality.coverage * 100).toFixed(0)}%
                  </b>
                </div>
                <div className="metric">
                  <label>SEQUENCE WINDOW</label>
                  <b>{analysis.model.sequenceLength} FRAMES</b>
                </div>
                <div className="metric">
                  <label>JOINT MODEL</label>
                  <b>{analysis.model.joints} NODES</b>
                </div>
                <div className="metric">
                  <label>ANALYSIS TIME</label>
                  <b>{analysis.timing.total.toFixed(2)}s</b>
                </div>
              </div>
              <div className="charts">
                <GaitSignature
                  a={analysis.subjectA.gaitSignature.velocityMagnitude}
                  b={analysis.subjectB.gaitSignature.velocityMagnitude}
                />
                <FeatureComposition values={analysis.subjectA.featureComposition} />
                <EmbeddingCompare a={analysis.subjectA.embedding} b={analysis.subjectB.embedding} />
              </div>
            </>
          ) : (
            <div className="core-label">AWAITING DUAL SUBJECT ACQUISITION</div>
          )}
          <details className="intel">
            <summary>SYSTEM INTELLIGENCE</summary>
            <div className="pipeline">
              <span>VIDEO</span>
              <span>→ POSE ESTIMATION</span>
              <span>→ 17-JOINT SKELETON</span>
              <span>→ 8 FEATURE CHANNELS</span>
              <span>→ 64-FRAME WINDOW</span>
              <span>→ ST-GCN</span>
              <span>→ 128-D EMBEDDING</span>
              <span>→ SIAMESE COMPARISON</span>
              <span>→ MATCH PROBABILITY</span>
            </div>
          </details>
        </HudFrame>
      </div>
    </div>
  );
}
