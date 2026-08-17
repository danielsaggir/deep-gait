import {
  useCallback,
  useEffect,
  useReducer,
  useRef,
  useState,
  type CSSProperties,
} from "react";
import { audio } from "../../audio/engine";
import { BootSequence } from "../cinematic/BootSequence";
import { EducationFlow } from "../cinematic/EducationFlow";
import { workstationReducer, initialState } from "../../hooks/workstationReducer";
import { runAnalysis } from "../../services/api";
import type { AnalysisResult, VideoMetadata } from "../../types/analysis";
import { AnalysisCore } from "../analysis/AnalysisCore";
import { FeatureComposition } from "../charts/FeatureComposition";
import { EmbeddingCompare } from "../charts/EmbeddingCompare";
import { GaitSignature } from "../charts/GaitSignature";
import { FlowConduit } from "../hud/FlowConduit";
import { HudFrame } from "../hud/HudFrame";
import { SubjectPanel } from "../video/SubjectPanel";
import { Header } from "./Header";

const STAGE_MS = 420;

function metricsFor(analysis: AnalysisResult): Array<[string, string]> {
  const coverage = (s: "subjectA" | "subjectB") =>
    (analysis[s].poseQuality.coverage * 100).toFixed(0);

  return [
    ["EMBEDDING COSINE", analysis.result.cosineSimilarity.toFixed(3)],
    ["DECISION THRESHOLD", analysis.result.threshold.toFixed(3)],
    ["POSE COVERAGE", `${coverage("subjectA")}% / ${coverage("subjectB")}%`],
    ["SEQUENCE WINDOW", `${analysis.model.sequenceLength} FRAMES`],
    ["JOINT MODEL", `${analysis.model.joints} NODES`],
    ["ANALYSIS TIME", `${analysis.timing.total.toFixed(2)}s`],
  ];
}

export function Workstation() {
  const mutedRef = useRef(sessionStorage.getItem("deepgait-muted") === "1");
  const [state, dispatch] = useReducer(workstationReducer, initialState(mutedRef.current));
  const stageTimer = useRef<number | null>(null);
  const videoARef = useRef<HTMLVideoElement | null>(null);
  const videoBRef = useRef<HTMLVideoElement | null>(null);
  const playbackCuePlayed = useRef(false);
  const [booting, setBooting] = useState(true);
  const [debrief, setDebrief] = useState(false);

  useEffect(() => {
    const unlock = () => audio.resume();
    window.addEventListener("pointerdown", unlock, { once: true });
    window.addEventListener("keydown", unlock, { once: true });
    return () => {
      window.removeEventListener("pointerdown", unlock);
      window.removeEventListener("keydown", unlock);
    };
  }, []);

  useEffect(() => {
    sessionStorage.setItem("deepgait-muted", state.muted ? "1" : "0");
    if (state.muted) audio.silence();
  }, [state.muted]);

  useEffect(() => {
    if (state.phase !== "RESULT") {
      playbackCuePlayed.current = false;
      return;
    }
    if (playbackCuePlayed.current) return;
    playbackCuePlayed.current = true;
    if (!state.muted) audio.playbackStart();
  }, [state.phase, state.muted]);

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

  const replayBoth = () => {
    [videoARef, videoBRef].forEach((ref) => {
      const v = ref.current;
      if (!v) return;
      v.currentTime = 0;
      void v.play().catch(() => undefined);
    });
  };

  const analyze = async () => {
    if (!state.subjectA.file || !state.subjectB.file) return;
    audio.resume();
    play(() => audio.analyzeStart());
    dispatch({ type: "START_ANALYSIS" });
    let stage = 0;
    stageTimer.current = window.setInterval(() => {
      stage = Math.min(stage + 1, 6);
      dispatch({ type: "SET_STAGE", index: stage });
      play(() => audio.stageAdvance());
      if (stage >= 6 && stageTimer.current) {
        window.clearInterval(stageTimer.current);
        stageTimer.current = null;
      }
    }, STAGE_MS);

    try {
      const result = await runAnalysis(state.subjectA.file, state.subjectB.file);
      const wait = Math.max(0, STAGE_MS * 7 - STAGE_MS);
      await new Promise((r) => setTimeout(r, Math.min(wait, 800)));
      audio.analyzeEnd();
      dispatch({ type: "ANALYSIS_SUCCESS", analysis: result });
      play(() => (result.result.verdict === "LIKELY_MATCH" ? audio.success() : audio.different()));
    } catch (err) {
      const error =
        err && typeof err === "object" && "code" in err
          ? (err as { code: string; message: string; subject?: string })
          : { code: "REQUEST_FAILURE", message: "Analysis request failed." };
      audio.analyzeEnd();
      dispatch({ type: "ANALYSIS_ERROR", error });
      play(() => audio.failure());
    } finally {
      audio.analyzeEnd();
      if (stageTimer.current) {
        window.clearInterval(stageTimer.current);
        stageTimer.current = null;
      }
    }
  };

  const analysis = state.analysis;
  const analyzing = state.phase === "ANALYZING";
  const isResult = state.phase === "RESULT";
  const match = analysis?.result.verdict === "LIKELY_MATCH";

  return (
    <div className="app-root">
      <div className="hud-bg" />
      {booting ? <BootSequence onComplete={() => setBooting(false)} /> : null}
      {analyzing ? <div className="screen-sweep" aria-hidden="true" /> : null}
      {analysis ? (
        <div
          className="result-flash"
          aria-hidden="true"
          style={{ color: match ? "var(--ok)" : "var(--warn)" }}
        />
      ) : null}
      <div
        className={`workstation ${analyzing ? "is-analyzing" : ""} ${isResult ? "is-result" : ""}`}
      >
        <Header
          muted={state.muted}
          onToggleMute={() => {
            dispatch({ type: "TOGGLE_MUTE" });
          }}
        />
        <div className="stage-grid">
          <FlowConduit
            chargedA={Boolean(state.subjectA.file)}
            chargedB={Boolean(state.subjectB.file)}
            flowing={analyzing}
          />
          <SubjectPanel
            label="SUBJECT A"
            slot={state.subjectA}
            poseFrames={analysis?.subjectA.poseFrames}
            edges={analysis?.subjectA.skeletonEdges}
            overlayEnabled={state.overlayEnabled}
            scanning={analyzing}
            playing={isResult}
            onVideoRef={(el) => {
              videoARef.current = el;
            }}
            onSelect={(file, url, meta) => setSubject("A", file, url, meta)}
            onClear={() => dispatch({ type: "CLEAR_SUBJECT", slot: "A" })}
          />
          <AnalysisCore
            phase={state.phase}
            canAnalyze={state.phase === "READY_TO_ANALYZE"}
            analyzing={analyzing}
            stageIndex={state.stageIndex}
            analysis={analysis}
            error={state.error}
            onAnalyze={() => void analyze()}
            onReset={() => dispatch({ type: "RESET" })}
            onDebrief={() => setDebrief(true)}
          />
          <SubjectPanel
            label="SUBJECT B"
            slot={state.subjectB}
            poseFrames={analysis?.subjectB.poseFrames}
            edges={analysis?.subjectB.skeletonEdges}
            overlayEnabled={state.overlayEnabled}
            scanning={analyzing}
            playing={isResult}
            onVideoRef={(el) => {
              videoBRef.current = el;
            }}
            onSelect={(file, url, meta) => setSubject("B", file, url, meta)}
            onClear={() => dispatch({ type: "CLEAR_SUBJECT", slot: "B" })}
          />
        </div>
        <HudFrame className="bottom">
          {analysis ? (
            <>
              <div className="bottom-head">
                <span>ANALYSIS TELEMETRY</span>
                <div className="bottom-actions">
                  {isResult ? (
                    <button type="button" className="ghost active" onClick={replayBoth}>
                      REPLAY BOTH
                    </button>
                  ) : null}
                  <button type="button" className="ghost" onClick={() => setDebrief(true)}>
                    PIPELINE DEBRIEF
                  </button>
                  <button
                    type="button"
                    className={state.overlayEnabled ? "ghost active" : "ghost"}
                    onClick={() => dispatch({ type: "TOGGLE_OVERLAY" })}
                  >
                    {state.overlayEnabled ? "SKELETON ON" : "SKELETON OFF"}
                  </button>
                </div>
              </div>
              <div className="metrics">
                {metricsFor(analysis).map(([label, value], i) => (
                  <div className="metric" key={label} style={{ "--i": i } as CSSProperties}>
                    <label>{label}</label>
                    <b>{value}</b>
                  </div>
                ))}
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
      {debrief && analysis ? (
        <EducationFlow
          analysis={analysis}
          soundEnabled={!state.muted}
          onClose={() => setDebrief(false)}
        />
      ) : null}
    </div>
  );
}
