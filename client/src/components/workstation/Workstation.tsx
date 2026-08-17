import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import { audio } from "../../audio/engine";
import { BootSequence } from "../cinematic/BootSequence";
import { EducationFlow } from "../cinematic/EducationFlow";
import { workstationReducer, initialState } from "../../hooks/workstationReducer";
import { runAnalysis } from "../../services/api";
import type { VideoMetadata } from "../../types/analysis";
import { AnalysisDetail } from "../analysis/AnalysisDetail";
import { Console } from "../analysis/Console";
import { AmbientField } from "../hud/AmbientField";
import { SubjectPanel } from "../video/SubjectPanel";
import { Header } from "./Header";

const STAGE_MS = 420;

export function Workstation() {
  const mutedRef = useRef(sessionStorage.getItem("deepgait-muted") === "1");
  const [state, dispatch] = useReducer(workstationReducer, initialState(mutedRef.current));
  const stageTimer = useRef<number | null>(null);
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
      await new Promise((r) => setTimeout(r, 800));
      audio.analyzeEnd();
      dispatch({ type: "ANALYSIS_SUCCESS", analysis: result });
      play(() => (result.result.verdict === "LIKELY_MATCH" ? audio.success() : audio.different()));
    } catch (err) {
      const error =
        err && typeof err === "object" && "code" in err
          ? (err as { code: string; message: string; subject?: string })
          : { code: "REQUEST_FAILURE", message: "The analysis request failed." };
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

  // Replacing a clip leaves the phase on READY_TO_ANALYZE, so the shortcut
  // effect below never re-subscribes and would otherwise keep firing an
  // `analyze` closed over the file the user just swapped out.
  const analyzeRef = useRef(analyze);
  useEffect(() => {
    analyzeRef.current = analyze;
  });

  const analysis = state.analysis;
  const analyzing = state.phase === "ANALYZING";
  const isResult = state.phase === "RESULT";
  const isError = state.phase === "ERROR";
  const canAnalyze = state.phase === "READY_TO_ANALYZE";

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (target && /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName)) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      // The debrief owns the keyboard while it is open, otherwise its own
      // shortcuts collide with these and "r" tears the app down behind it.
      if (debrief || booting) return;

      if (e.key === "Enter" && canAnalyze) {
        e.preventDefault();
        void analyzeRef.current();
      }
      if (e.key.toLowerCase() === "r" && (isResult || isError)) dispatch({ type: "RESET" });
      if (e.key.toLowerCase() === "m") dispatch({ type: "TOGGLE_MUTE" });
      if (isResult && e.key === "?") setDebrief(true);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [canAnalyze, isResult, isError, debrief, booting]);

  return (
    <div className="app-root">
      <AmbientField charged={analyzing} dimmed={!analyzing} />
      {booting ? <BootSequence onComplete={() => setBooting(false)} /> : null}

      <div className={`shell ${analyzing ? "is-analyzing" : ""} ${isResult ? "is-result" : ""}`}>
        <Header muted={state.muted} onToggleMute={() => dispatch({ type: "TOGGLE_MUTE" })} />

        <main className="deck">
          <div className="subjects">
            <SubjectPanel
              label="Video A"
              tone="a"
              slot={state.subjectA}
              poseFrames={analysis?.subjectA.poseFrames}
              edges={analysis?.subjectA.skeletonEdges}
              overlayEnabled={state.overlayEnabled}
              scanning={analyzing}
              playing={isResult}
              onSelect={(file, url, meta) => setSubject("A", file, url, meta)}
              onClear={() => dispatch({ type: "CLEAR_SUBJECT", slot: "A" })}
            />
            <SubjectPanel
              label="Video B"
              tone="b"
              slot={state.subjectB}
              poseFrames={analysis?.subjectB.poseFrames}
              edges={analysis?.subjectB.skeletonEdges}
              overlayEnabled={state.overlayEnabled}
              scanning={analyzing}
              playing={isResult}
              onSelect={(file, url, meta) => setSubject("B", file, url, meta)}
              onClear={() => dispatch({ type: "CLEAR_SUBJECT", slot: "B" })}
            />
          </div>

          <Console
            phase={state.phase}
            subjectA={state.subjectA}
            subjectB={state.subjectB}
            stageIndex={state.stageIndex}
            analysis={analysis}
            error={state.error}
            onAnalyze={() => void analyze()}
            onReset={() => dispatch({ type: "RESET" })}
            onDebrief={() => setDebrief(true)}
          />

          {isResult && analysis ? (
            <AnalysisDetail
              analysis={analysis}
              overlayEnabled={state.overlayEnabled}
              onToggleOverlay={() => dispatch({ type: "TOGGLE_OVERLAY" })}
            />
          ) : null}
        </main>
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
