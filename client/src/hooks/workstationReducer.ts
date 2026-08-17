import type { SubjectSlot, WorkstationAction, WorkstationState } from "../types/analysis";

const emptySlot = (): SubjectSlot => ({ file: null, objectUrl: null, metadata: null });

export function initialState(muted = false): WorkstationState {
  return {
    phase: "READY",
    subjectA: emptySlot(),
    subjectB: emptySlot(),
    analysis: null,
    error: null,
    stageIndex: -1,
    muted,
    overlayEnabled: true,
  };
}

function derivePhase(a: SubjectSlot, b: SubjectSlot): WorkstationState["phase"] {
  if (a.file && b.file) return "READY_TO_ANALYZE";
  if (a.file || b.file) return "PARTIAL_UPLOAD";
  return "READY";
}

export function workstationReducer(
  state: WorkstationState,
  action: WorkstationAction
): WorkstationState {
  switch (action.type) {
    case "SET_SUBJECT": {
      if (state.subjectA.objectUrl && action.slot === "A") URL.revokeObjectURL(state.subjectA.objectUrl);
      if (state.subjectB.objectUrl && action.slot === "B") URL.revokeObjectURL(state.subjectB.objectUrl);
      const next = {
        ...state,
        analysis: null,
        error: null,
        subjectA:
          action.slot === "A"
            ? { file: action.file, objectUrl: action.objectUrl, metadata: action.metadata }
            : state.subjectA,
        subjectB:
          action.slot === "B"
            ? { file: action.file, objectUrl: action.objectUrl, metadata: action.metadata }
            : state.subjectB,
      };
      next.phase = derivePhase(next.subjectA, next.subjectB);
      return next;
    }
    case "CLEAR_SUBJECT": {
      const slot = action.slot === "A" ? state.subjectA : state.subjectB;
      if (slot.objectUrl) URL.revokeObjectURL(slot.objectUrl);
      const next = {
        ...state,
        analysis: null,
        error: null,
        subjectA: action.slot === "A" ? emptySlot() : state.subjectA,
        subjectB: action.slot === "B" ? emptySlot() : state.subjectB,
      };
      next.phase = derivePhase(next.subjectA, next.subjectB);
      return next;
    }
    case "START_ANALYSIS":
      return { ...state, phase: "ANALYZING", error: null, analysis: null, stageIndex: 0 };
    case "SET_STAGE":
      return { ...state, stageIndex: action.index };
    case "ANALYSIS_SUCCESS":
      return { ...state, phase: "RESULT", analysis: action.analysis, error: null, stageIndex: 6 };
    case "ANALYSIS_ERROR":
      return { ...state, phase: "ERROR", error: action.error, analysis: null };
    case "RESET": {
      if (state.subjectA.objectUrl) URL.revokeObjectURL(state.subjectA.objectUrl);
      if (state.subjectB.objectUrl) URL.revokeObjectURL(state.subjectB.objectUrl);
      return initialState(state.muted);
    }
    case "TOGGLE_MUTE":
      return { ...state, muted: !state.muted };
    case "TOGGLE_OVERLAY":
      return { ...state, overlayEnabled: !state.overlayEnabled };
    default:
      return state;
  }
}
