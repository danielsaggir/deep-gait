import { describe, expect, it } from "vitest";
import { initialState, workstationReducer } from "../hooks/workstationReducer";

const file = new File(["x"], "walk-a.mp4", { type: "video/mp4" });
const meta = { source: "walk-a.mp4", duration: 8, width: 1920, height: 1080, format: "MP4" };

describe("workstation state", () => {
  it("starts ready with analyze disabled until two videos", () => {
    let state = initialState();
    expect(state.phase).toBe("READY");
    state = workstationReducer(state, {
      type: "SET_SUBJECT",
      slot: "A",
      file,
      objectUrl: "blob:a",
      metadata: meta,
    });
    expect(state.phase).toBe("PARTIAL_UPLOAD");
    state = workstationReducer(state, {
      type: "SET_SUBJECT",
      slot: "B",
      file,
      objectUrl: "blob:b",
      metadata: { ...meta, source: "walk-b.mp4" },
    });
    expect(state.phase).toBe("READY_TO_ANALYZE");
  });

  it("moves analyzing → result → reset", () => {
    let state = initialState();
    state = workstationReducer(state, { type: "START_ANALYSIS" });
    expect(state.phase).toBe("ANALYZING");
    state = workstationReducer(state, {
      type: "ANALYSIS_SUCCESS",
      analysis: {
        result: {
          samePersonProbability: 0.87,
          cosineSimilarity: 0.9,
          threshold: 0.5,
          verdict: "LIKELY_MATCH",
        },
        subjectA: {} as never,
        subjectB: {} as never,
        model: {
          architecture: "SiameseGaitVerifier",
          embeddingDimension: 128,
          inputChannels: 8,
          sequenceLength: 64,
          joints: 17,
          device: "cpu",
        },
        timing: { poseExtraction: 1, preprocessing: 0, inference: 0, total: 1 },
      },
    });
    expect(state.phase).toBe("RESULT");
    expect(state.analysis?.result.verdict).toBe("LIKELY_MATCH");
    state = workstationReducer(state, { type: "RESET" });
    expect(state.phase).toBe("READY");
    expect(state.analysis).toBeNull();
  });

  it("renders error without losing mute preference", () => {
    let state = initialState(true);
    state = workstationReducer(state, {
      type: "ANALYSIS_ERROR",
      error: { code: "INSUFFICIENT_GAIT_DATA", message: "bad pose", subject: "B" },
    });
    expect(state.phase).toBe("ERROR");
    expect(state.muted).toBe(true);
  });
});
