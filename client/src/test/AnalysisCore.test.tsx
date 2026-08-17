import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { AnalysisCore } from "../components/analysis/AnalysisCore";

describe("AnalysisCore", () => {
  it("keeps initiate disabled until ready", () => {
    render(
      <AnalysisCore
        phase="PARTIAL_UPLOAD"
        canAnalyze={false}
        analyzing={false}
        stageIndex={-1}
        analysis={null}
        error={null}
        onAnalyze={() => undefined}
        onReset={() => undefined}
      />
    );
    expect(screen.getByRole("button", { name: "INITIATE ANALYSIS" })).toBeDisabled();
  });

  it("shows likely match from real result payload", () => {
    render(
      <AnalysisCore
        phase="RESULT"
        canAnalyze={false}
        analyzing={false}
        stageIndex={6}
        analysis={{
          result: {
            samePersonProbability: 0.874,
            cosineSimilarity: 0.91,
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
        }}
        error={null}
        onAnalyze={() => undefined}
        onReset={() => undefined}
      />
    );
    expect(screen.getByText("LIKELY MATCH")).toBeTruthy();
    expect(screen.getByText("87.4%")).toBeTruthy();
  });
});
