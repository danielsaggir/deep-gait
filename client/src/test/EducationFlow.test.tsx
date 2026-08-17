import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { EducationFlow } from "../components/cinematic/EducationFlow";
import type { AnalysisResult } from "../types/analysis";

const analysis = {
  result: {
    samePersonProbability: 0.874,
    cosineSimilarity: 0.912,
    threshold: 0.5,
    verdict: "LIKELY_MATCH",
  },
  subjectA: {
    poseQuality: { framesDetected: 118, framesUsed: 110, framesSampled: 120, coverage: 0.96 },
    featureComposition: {
      position: 0.4,
      angles: 0.2,
      proportions: 0.1,
      velocity: 0.2,
      acceleration: 0.1,
    },
  },
  subjectB: {
    poseQuality: { framesDetected: 88, framesUsed: 84, framesSampled: 96, coverage: 0.88 },
  },
  model: {
    architecture: "SiameseGaitVerifier",
    embeddingDimension: 128,
    inputChannels: 8,
    sequenceLength: 64,
    joints: 17,
    device: "cpu",
  },
  timing: { poseExtraction: 1, preprocessing: 0, inference: 0, total: 1 },
} as unknown as AnalysisResult;

describe("EducationFlow", () => {
  it("opens on ingestion and reports the real sampled frame counts", () => {
    render(<EducationFlow analysis={analysis} onClose={() => undefined} />);

    expect(screen.getByRole("heading", { name: "FOOTAGE INGESTION" })).toBeTruthy();
    expect(screen.getByText("120")).toBeTruthy();
    expect(screen.getByText("96")).toBeTruthy();
  });

  it("advances to pose acquisition and shows measured coverage", async () => {
    const user = userEvent.setup();
    render(<EducationFlow analysis={analysis} onClose={() => undefined} />);

    await user.click(screen.getByRole("button", { name: "NEXT" }));

    expect(screen.getByRole("heading", { name: "POSE ACQUISITION" })).toBeTruthy();
    expect(screen.getByText("96% / 88%")).toBeTruthy();
  });

  it("closes from the final step", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    render(<EducationFlow analysis={analysis} onClose={onClose} />);

    for (let i = 0; i < 6; i += 1) {
      await user.click(screen.getByRole("button", { name: "NEXT" }));
    }

    expect(screen.getByRole("heading", { name: "CLASSIFICATION" })).toBeTruthy();
    await user.click(screen.getByRole("button", { name: "DONE" }));

    expect(onClose).toHaveBeenCalled();
  });
});
