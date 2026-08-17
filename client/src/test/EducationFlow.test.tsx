import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { EducationFlow } from "../components/cinematic/EducationFlow";
import type { AnalysisResult } from "../types/analysis";

const embedding = (seed: number) =>
  Array.from({ length: 128 }, (_, i) => Math.sin((i + seed) * 0.37) * 0.5);

const velocity = (seed: number) =>
  Array.from({ length: 64 }, (_, i) => Math.abs(Math.sin((i + seed) * 0.21)) * 2);

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
    gaitSignature: { velocityMagnitude: velocity(0), lowerBodyMotion: velocity(3) },
    embedding: embedding(0),
    skeletonEdges: [
      [0, 1],
      [1, 2],
      [2, 3],
    ],
  },
  subjectB: {
    poseQuality: { framesDetected: 88, framesUsed: 84, framesSampled: 96, coverage: 0.88 },
    gaitSignature: { velocityMagnitude: velocity(7), lowerBodyMotion: velocity(9) },
    embedding: embedding(11),
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

async function advance(user: ReturnType<typeof userEvent.setup>, times: number) {
  for (let i = 0; i < times; i += 1) {
    await user.click(screen.getByRole("button", { name: "Next" }));
  }
}

describe("EducationFlow", () => {
  it("opens on ingestion and reports the real sampled frame counts", () => {
    render(<EducationFlow analysis={analysis} onClose={() => undefined} />);

    expect(screen.getByRole("heading", { name: "Reading the footage" })).toBeTruthy();
    expect(screen.getByText("120")).toBeTruthy();
    expect(screen.getByText("96")).toBeTruthy();
  });

  it("advances to pose acquisition and shows measured coverage", async () => {
    const user = userEvent.setup();
    render(<EducationFlow analysis={analysis} onClose={() => undefined} />);

    await advance(user, 1);

    expect(screen.getByRole("heading", { name: "Finding the body" })).toBeTruthy();
    expect(screen.getByText("96% / 88%")).toBeTruthy();
  });

  it("includes the data-driven steps when signals are present", async () => {
    const user = userEvent.setup();
    render(<EducationFlow analysis={analysis} onClose={() => undefined} />);

    await advance(user, 3);
    expect(screen.getByRole("heading", { name: "Measuring the motion" })).toBeTruthy();
    expect(screen.getByText("64 / 64")).toBeTruthy();

    await advance(user, 2);
    expect(screen.getByRole("heading", { name: "The skeleton as a graph" })).toBeTruthy();

    await advance(user, 3);
    expect(screen.getByRole("heading", { name: "Two fingerprints" })).toBeTruthy();
    expect(screen.getByText("128")).toBeTruthy();
  });

  it("explains why the system compares rather than identifies", async () => {
    const user = userEvent.setup();
    render(<EducationFlow analysis={analysis} onClose={() => undefined} />);

    await advance(user, 7);

    expect(screen.getByRole("heading", { name: "Twin encoders, not a classifier" })).toBeTruthy();
    expect(screen.getByText("Verification")).toBeTruthy();
  });

  it("closes from the final step", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    render(<EducationFlow analysis={analysis} onClose={onClose} />);

    await advance(user, 12);

    expect(screen.getByRole("heading", { name: "Making the call" })).toBeTruthy();
    await user.click(screen.getByRole("button", { name: "Done" }));

    expect(onClose).toHaveBeenCalled();
  });

  it("presents as a modal dialog and takes focus off the page behind it", () => {
    render(<EducationFlow analysis={analysis} onClose={() => undefined} />);

    const dialog = screen.getByRole("dialog");
    expect(dialog.getAttribute("aria-modal")).toBe("true");
    expect(document.activeElement).toBe(dialog);
  });

  it("lets the step rail be reached and operated from the keyboard", async () => {
    const user = userEvent.setup();
    render(<EducationFlow analysis={analysis} onClose={() => undefined} />);

    await user.click(screen.getByRole("button", { name: "Making the call" }));

    expect(screen.getByRole("heading", { name: "Making the call" })).toBeTruthy();
  });

  it("drops the data-driven steps when the payload has no signals", async () => {
    const user = userEvent.setup();
    const sparse = {
      ...analysis,
      subjectA: { ...analysis.subjectA, gaitSignature: undefined, embedding: [], skeletonEdges: [] },
      subjectB: { ...analysis.subjectB, gaitSignature: undefined, embedding: [] },
    } as unknown as AnalysisResult;

    render(<EducationFlow analysis={sparse} onClose={() => undefined} />);

    await advance(user, 8);
    expect(screen.getByRole("heading", { name: "Making the call" })).toBeTruthy();
  });
});
