import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { AnalysisDetail } from "../components/analysis/AnalysisDetail";
import { EmbeddingCompare } from "../components/charts/EmbeddingCompare";
import { GaitSignature } from "../components/charts/GaitSignature";
import { trimPadding } from "../utils/series";
import type { AnalysisResult } from "../types/analysis";

const series = (scale: number) =>
  Array.from({ length: 64 }, (_, i) => (Math.sin(i / 4) + 1.2) * scale);

const embedding = (seed: number) =>
  Array.from({ length: 128 }, (_, i) => Math.sin((i + seed) / 6));

const analysis = {
  result: {
    samePersonProbability: 0.874,
    cosineSimilarity: 0.91,
    threshold: 0.5,
    verdict: "LIKELY_MATCH",
  },
  subjectA: {
    poseQuality: { framesDetected: 90, framesUsed: 86, framesSampled: 96, coverage: 0.9 },
    poseFrames: [],
    gaitSignature: { velocityMagnitude: series(1), lowerBodyMotion: series(0.6) },
    featureComposition: {
      position: 0.4,
      angles: 0.3,
      proportions: 0.1,
      velocity: 0.2,
      acceleration: 0.1,
    },
    embedding: embedding(0),
  },
  subjectB: {
    poseQuality: { framesDetected: 88, framesUsed: 84, framesSampled: 96, coverage: 0.88 },
    poseFrames: [],
    // Two orders of magnitude quieter than A — a shared axis would flatten it.
    gaitSignature: { velocityMagnitude: series(0.01), lowerBodyMotion: series(0.006) },
    embedding: embedding(11),
  },
  model: {
    architecture: "SiameseGaitVerifier",
    sequenceLength: 64,
    joints: 17,
    inputChannels: 8,
    embeddingDimension: 128,
    device: "cpu",
  },
  timing: { total: 4.2, poseExtraction: 3.5, preprocessing: 0.4, inference: 0.3 },
} as unknown as AnalysisResult;

const renderDetail = () =>
  render(
    <AnalysisDetail analysis={analysis} overlayEnabled={false} onToggleOverlay={() => undefined} />
  );

describe("trimPadding", () => {
  it("drops the zero tail left by padding a short clip", () => {
    expect(trimPadding([3, 4, 5, 0, 0, 0])).toEqual([3, 4, 5]);
  });

  it("keeps interior zeros", () => {
    expect(trimPadding([3, 0, 5])).toEqual([3, 0, 5]);
  });

  it("reports a clip that is nothing but padding as empty", () => {
    expect(trimPadding([0, 0, 0, 0])).toEqual([]);
    expect(trimPadding([])).toEqual([]);
  });

  it("does not keep a trailing zero just to reach two samples", () => {
    expect(trimPadding([7, 0])).toEqual([7]);
  });
});

const coords = (d: string, group: 1 | 2) =>
  [...d.matchAll(/[ML]([\d.]+),([\d.]+)/g)].map((m) => Number(m[group]));

const yExtent = (d: string) => {
  const ys = coords(d, 2);
  return Math.max(...ys) - Math.min(...ys);
};

const pathOf = (container: HTMLElement, cls: string) =>
  container.querySelector(cls)?.getAttribute("d") ?? "";

describe("GaitSignature", () => {
  it("gives a quiet clip the full plot height instead of flattening it", () => {
    const { container } = render(<GaitSignature a={series(1)} b={series(0.01)} />);

    const a = container.querySelector(".series-a")?.getAttribute("d") ?? "";
    const b = container.querySelector(".series-b")?.getAttribute("d") ?? "";
    expect(a).toBeTruthy();
    expect(b).toBeTruthy();
    // Both traces use their own range, so B is as legible as A despite being 100x smaller.
    expect(yExtent(b)).toBeGreaterThan(180);
    expect(yExtent(b)).toBeCloseTo(yExtent(a), 0);
  });

  it("ignores the zero tail that padding adds to a short clip", () => {
    const short = [...series(1).slice(0, 20), ...Array(44).fill(0)];
    const { container } = render(<GaitSignature a={short} b={series(1)} />);

    const d = container.querySelector(".series-a")?.getAttribute("d") ?? "";
    expect(d.split("L").length - 1).toBe(19);
  });

  it("says so rather than drawing an empty frame when there is no signal", () => {
    render(<GaitSignature a={[]} b={[]} />);
    expect(screen.getByText("No motion signal returned")).toBeTruthy();
  });

  it("keeps both clips on one time axis so peak spacing stays comparable", () => {
    // A is 20 real frames padded to 64; B fills the window. Stretching A across
    // the full width would space its peaks 3x further apart than they are.
    const short = [...series(1).slice(0, 20), ...Array(44).fill(0)];
    const { container } = render(<GaitSignature a={short} b={series(1)} />);

    const endA = coords(pathOf(container, ".series-a"), 1).at(-1) ?? 0;
    const endB = coords(pathOf(container, ".series-b"), 1).at(-1) ?? 0;
    const startA = coords(pathOf(container, ".series-a"), 1)[0];

    // 19 of 63 intervals drawn, so A covers a little under a third of the width.
    expect((endA - startA) / (endB - startA)).toBeCloseTo(19 / 63, 2);
  });

  it("treats an all-zero clip as no signal instead of a line on the baseline", () => {
    const { container } = render(<GaitSignature a={Array(64).fill(0)} b={series(1)} />);

    expect(pathOf(container, ".series-a")).toBe("");
    expect(pathOf(container, ".series-b")).toBeTruthy();
  });

  it("drops non-finite samples rather than emitting a NaN path", () => {
    const broken = series(1).map((v, i) => (i === 7 ? NaN : v));
    const { container } = render(<GaitSignature a={broken} b={series(1)} />);

    const d = pathOf(container, ".series-a");
    expect(d).toBeTruthy();
    expect(d).not.toContain("NaN");
    expect(screen.queryByText(/peak NaN/)).toBeNull();
  });
});

describe("EmbeddingCompare", () => {
  it("spreads the colour ramp across real L2-normalised magnitudes", () => {
    // A unit vector over 128 dimensions has components near ±0.09, nowhere near
    // the ±1 a naive mapping assumes — that mapping rendered both strips flat.
    const raw = Array.from({ length: 128 }, (_, i) => Math.sin(i * 0.7));
    const norm = Math.hypot(...raw);
    const unit = raw.map((v) => v / norm);

    expect(Math.max(...unit.map(Math.abs))).toBeLessThan(0.2);

    const { container } = render(<EmbeddingCompare a={unit} b={unit} />);
    const intensities = [...container.querySelectorAll<HTMLElement>(".embedding-cell")].map((el) =>
      Number(el.style.getPropertyValue("--intensity"))
    );

    expect(Math.min(...intensities)).toBeLessThan(0.05);
    expect(Math.max(...intensities)).toBeGreaterThan(0.95);
  });
});

describe("AnalysisDetail", () => {
  it("opens on the readout without needing to be expanded first", () => {
    renderDetail();
    expect(screen.getByText("Cosine similarity")).toBeTruthy();
    expect(screen.getByText("0.910")).toBeTruthy();
  });

  it("swaps panels when another tab is selected", async () => {
    const user = userEvent.setup();
    renderDetail();

    await user.click(screen.getByRole("tab", { name: "Motion" }));
    expect(screen.getByRole("heading", { name: "Gait signature over time" })).toBeTruthy();
    expect(screen.queryByText("Cosine similarity")).toBeNull();

    await user.click(screen.getByRole("tab", { name: "Fingerprint" }));
    expect(screen.getByRole("heading", { name: /Embedding comparison/ })).toBeTruthy();
    expect(screen.getByRole("heading", { name: "Feature composition" })).toBeTruthy();
  });

  it("moves between tabs with the arrow keys", async () => {
    const user = userEvent.setup();
    renderDetail();

    const tablist = within(screen.getByRole("tablist"));
    tablist.getByRole("tab", { name: "Readout" }).focus();
    await user.keyboard("{ArrowRight}");

    expect(tablist.getByRole("tab", { name: "Motion" }).getAttribute("aria-selected")).toBe("true");
  });

  it("jumps to the first and last tab with Home and End", async () => {
    const user = userEvent.setup();
    renderDetail();

    const tablist = within(screen.getByRole("tablist"));
    tablist.getByRole("tab", { name: "Readout" }).focus();

    await user.keyboard("{End}");
    expect(tablist.getByRole("tab", { name: "Fingerprint" }).getAttribute("aria-selected")).toBe(
      "true"
    );

    await user.keyboard("{Home}");
    expect(tablist.getByRole("tab", { name: "Readout" }).getAttribute("aria-selected")).toBe("true");
  });

  it("splits pose coverage into one number per subject so it never wraps", () => {
    const { container } = renderDetail();

    const pair = container.querySelector(".metric-pair");
    expect(pair?.querySelector(".tone-a")?.textContent).toBe("90%");
    expect(pair?.querySelector(".tone-b")?.textContent).toBe("88%");
  });
});
