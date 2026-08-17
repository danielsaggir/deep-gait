import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { Console } from "../components/analysis/Console";
import type { AnalysisResult, SubjectSlot } from "../types/analysis";

const empty: SubjectSlot = { file: null, objectUrl: null, metadata: null };

const noop = {
  onAnalyze: () => undefined,
  onReset: () => undefined,
  onDebrief: () => undefined,
};

describe("Console", () => {
  it("keeps the run action disabled until both clips are loaded", () => {
    render(
      <Console
        phase="PARTIAL_UPLOAD"
        subjectA={empty}
        subjectB={empty}
        stageIndex={-1}
        analysis={null}
        error={null}
        {...noop}
      />
    );

    expect(screen.getByRole("button", { name: "Run comparison" })).toBeDisabled();
  });

  it("marks the stage in progress while analysing", () => {
    render(
      <Console
        phase="ANALYZING"
        subjectA={empty}
        subjectB={empty}
        stageIndex={1}
        analysis={null}
        error={null}
        {...noop}
      />
    );

    const rail = within(screen.getByRole("list"));
    expect(rail.getByText("Extracting pose").closest("li")?.className).toContain("is-processing");
    expect(rail.getByText("Decoding clips").closest("li")?.className).toContain("is-complete");
  });

  it("leads with a plain-language verdict once a result lands", () => {
    render(
      <Console
        phase="RESULT"
        subjectA={empty}
        subjectB={empty}
        stageIndex={6}
        analysis={
          {
            result: {
              samePersonProbability: 0.874,
              cosineSimilarity: 0.91,
              threshold: 0.5,
              verdict: "LIKELY_MATCH",
            },
          } as unknown as AnalysisResult
        }
        error={null}
        {...noop}
      />
    );

    expect(screen.getByRole("heading", { name: "Likely the same person" })).toBeTruthy();
    expect(screen.getByText(/above the 50% decision threshold/)).toBeTruthy();
  });

  it("explains a failure and offers a way out", async () => {
    const user = userEvent.setup();
    const onReset = vi.fn();
    render(
      <Console
        phase="ERROR"
        subjectA={empty}
        subjectB={empty}
        stageIndex={2}
        analysis={null}
        error={{
          code: "INSUFFICIENT_GAIT_DATA",
          message: "Not enough walking was visible.",
          subject: "B",
        }}
        {...noop}
        onReset={onReset}
      />
    );

    expect(screen.getByText("Insufficient gait data · Video B")).toBeTruthy();
    await user.click(screen.getByRole("button", { name: "Start over" }));
    expect(onReset).toHaveBeenCalled();
  });
});
