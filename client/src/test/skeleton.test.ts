import { describe, expect, it } from "vitest";
import { nearestPoseFrame } from "../utils/skeleton";

describe("nearestPoseFrame", () => {
  it("returns the closest timestamped frame", () => {
    const frames = [
      { timestamp: 0, joints: [] },
      { timestamp: 0.08, joints: [] },
      { timestamp: 0.16, joints: [] },
    ];
    expect(nearestPoseFrame(frames, 0.09)?.timestamp).toBe(0.08);
    expect(nearestPoseFrame([], 0)).toBeNull();
  });
});
