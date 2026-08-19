import request from "supertest";
import { beforeEach, describe, expect, it, vi } from "vitest";

const checkPythonReady = vi.fn();
const runAnalysis = vi.fn();

vi.mock("../src/services/mlService.js", async () => {
  const actual = await vi.importActual<typeof import("../src/services/mlService.js")>(
    "../src/services/mlService.js"
  );
  return {
    ...actual,
    checkPythonReady: (...args: unknown[]) => checkPythonReady(...args),
    runAnalysis: (...args: unknown[]) => runAnalysis(...args),
  };
});

import { app } from "../src/index.js";

describe("API", () => {
  beforeEach(() => {
    checkPythonReady.mockResolvedValue({
      pythonAvailable: true,
      torchAvailable: true,
      modelAvailable: true,
      device: "cpu",
    });
    runAnalysis.mockReset();
  });

  it("GET /api/health is fast liveness for Render", async () => {
    const res = await request(app).get("/api/health");
    expect(res.status).toBe(200);
    expect(res.body.status).toBe("ok");
    expect(res.body.modelAvailable).toBe(true);
    expect(checkPythonReady).not.toHaveBeenCalled();
  });

  it("GET /api/health/ready checks the ML stack", async () => {
    const res = await request(app).get("/api/health/ready");
    expect(res.status).toBe(200);
    expect(res.body.status).toBe("online");
    expect(res.body.modelAvailable).toBe(true);
    expect(checkPythonReady).toHaveBeenCalled();
  });

  it("POST /api/analysis rejects missing files", async () => {
    const res = await request(app).post("/api/analysis");
    expect(res.status).toBe(400);
    expect(res.body.error.code).toBe("MISSING_VIDEO");
  });

  it("POST /api/analysis accepts two videos when ML succeeds", async () => {
    runAnalysis.mockResolvedValue({
      result: {
        samePersonProbability: 0.81,
        cosineSimilarity: 0.7,
        threshold: 0.5,
        verdict: "LIKELY_MATCH",
      },
      subjectA: {},
      subjectB: {},
      model: { device: "cpu" },
      timing: { total: 1 },
    });
    const res = await request(app)
      .post("/api/analysis")
      .attach("videoA", Buffer.from("fake"), { filename: "a.mp4", contentType: "video/mp4" })
      .attach("videoB", Buffer.from("fake"), { filename: "b.mp4", contentType: "video/mp4" });
    expect(res.status).toBe(200);
    expect(res.body.result.verdict).toBe("LIKELY_MATCH");
    expect(runAnalysis).toHaveBeenCalled();
  });
});
