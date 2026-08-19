import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { PYTHON_BIN, REPO_ROOT, ANALYSIS_TIMEOUT_MS, resolveCheckpoint } from "../config/env.js";
import type { AnalysisErrorBody, AnalysisSuccess } from "../types/analysis.js";

export class MlProcessError extends Error {
  status: number;
  code: string;
  subject?: string;

  constructor(status: number, code: string, message: string, subject?: string) {
    super(message);
    this.status = status;
    this.code = code;
    this.subject = subject;
  }
}

function pythonEnv(): NodeJS.ProcessEnv {
  const serverMl = path.join(REPO_ROOT, "server");
  return {
    ...process.env,
    PYTHONPATH: [serverMl, process.env.PYTHONPATH || ""].filter(Boolean).join(path.delimiter),
  };
}

type PythonReadyStatus = {
  pythonAvailable: boolean;
  torchAvailable: boolean;
  modelAvailable: boolean;
  device: string;
};

const READY_CACHE_MS = 60_000;
const READY_CHECK_TIMEOUT_MS = 8_000;

let readyCache: { at: number; value: PythonReadyStatus } | null = null;
let readyInflight: Promise<PythonReadyStatus> | null = null;

function unavailableReady(modelAvailable: boolean): PythonReadyStatus {
  return {
    pythonAvailable: false,
    torchAvailable: false,
    modelAvailable,
    device: "cpu",
  };
}

export function checkPythonReady(): Promise<PythonReadyStatus> {
  const checkpoint = resolveCheckpoint();
  const modelAvailable = fs.existsSync(checkpoint);

  if (readyCache && Date.now() - readyCache.at < READY_CACHE_MS) {
    return Promise.resolve(readyCache.value);
  }
  if (readyInflight) return readyInflight;

  readyInflight = new Promise((resolve) => {
    const finish = (value: PythonReadyStatus) => {
      readyCache = { at: Date.now(), value };
      readyInflight = null;
      resolve(value);
    };

    const child = spawn(
      PYTHON_BIN,
      [
        "-c",
        "import json,sys\n"
          + "out={'pythonAvailable':True,'torchAvailable':False,'device':'cpu'}\n"
          + "try:\n"
          + " import torch\n"
          + " out['torchAvailable']=True\n"
          + " out['device']='cuda' if torch.cuda.is_available() else 'cpu'\n"
          + "except Exception:\n"
          + " pass\n"
          + "print(json.dumps(out))",
      ],
      { env: pythonEnv() }
    );

    let stdout = "";
    const timer = setTimeout(() => {
      child.kill("SIGKILL");
      finish(unavailableReady(modelAvailable));
    }, READY_CHECK_TIMEOUT_MS);

    child.stdout?.on("data", (d) => {
      stdout += d;
    });
    child.on("error", () => {
      clearTimeout(timer);
      finish(unavailableReady(modelAvailable));
    });
    child.on("close", (code) => {
      clearTimeout(timer);
      if (code !== 0) {
        finish(unavailableReady(modelAvailable));
        return;
      }
      try {
        const parsed = JSON.parse(stdout.trim().split("\n").filter(Boolean).pop() || "{}");
        finish({
          pythonAvailable: true,
          torchAvailable: Boolean(parsed.torchAvailable),
          modelAvailable,
          device: parsed.device === "cuda" ? "cuda" : "cpu",
        });
      } catch {
        finish({ pythonAvailable: true, torchAvailable: false, modelAvailable, device: "cpu" });
      }
    });
  });

  return readyInflight;
}

const DETAIL_LIMIT = 220;

/**
 * Python writes results to stdout and diagnostics to stderr. When the result is
 * missing, stderr holds the only explanation, so surface its tail rather than
 * reporting a bare failure.
 */
function mlFailure(stderr: string, summary: string): MlProcessError {
  const trimmed = stderr.trim();
  if (trimmed) console.error(`[ml] ${trimmed}`);

  const detail = trimmed.split("\n").map((l) => l.trim()).filter(Boolean).pop() ?? "";
  const short = detail.length > DETAIL_LIMIT ? `${detail.slice(0, DETAIL_LIMIT)}…` : detail;
  return new MlProcessError(
    500,
    "INFERENCE_FAILURE",
    short ? `${summary} ${short}` : summary
  );
}

export function runAnalysis(videoA: string, videoB: string): Promise<AnalysisSuccess> {
  const checkpoint = resolveCheckpoint();
  if (!fs.existsSync(checkpoint)) {
    return Promise.reject(
      new MlProcessError(503, "CHECKPOINT_ERROR", "Trained checkpoint is not available.")
    );
  }

  const args = [
    "-m",
    "ml.inference",
    "--video-a",
    videoA,
    "--video-b",
    videoB,
    "--checkpoint",
    checkpoint,
  ];

  return new Promise((resolve, reject) => {
    const child = spawn(PYTHON_BIN, args, {
      cwd: REPO_ROOT,
      env: pythonEnv(),
    });

    let stdout = "";
    let stderr = "";
    const timer = setTimeout(() => {
      child.kill("SIGKILL");
      reject(new MlProcessError(504, "INFERENCE_TIMEOUT", "Analysis exceeded the time limit."));
    }, ANALYSIS_TIMEOUT_MS);

    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (d) => {
      stdout += d;
    });
    child.stderr.on("data", (d) => {
      stderr += d;
    });
    child.on("error", (err) => {
      clearTimeout(timer);
      reject(new MlProcessError(503, "PYTHON_UNAVAILABLE", "Python runtime could not be started."));
      void err;
    });
    child.on("close", (code) => {
      clearTimeout(timer);

      const line = stdout.trim().split("\n").filter(Boolean).pop();
      if (!line) {
        reject(mlFailure(stderr, "The ML runtime exited without returning a result."));
        return;
      }

      let parsed: AnalysisSuccess | AnalysisErrorBody;
      try {
        parsed = JSON.parse(line);
      } catch {
        reject(mlFailure(stderr, "The ML runtime returned malformed output."));
        return;
      }

      if ("error" in parsed && parsed.error) {
        const status = parsed.error.code === "INSUFFICIENT_GAIT_DATA" ? 422 : 500;
        reject(
          new MlProcessError(
            status,
            parsed.error.code || "INFERENCE_FAILURE",
            parsed.error.message || "Analysis failed.",
            parsed.error.subject
          )
        );
        return;
      }

      if (code !== 0) {
        reject(mlFailure(stderr, "Analysis failed."));
        return;
      }

      resolve(parsed as AnalysisSuccess);
    });
  });
}
