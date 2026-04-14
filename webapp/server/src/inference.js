import { spawn } from "node:child_process";
import path from "node:path";
import fs from "node:fs";

/** Default relative to repo root: models/checkpoint.pth */
const DEFAULT_CHECKPOINT_REL = path.join("models", "checkpoint.pth");

/**
 * Absolute path to weights for inference. Training is never run by the server;
 * set CHECKPOINT_PATH in .env to the .pth file you saved (relative to repo or absolute).
 */
export function resolveCheckpointPath(repoRoot, checkpointPathFromEnv) {
  if (!checkpointPathFromEnv) {
    return path.join(repoRoot, DEFAULT_CHECKPOINT_REL);
  }
  return path.isAbsolute(checkpointPathFromEnv)
    ? checkpointPathFromEnv
    : path.join(repoRoot, checkpointPathFromEnv);
}

/**
 * Run Python ml.inference on a video file; returns 128-float signature array.
 */
export function runPythonInference(videoPath, env) {
  const repoRoot = env.REPO_ROOT;
  const pythonBin = env.PYTHON_BIN || "python3";
  const configPath = path.join(repoRoot, "config.yaml");

  if (!fs.existsSync(configPath)) {
    return Promise.reject(new Error(`config.yaml not found at ${configPath}`));
  }

  const checkpoint = resolveCheckpointPath(repoRoot, env.CHECKPOINT_PATH);

  const args = [
    "-m",
    "ml.inference",
    "--video",
    videoPath,
    "--checkpoint",
    checkpoint,
    "--config",
    configPath,
  ];

  return new Promise((resolve, reject) => {
    const child = spawn(pythonBin, args, {
      cwd: repoRoot,
      env: {
        ...process.env,
        PYTHONPATH: [path.join(repoRoot, "src"), process.env.PYTHONPATH || ""]
          .filter(Boolean)
          .join(path.delimiter),
      },
    });

    let stdout = "";
    let stderr = "";
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (d) => {
      stdout += d;
    });
    child.stderr.on("data", (d) => {
      stderr += d;
    });
    child.on("error", reject);
    child.on("close", (code) => {
      if (code !== 0) {
        reject(
          new Error(
            `inference exited ${code}: ${stderr.slice(-4000) || stdout}`
          )
        );
        return;
      }
      const line = stdout.trim().split("\n").filter(Boolean).pop();
      if (!line) {
        reject(new Error("empty inference output"));
        return;
      }
      try {
        const vec = JSON.parse(line);
        if (!Array.isArray(vec) || vec.length !== 128) {
          reject(new Error(`bad signature length: ${vec?.length}`));
          return;
        }
        resolve(vec);
      } catch (e) {
        reject(new Error(`invalid JSON from inference: ${line.slice(0, 200)}`));
      }
    });
  });
}
