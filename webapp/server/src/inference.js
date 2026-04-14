import { spawn } from "node:child_process";
import path from "node:path";
import fs from "node:fs";

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

  const checkpoint =
    env.CHECKPOINT_PATH ||
    path.join(repoRoot, "models", "checkpoint.pth");

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
