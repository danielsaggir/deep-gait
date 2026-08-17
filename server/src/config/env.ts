import path from "node:path";
import { fileURLToPath } from "node:url";
import dotenv from "dotenv";

const here = path.dirname(fileURLToPath(import.meta.url));
export const SERVER_DIR = path.resolve(here, "..", "..");
export const REPO_ROOT = path.resolve(SERVER_DIR, "..");

dotenv.config({ path: path.join(SERVER_DIR, ".env") });
dotenv.config({ path: path.join(REPO_ROOT, ".env") });

export const PORT = Number(process.env.PORT || 3001);
export const PYTHON_BIN = process.env.PYTHON_BIN || "python3";
export const MAX_UPLOAD_MB = Number(process.env.MAX_UPLOAD_MB || 80);
export const ANALYSIS_TIMEOUT_MS = Number(process.env.ANALYSIS_TIMEOUT_MS || 180_000);

export function resolveCheckpoint(): string {
  const fromEnv = process.env.CHECKPOINT_PATH;
  if (fromEnv) {
    return path.isAbsolute(fromEnv) ? fromEnv : path.join(REPO_ROOT, fromEnv);
  }
  return path.join(SERVER_DIR, "ml", "weights", "best_gait_verifier.pth");
}
