import type { Request, Response } from "express";
import fs from "node:fs";
import { resolveCheckpoint } from "../config/env.js";
import { checkPythonReady } from "../services/mlService.js";

/** Fast liveness probe for Render (must respond within ~5s; no Python spawn). */
export function health(_req: Request, res: Response): void {
  const checkpoint = resolveCheckpoint();
  res.json({
    status: "ok",
    modelAvailable: fs.existsSync(checkpoint),
  });
}

/** Full ML stack check for the UI header (spawns Python + imports torch). */
export async function healthReady(_req: Request, res: Response): Promise<void> {
  const status = await checkPythonReady();
  const online = status.pythonAvailable && status.modelAvailable;
  res.json({
    status: online ? "online" : "degraded",
    pythonAvailable: status.pythonAvailable,
    torchAvailable: status.torchAvailable,
    modelAvailable: status.modelAvailable,
    device: status.device,
  });
}
