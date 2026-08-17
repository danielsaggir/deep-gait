import type { Request, Response } from "express";
import { checkPythonReady } from "../services/mlService.js";

export async function health(_req: Request, res: Response): Promise<void> {
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
