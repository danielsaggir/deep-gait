import type { NextFunction, Request, Response } from "express";
import { runAnalysis } from "../services/mlService.js";
import { makeTempDir, removeDir } from "../services/tempFiles.js";

type Uploaded = { [field: string]: Express.Multer.File[] };

export function attachTempDir(req: Request, _res: Response, next: NextFunction): void {
  (req as Request & { tempDir?: string }).tempDir = makeTempDir();
  next();
}

export async function analyze(req: Request, res: Response, next: NextFunction): Promise<void> {
  const tempDir = (req as Request & { tempDir?: string }).tempDir;
  try {
    const files = req.files as Uploaded | undefined;
    const videoA = files?.videoA?.[0];
    const videoB = files?.videoB?.[0];
    if (!videoA || !videoB) {
      res.status(400).json({
        error: {
          code: "MISSING_VIDEO",
          message: "Both Subject A and Subject B footage are required.",
        },
      });
      return;
    }
    const result = await runAnalysis(videoA.path, videoB.path);
    res.json(result);
  } catch (err) {
    next(err);
  } finally {
    removeDir(tempDir);
  }
}
