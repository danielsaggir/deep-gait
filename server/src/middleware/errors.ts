import type { NextFunction, Request, Response } from "express";
import { MlProcessError } from "../services/mlService.js";
import { removeDir } from "../services/tempFiles.js";

export function errorHandler(err: unknown, req: Request, res: Response, _next: NextFunction): void {
  const tempDir = (req as Request & { tempDir?: string }).tempDir;
  removeDir(tempDir);

  if (err instanceof MlProcessError) {
    res.status(err.status).json({
      error: {
        code: err.code,
        message: err.message,
        ...(err.subject ? { subject: err.subject } : {}),
      },
    });
    return;
  }

  const message = err instanceof Error ? err.message : "Unexpected error";
  if (message === "UNSUPPORTED_VIDEO") {
    res.status(400).json({
      error: { code: "UNSUPPORTED_VIDEO", message: "Unsupported or unreadable video file." },
    });
    return;
  }
  if (message.includes("File too large")) {
    res.status(400).json({
      error: { code: "FILE_TOO_LARGE", message: "Video exceeds the upload size limit." },
    });
    return;
  }

  res.status(500).json({
    error: { code: "SERVER_ERROR", message: "The analysis workstation encountered an error." },
  });
}
