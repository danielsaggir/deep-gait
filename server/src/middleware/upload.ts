import type { Request } from "express";
import multer from "multer";
import path from "node:path";
import { MAX_UPLOAD_MB } from "../config/env.js";

const ALLOWED = new Set([
  "video/mp4",
  "video/webm",
  "video/quicktime",
  "video/x-msvideo",
  "video/x-matroska",
  "application/octet-stream",
]);

type RequestWithTemp = Request & { tempDir?: string };

export const uploadPair = multer({
  storage: multer.diskStorage({
    destination: (req, _file, cb) => {
      const dir = (req as RequestWithTemp).tempDir;
      if (!dir) {
        cb(new Error("missing temp directory"), "");
        return;
      }
      cb(null, dir);
    },
    filename: (_req, file, cb) => {
      const safe = file.originalname.replace(/[^a-zA-Z0-9._-]/g, "_");
      cb(null, `${Date.now()}-${safe}`);
    },
  }),
  limits: { fileSize: MAX_UPLOAD_MB * 1024 * 1024, files: 2 },
  fileFilter: (_req, file, cb) => {
    const ext = path.extname(file.originalname).toLowerCase();
    const okExt = [".mp4", ".webm", ".mov", ".avi", ".mkv"].includes(ext);
    if (ALLOWED.has(file.mimetype) || okExt) {
      cb(null, true);
      return;
    }
    cb(new Error("UNSUPPORTED_VIDEO"));
  },
}).fields([
  { name: "videoA", maxCount: 1 },
  { name: "videoB", maxCount: 1 },
]);
