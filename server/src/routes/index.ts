import { Router } from "express";
import { analyze, attachTempDir } from "../controllers/analysisController.js";
import { health, healthReady } from "../controllers/healthController.js";
import { uploadPair } from "../middleware/upload.js";

export const router = Router();

router.get("/health", health);
router.get("/health/ready", healthReady);
router.post("/analysis", attachTempDir, uploadPair, analyze);
