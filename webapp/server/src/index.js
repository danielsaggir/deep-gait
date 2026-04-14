import "dotenv/config";
import express from "express";
import cors from "cors";
import mongoose from "mongoose";
import multer from "multer";
import path from "path";
import fs from "fs";
import { fileURLToPath } from "url";
import pino from "pino";
import { Suspect } from "./models/Suspect.js";
import { MatchResult } from "./models/MatchResult.js";
import { runPythonInference, resolveCheckpointPath } from "./inference.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const log = pino({
  transport: { target: "pino-pretty", options: { colorize: true } },
});

const app = express();
app.use(cors());
app.use(express.json());

const uploadsDir = path.join(__dirname, "..", "uploads");
fs.mkdirSync(uploadsDir, { recursive: true });

const storage = multer.diskStorage({
  destination: (_req, _file, cb) => cb(null, uploadsDir),
  filename: (_req, file, cb) => {
    const safe = `${Date.now()}-${file.originalname.replace(/[^a-zA-Z0-9._-]/g, "_")}`;
    cb(null, safe);
  },
});
const upload = multer({ storage });

const repoRoot = process.env.REPO_ROOT || path.resolve(__dirname, "..", "..", "..");
const checkpointFile = resolveCheckpointPath(
  repoRoot,
  process.env.CHECKPOINT_PATH
);
const envForInference = {
  REPO_ROOT: repoRoot,
  PYTHON_BIN: process.env.PYTHON_BIN || "python3",
  CHECKPOINT_PATH: process.env.CHECKPOINT_PATH,
};

function cosineSimilarity(a, b) {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  const d = Math.sqrt(na) * Math.sqrt(nb);
  return d === 0 ? 0 : dot / d;
}

app.get("/api/health", (_req, res) => {
  res.json({ ok: true, repoRoot });
});

app.get("/api/suspects", async (_req, res) => {
  try {
    const list = await Suspect.find().sort({ createdAt: -1 }).lean();
    res.json(list);
  } catch (e) {
    log.error(e);
    res.status(500).json({ error: String(e.message) });
  }
});

app.post("/api/suspects", upload.single("video"), async (req, res) => {
  try {
    const name = req.body?.name || "Unknown";
    if (!req.file) {
      res.status(400).json({ error: "missing video file field 'video'" });
      return;
    }
    const videoPath = req.file.path;
    log.info({ videoPath }, "running inference for new suspect");
    const signature = await runPythonInference(videoPath, envForInference);
    const doc = await Suspect.create({
      name,
      signature,
      video_path: videoPath,
    });
    res.status(201).json(doc);
  } catch (e) {
    log.error(e);
    res.status(500).json({ error: String(e.message) });
  }
});

app.post("/api/match", upload.single("video"), async (req, res) => {
  try {
    if (!req.file) {
      res.status(400).json({ error: "missing video file field 'video'" });
      return;
    }
    const videoPath = req.file.path;
    log.info({ videoPath }, "match query inference");
    const querySig = await runPythonInference(videoPath, envForInference);

    const suspects = await Suspect.find().lean();
    const scores = suspects.map((s) => ({
      suspectId: s._id,
      name: s.name,
      similarity: cosineSimilarity(querySig, s.signature),
    }));
    scores.sort((a, b) => b.similarity - a.similarity);
    const top = scores[0];

    const mr = await MatchResult.create({
      query_video_path: videoPath,
      scores: scores.map((x) => ({
        suspectId: x.suspectId,
        name: x.name,
        similarity: x.similarity,
      })),
      top_suspect: top?.suspectId,
    });

    res.json({
      matchResult: mr,
      ranked: scores,
    });
  } catch (e) {
    log.error(e);
    res.status(500).json({ error: String(e.message) });
  }
});

const port = Number(process.env.PORT || 3001);
const mongoUri = process.env.MONGODB_URI || "mongodb://127.0.0.1:27017/deepgait";

mongoose
  .connect(mongoUri)
  .then(() => {
    log.info("MongoDB connected");
    log.info(
      {
        checkpoint: checkpointFile,
        inferenceOnly: true,
        exists: fs.existsSync(checkpointFile),
      },
      "loads saved weights for inference only (training is never started by this server)"
    );
    app.listen(port, () => log.info({ port }, "DeepGait API listening"));
  })
  .catch((err) => {
    log.error(err);
    process.exit(1);
  });
