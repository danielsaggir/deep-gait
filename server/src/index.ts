import express from "express";
import cors from "cors";
import fs from "node:fs";
import path from "node:path";
import { PORT, REPO_ROOT } from "./config/env.js";
import { errorHandler } from "./middleware/errors.js";
import { router } from "./routes/index.js";

const app = express();
app.use(cors());
app.use(express.json({ limit: "1mb" }));
app.use("/api", router);

// Serves the built React SPA when present, so a single deployed process
// (e.g. one Docker web service on Render) can host both the API and the UI.
const clientDist = path.join(REPO_ROOT, "client", "dist");
if (fs.existsSync(clientDist)) {
  app.use(express.static(clientDist));
  app.get(/^\/(?!api\/).*/, (_req, res) => {
    res.sendFile(path.join(clientDist, "index.html"));
  });
}

app.use(errorHandler);

export { app };

if (process.env.VITEST !== "true") {
  app.listen(PORT, "0.0.0.0", () => {
    console.error(`DeepGait API listening on ${PORT}`);
  });
}
