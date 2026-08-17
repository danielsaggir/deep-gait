import express from "express";
import cors from "cors";
import { PORT } from "./config/env.js";
import { errorHandler } from "./middleware/errors.js";
import { router } from "./routes/index.js";

const app = express();
app.use(cors());
app.use(express.json({ limit: "1mb" }));
app.use("/api", router);
app.use(errorHandler);

export { app };

if (process.env.VITEST !== "true") {
  app.listen(PORT, () => {
    console.error(`DeepGait API listening on ${PORT}`);
  });
}
