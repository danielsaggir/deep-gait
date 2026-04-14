# NPM scripts reference

Run these from the **repository root** unless you `cd` into a workspace package.

## Root `package.json`

| Script | What it does |
|--------|----------------|
| **`run:all`** | Full local stack: ensures MongoDB (Docker on `127.0.0.1:27017` unless `SKIP_MONGO_DOCKER=1`), checks `webapp/server/.env` exists, then runs **`dev:all`**. |
| **`dev:all`** | API + Vite together: runs `scripts/free-api-port.sh`, then starts the **server** with `npm run start:server` and the **client** with `npm run dev:client` (via `concurrently`). Ctrl+C stops both. |
| **`start:server`** | Express API only: `node src/index.js` in `webapp/server` (no file watching). |
| **`dev:server`** | API only with **`node --watch`** (restarts on file changes). |
| **`dev:client`** | React/Vite dev server only (`vite`). |
| **`build:client`** | Production build of the frontend (`vite build`). |

There is **no** plain `npm run dev` at the repo root. Use **`dev:all`** or **`run:all`**, or run `dev` inside `webapp/client` or `webapp/server`.

**Note:** `dev:all` uses **`start:server`**, not `dev:server`, so the API does not use `--watch` in that combined flow. Use `dev:server` separately if you want auto-restart while editing the server.

## Workspace packages (`webapp/server`, `webapp/client`)

| Package | Script | What it does |
|---------|--------|----------------|
| **deepgait-server** | `start` | `node src/index.js` |
| | `dev` | `node --watch src/index.js` |
| **deepgait-client** | `dev` | Vite dev server |
| | `build` | `vite build` |
| | `preview` | `vite preview` (serves production build locally) |

## Python (not npm)

From `pyproject.toml`:

- Optional **`dev`** extra: includes **pytest** for tests.
- Optional **`inference`** / **`train`** extras: **torch**, **opencv-python-headless**, **mediapipe** for ML workflows.

## Quick picks

| Goal | Command |
|------|---------|
| Full app + Mongo (typical local setup) | `npm run run:all` |
| API + client only (Mongo already running) | `npm run dev:all` |
| Frontend only | `npm run dev:client` |
| Backend only | `npm run dev:server` |
