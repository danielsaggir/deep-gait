import fs from "node:fs";
import os from "node:os";
import path from "node:path";

export function makeTempDir(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), "deepgait-"));
}

export function removeDir(dir: string | undefined): void {
  if (!dir) return;
  fs.rmSync(dir, { recursive: true, force: true });
}
