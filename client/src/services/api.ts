import type { AnalysisResult, ApiError } from "../types/analysis";

export async function fetchHealth(): Promise<{
  status: string;
  pythonAvailable: boolean;
  torchAvailable: boolean;
  modelAvailable: boolean;
  device: string;
}> {
  const res = await fetch("/api/health/ready");
  if (!res.ok) throw new Error("Health check failed");
  return res.json();
}

export async function runAnalysis(videoA: File, videoB: File): Promise<AnalysisResult> {
  const body = new FormData();
  body.append("videoA", videoA);
  body.append("videoB", videoB);
  const res = await fetch("/api/analysis", { method: "POST", body });
  const data = await res.json();
  if (!res.ok) {
    const err = (data?.error ?? {}) as ApiError;
    const error: ApiError = {
      code: err.code || "REQUEST_FAILURE",
      message: err.message || "Analysis request failed.",
      subject: err.subject,
    };
    throw error;
  }
  return data as AnalysisResult;
}
