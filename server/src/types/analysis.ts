export type Verdict = "LIKELY_MATCH" | "LIKELY_DIFFERENT";

export type PoseJoint = {
  x: number;
  y: number;
  confidence?: number;
};

export type PoseFrame = {
  timestamp: number;
  detected?: boolean;
  joints: PoseJoint[];
};

export type AnalysisSuccess = {
  result: {
    samePersonProbability: number;
    cosineSimilarity: number;
    threshold: number;
    verdict: Verdict;
  };
  subjectA: Record<string, unknown>;
  subjectB: Record<string, unknown>;
  model: Record<string, unknown>;
  timing: Record<string, unknown>;
};

export type AnalysisErrorBody = {
  error: {
    code: string;
    message: string;
    subject?: string;
  };
};
