export const STAGES = [
  "Decoding clips",
  "Extracting pose",
  "Normalising skeletons",
  "Building channels",
  "Encoding signatures",
  "Fusing the pair",
  "Scoring",
] as const;

export type Phase =
  | "READY"
  | "PARTIAL_UPLOAD"
  | "READY_TO_ANALYZE"
  | "ANALYZING"
  | "RESULT"
  | "ERROR";

export type StageStatus = "WAITING" | "PROCESSING" | "COMPLETE" | "FAILED";

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

export type VideoMetadata = {
  source: string;
  duration: number;
  width: number;
  height: number;
  format: string;
};

export type SubjectPayload = {
  metadata: {
    source: string;
    duration: number;
    width: number;
    height: number;
    fps: number;
    format: string;
  };
  poseQuality: {
    framesDetected: number;
    framesUsed: number;
    framesSampled: number;
    coverage: number;
  };
  poseFrames: PoseFrame[];
  gaitSignature: {
    velocityMagnitude: number[];
    lowerBodyMotion: number[];
  };
  featureComposition: {
    position: number;
    angles: number;
    proportions: number;
    velocity: number;
    acceleration: number;
  };
  embedding: number[];
  skeletonEdges: Array<[number, number]>;
};

export type AnalysisResult = {
  result: {
    samePersonProbability: number;
    cosineSimilarity: number;
    threshold: number;
    verdict: "LIKELY_MATCH" | "LIKELY_DIFFERENT";
  };
  subjectA: SubjectPayload;
  subjectB: SubjectPayload;
  model: {
    architecture: string;
    embeddingDimension: number;
    inputChannels: number;
    sequenceLength: number;
    joints: number;
    device: string;
  };
  timing: {
    poseExtraction: number;
    preprocessing: number;
    inference: number;
    total: number;
  };
};

export type ApiError = {
  code: string;
  message: string;
  subject?: string;
};

export type SubjectSlot = {
  file: File | null;
  objectUrl: string | null;
  metadata: VideoMetadata | null;
};

export type WorkstationState = {
  phase: Phase;
  subjectA: SubjectSlot;
  subjectB: SubjectSlot;
  analysis: AnalysisResult | null;
  error: ApiError | null;
  stageIndex: number;
  muted: boolean;
  overlayEnabled: boolean;
};

export type WorkstationAction =
  | { type: "SET_SUBJECT"; slot: "A" | "B"; file: File; objectUrl: string; metadata: VideoMetadata }
  | { type: "CLEAR_SUBJECT"; slot: "A" | "B" }
  | { type: "START_ANALYSIS" }
  | { type: "SET_STAGE"; index: number }
  | { type: "ANALYSIS_SUCCESS"; analysis: AnalysisResult }
  | { type: "ANALYSIS_ERROR"; error: ApiError }
  | { type: "RESET" }
  | { type: "TOGGLE_MUTE" }
  | { type: "TOGGLE_OVERLAY" };
