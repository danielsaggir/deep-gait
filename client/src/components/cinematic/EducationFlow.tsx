import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { audio } from "../../audio/engine";
import type { AnalysisResult } from "../../types/analysis";
import {
  AdjacencyMatrix,
  CenteringDiagram,
  ChannelBars,
  CosineDiagram,
  DecisionCurve,
  DecisionDiagram,
  DimensionDelta,
  EmbeddingFingerprint,
  EncodingDiagram,
  FrameStrip,
  JointCloud,
  MotionTrace,
} from "./EduDiagrams";

type Props = {
  analysis: AnalysisResult;
  soundEnabled?: boolean;
  onClose: () => void;
};

type Step = {
  title: string;
  body: string;
  figures: Array<[string, string]>;
  diagram: ReactNode;
};

const STEP_MS = 9000;

function buildSteps(a: AnalysisResult): Step[] {
  const pct = (v: number) => `${(v * 100).toFixed(0)}%`;
  const comp = a.subjectA.featureComposition;
  const velA = a.subjectA.gaitSignature?.velocityMagnitude ?? [];
  const velB = a.subjectB.gaitSignature?.velocityMagnitude ?? [];
  const embA = a.subjectA.embedding ?? [];
  const embB = a.subjectB.embedding ?? [];
  const edges = a.subjectA.skeletonEdges ?? [];

  const steps: Step[] = [
    {
      title: "Reading the footage",
      body: "Both clips are decoded and sampled onto the same 64-frame window. Gait is a rhythm, so two recordings have to be measured on a common time base before their rhythms can be compared at all.",
      figures: [
        ["Frames sampled (A)", String(a.subjectA.poseQuality.framesSampled)],
        ["Frames sampled (B)", String(a.subjectB.poseQuality.framesSampled)],
      ],
      diagram: <FrameStrip />,
    },
    {
      title: "Finding the body",
      body: "YOLO11-pose locates the body in every frame and returns 17 COCO joints. Fingers and facial detail were never tracked in the first place. From here the system never sees pixels again — no clothing, no background, no face.",
      figures: [
        ["Joints tracked", `${a.model.joints}`],
        [
          "Pose coverage",
          `${pct(a.subjectA.poseQuality.coverage)} / ${pct(a.subjectB.poseQuality.coverage)}`,
        ],
      ],
      diagram: <JointCloud />,
    },
    {
      title: "Normalising the skeleton",
      body: "Every skeleton is re-centred on the pelvis and rescaled to a standard body size. This throws away where the person stood and how close they were to the camera, so a walker filmed from ten metres matches the same walker filmed from three.",
      figures: [
        ["Origin", "Pelvis"],
        ["Sequence window", `${a.model.sequenceLength} frames`],
      ],
      diagram: <CenteringDiagram />,
    },
  ];

  if (velA.length > 1 || velB.length > 1) {
    steps.push({
      title: "Measuring the motion",
      body: "Joint velocity is tracked across the whole window. These are the real traces from your two clips. Each is scaled to its own range, because what identifies a walker is where the peaks fall and how regularly they repeat, not how tall they are.",
      figures: [
        ["Frames plotted", `${velA.length} / ${velB.length}`],
        ["Peak velocity", `${Math.max(...velA, 0).toFixed(2)} / ${Math.max(...velB, 0).toFixed(2)}`],
      ],
      diagram: <MotionTrace a={velA} b={velB} />,
    });
  }

  steps.push({
    title: "Eight channels per joint",
    body: "Each joint carries 8 numbers: 2 for its position, 1 for the angle at that joint, 1 for its distance from the torso, and 4 for velocity and acceleration. The last four are what make this gait analysis rather than posture analysis — they describe movement, not a pose.",
    figures: [
      ["Input channels", String(a.model.inputChannels)],
      ["Tensor", `${a.model.inputChannels} × ${a.model.sequenceLength} × ${a.model.joints}`],
    ],
    diagram: (
      <ChannelBars
        values={[
          ["Pos", comp.position],
          ["Ang", comp.angles],
          ["Prp", comp.proportions],
          ["Vel", comp.velocity],
          ["Acc", comp.acceleration],
        ]}
      />
    ),
  });

  if (edges.length) {
    steps.push({
      title: "The skeleton as a graph",
      body: "The body is a graph: joints are vertices, bones are edges. This adjacency matrix is hand-built from anatomy rather than learned — undirected, with every joint also linked to itself, then row-normalised. Handing the network the skeleton instead of making it discover one was worth about five points of accuracy.",
      figures: [
        ["Vertices", String(a.model.joints)],
        ["Edges", String(edges.length)],
      ],
      diagram: <AdjacencyMatrix edges={edges} joints={a.model.joints} />,
    });
  }

  steps.push({
    title: "Encoding the gait signature",
    body: "Three spatio-temporal graph convolution blocks widen the 8 channels to 256, alternating between passing information along the bones and along time. Pooling then crushes the whole sequence into one 128-number signature of how this person walks.",
    figures: [
      ["Architecture", a.model.architecture],
      ["Channel path", "8 → 64 → 128 → 256"],
    ],
    diagram: <EncodingDiagram />,
  });

  steps.push({
    title: "Twin encoders, not a classifier",
    body: "Both clips go through the same encoder with the same weights, and the network is only ever asked whether two signatures match. An earlier version tried to name people instead, and it simply memorised the training subjects. Comparing rather than naming is what lets this work on someone it has never seen.",
    figures: [
      ["Task", "Verification"],
      ["Embedding", `${a.model.embeddingDimension}-D, L2-normalised`],
    ],
    diagram: <EncodingDiagram />,
  });

  if (embA.length && embB.length) {
    steps.push({
      title: "Two fingerprints",
      body: "These are the actual signatures produced for your clips, mirrored around a shared axis — Video A above, Video B below. Similar walkers produce visibly similar silhouettes. The model never compares pixels, only these numbers.",
      figures: [
        ["Dimensions", String(Math.min(embA.length, embB.length))],
        ["Compute device", a.model.device.toUpperCase()],
      ],
      diagram: <EmbeddingFingerprint a={embA} b={embB} />,
    });

    steps.push({
      title: "Where they differ",
      body: "Subtracting one signature from the other shows which dimensions disagree. A handful of tall spikes usually means one specific movement trait differs; a uniformly high field means the two walks have little in common at all.",
      figures: [
        [
          "Mean difference",
          (
            embA
              .slice(0, Math.min(embA.length, embB.length))
              .reduce((s, v, i) => s + Math.abs(v - embB[i]), 0) /
            Math.min(embA.length, embB.length)
          ).toFixed(4),
        ],
        ["Cosine similarity", a.result.cosineSimilarity.toFixed(3)],
      ],
      diagram: <DimensionDelta a={embA} b={embB} />,
    });
  }

  steps.push({
    title: "Comparing the signatures",
    body: "The two signatures are combined three ways at once: the gap between them, their element-wise product, and the cosine of the angle separating them. Cosine similarity alone is useful evidence but is deliberately not what decides the outcome.",
    figures: [
      ["Cosine similarity", a.result.cosineSimilarity.toFixed(3)],
      ["Angle", `${((Math.acos(Math.max(-1, Math.min(1, a.result.cosineSimilarity))) * 180) / Math.PI).toFixed(1)}°`],
    ],
    diagram: <CosineDiagram cosine={a.result.cosineSimilarity} />,
  });

  steps.push({
    title: "From similarity to probability",
    body: "That combined comparison goes through a small classifier which emits a single number, squashed into a probability. The curve is steep near the threshold: small shifts in similarity swing the answer there, which is exactly why borderline results deserve less confidence than the percentage suggests.",
    figures: [
      ["Match probability", pct(a.result.samePersonProbability)],
      ["Threshold", pct(a.result.threshold)],
    ],
    diagram: (
      <DecisionCurve
        cosine={a.result.cosineSimilarity}
        probability={a.result.samePersonProbability}
        threshold={a.result.threshold}
      />
    ),
  });

  steps.push({
    title: "Making the call",
    body: "Above the threshold the system reports a likely match. Tested on twelve people it had never seen, it was right 86.9% of the time when the clips showed different people and 83.3% when they showed the same one. Camera angle, footwear and carried load all move the number, so treat one comparison as evidence, never as proof.",
    figures: [
      ["Match probability", pct(a.result.samePersonProbability)],
      ["Verdict", a.result.verdict.replace("_", " ").toLowerCase()],
    ],
    diagram: (
      <DecisionDiagram
        probability={a.result.samePersonProbability}
        threshold={a.result.threshold}
      />
    ),
  });

  return steps;
}

const FOCUSABLE = 'button:not([disabled]), [href], [tabindex]:not([tabindex="-1"])';

export function EducationFlow({ analysis, soundEnabled = true, onClose }: Props) {
  // Rebuilding every step (and every diagram element) on each tick of the
  // auto-advance timer throws away the whole subtree for no reason.
  const steps = useMemo(() => buildSteps(analysis), [analysis]);
  const [index, setIndex] = useState(0);
  const [paused, setPaused] = useState(false);
  const indexRef = useRef(0);
  const dialogRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (index === indexRef.current) return;
    indexRef.current = index;
    if (soundEnabled) audio.debriefStep();
  }, [index, soundEnabled]);

  // Take focus on open and hand it back to whatever opened us on close.
  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null;
    dialogRef.current?.focus();
    return () => opener?.focus?.();
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      if (e.key === "ArrowRight") setIndex((i) => Math.min(i + 1, steps.length - 1));
      if (e.key === "ArrowLeft") setIndex((i) => Math.max(i - 1, 0));
      if (e.key === " ") {
        // Without this, Space both toggles the pause and re-triggers whichever
        // control happens to hold focus.
        e.preventDefault();
        setPaused((p) => !p);
      }
      if (e.key !== "Tab") return;

      const nodes = Array.from(dialogRef.current?.querySelectorAll<HTMLElement>(FOCUSABLE) ?? []);
      if (!nodes.length) return;
      const first = nodes[0];
      const last = nodes[nodes.length - 1];
      const active = document.activeElement as HTMLElement | null;
      const inside = dialogRef.current?.contains(active) ?? false;

      if (!inside || (e.shiftKey && active === first)) {
        e.preventDefault();
        (e.shiftKey ? last : first).focus();
      } else if (!e.shiftKey && active === last) {
        e.preventDefault();
        first.focus();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose, steps.length]);

  // Pausing has to hold the remaining time rather than restart the step, so the
  // progress bar and the timer stay in agreement across a pause.
  const remaining = useRef(STEP_MS);
  const startedAt = useRef(0);

  useEffect(() => {
    remaining.current = STEP_MS;
  }, [index]);

  useEffect(() => {
    if (paused) {
      remaining.current = Math.max(0, remaining.current - (Date.now() - startedAt.current));
      return;
    }
    startedAt.current = Date.now();
    const timer = window.setTimeout(() => {
      setIndex((i) => (i < steps.length - 1 ? i + 1 : i));
    }, remaining.current);
    return () => window.clearTimeout(timer);
  }, [index, paused, steps.length]);

  const step = steps[index];
  const last = index === steps.length - 1;

  return (
    <div
      className="edu"
      role="dialog"
      aria-modal="true"
      aria-label="How this works"
      ref={dialogRef}
      tabIndex={-1}
    >
      <div className="edu-head">
        <span className="edu-kicker">How this works</span>
        <span className="edu-count">
          {String(index + 1).padStart(2, "0")} / {String(steps.length).padStart(2, "0")}
        </span>
        <button type="button" className="btn btn-quiet btn-sm" onClick={onClose}>
          Close
        </button>
      </div>

      <div className="edu-body">
        <ol className="edu-rail">
          {steps.map((s, i) => (
            <li key={s.title}>
              <button
                type="button"
                className={i === index ? "is-active" : i < index ? "is-done" : ""}
                aria-current={i === index ? "step" : undefined}
                onClick={() => setIndex(i)}
              >
                <span className="edu-rail-node" aria-hidden="true" />
                {s.title}
              </button>
            </li>
          ))}
        </ol>

        <div className="edu-stage" key={step.title}>
          <h2>{step.title}</h2>
          <p>{step.body}</p>
          <div className="edu-figures">
            {step.figures.map(([label, value]) => (
              <div key={label}>
                <label>{label}</label>
                <b>{value}</b>
              </div>
            ))}
          </div>
        </div>

        <div className="edu-visual" key={`${step.title}-vis`}>
          {step.diagram}
        </div>
      </div>

      <div className="edu-foot">
        <div className="edu-progress">
          {/* Keyed on the step alone: keying on `paused` too remounted the span
              and snapped the bar back to empty instead of holding position. */}
          <span
            className={paused ? "is-paused" : ""}
            style={{ animationDuration: `${STEP_MS}ms` }}
            key={index}
          />
        </div>
        <span className="edu-keys">Space to pause · ← → to step · Esc to close</span>

        <div className="edu-controls">
          <button
            type="button"
            className="btn btn-quiet"
            disabled={index === 0}
            onClick={() => setIndex((i) => Math.max(i - 1, 0))}
          >
            Back
          </button>
          {last ? (
            <button type="button" className="btn btn-secondary" onClick={onClose}>
              Done
            </button>
          ) : (
            <button
              type="button"
              className="btn btn-secondary"
              onClick={() => setIndex((i) => Math.min(i + 1, steps.length - 1))}
            >
              Next
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
