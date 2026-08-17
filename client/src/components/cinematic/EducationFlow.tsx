import { useEffect, useRef, useState, type CSSProperties, type ReactNode } from "react";
import { audio } from "../../audio/engine";
import type { AnalysisResult } from "../../types/analysis";

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

const STEP_MS = 7600;

function FrameStrip() {
  return (
    <svg viewBox="0 0 220 120" className="edu-diagram" aria-hidden="true">
      {[0, 1, 2, 3].map((i) => (
        <g key={i} style={{ "--i": i } as CSSProperties} className="edu-frame">
          <rect x={8 + i * 52} y="34" width="44" height="52" rx="1" />
          <circle cx={30 + i * 52} cy="52" r="6" />
          <path d={`M ${30 + i * 52} 58 L ${30 + i * 52} 74`} />
        </g>
      ))}
      <path className="edu-arrow" d="M 8 100 L 212 100" />
    </svg>
  );
}

function JointCloud() {
  const joints: Array<[number, number]> = [
    [110, 22], [98, 34], [122, 34], [88, 40], [132, 40],
    [80, 58], [140, 58], [70, 78], [150, 78], [64, 96], [156, 96],
    [94, 84], [126, 84], [90, 106], [130, 106], [88, 118], [132, 118],
  ];
  const bones: Array<[number, number]> = [
    [0, 1], [0, 2], [1, 3], [2, 4], [5, 6], [5, 7], [7, 9],
    [6, 8], [8, 10], [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
  ];
  return (
    <svg viewBox="0 0 220 140" className="edu-diagram" aria-hidden="true">
      {bones.map(([a, b], i) => (
        <line
          key={i}
          className="edu-bone"
          x1={joints[a][0]}
          y1={joints[a][1]}
          x2={joints[b][0]}
          y2={joints[b][1]}
          style={{ "--i": i } as CSSProperties}
        />
      ))}
      {joints.map(([x, y], i) => (
        <circle
          key={i}
          className="edu-joint"
          cx={x}
          cy={y}
          r="3"
          style={{ "--i": i } as CSSProperties}
        />
      ))}
    </svg>
  );
}

function CenteringDiagram() {
  return (
    <svg viewBox="0 0 220 140" className="edu-diagram" aria-hidden="true">
      <line className="edu-axis" x1="110" y1="10" x2="110" y2="130" />
      <line className="edu-axis" x1="20" y1="70" x2="200" y2="70" />
      <circle className="edu-origin" cx="110" cy="70" r="5" />
      <circle className="edu-orbit" cx="110" cy="70" r="34" pathLength={1000} />
      <circle className="edu-orbit slow" cx="110" cy="70" r="52" pathLength={1000} />
    </svg>
  );
}

function ChannelBars({ values }: { values: Array<[string, number]> }) {
  const max = Math.max(...values.map(([, v]) => v), 0.0001);
  return (
    <svg viewBox="0 0 220 140" className="edu-diagram" aria-hidden="true">
      {values.map(([label, v], i) => {
        const h = Math.max(4, (v / max) * 92);
        return (
          <g key={label} style={{ "--i": i } as CSSProperties} className="edu-bar">
            <rect x={14 + i * 40} y={116 - h} width="22" height={h} />
            <text x={25 + i * 40} y="132" textAnchor="middle">
              {label}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function EncodingDiagram() {
  return (
    <svg viewBox="0 0 220 140" className="edu-diagram" aria-hidden="true">
      {[0, 1, 2].map((layer) =>
        [0, 1, 2, 3].map((n) => (
          <circle
            key={`${layer}-${n}`}
            className="edu-node"
            cx={30 + layer * 46}
            cy={30 + n * 28}
            r="4"
            style={{ "--i": layer * 4 + n } as CSSProperties}
          />
        ))
      )}
      {[0, 1].map((layer) =>
        [0, 1, 2, 3].map((from) =>
          [0, 1, 2, 3].map((to) => (
            <line
              key={`${layer}-${from}-${to}`}
              className="edu-edge"
              x1={30 + layer * 46}
              y1={30 + from * 28}
              x2={76 + layer * 46}
              y2={30 + to * 28}
            />
          ))
        )
      )}
      {Array.from({ length: 14 }).map((_, i) => (
        <rect
          key={i}
          className="edu-vector"
          x={172}
          y={16 + i * 8}
          width="34"
          height="5"
          style={{ "--i": i } as CSSProperties}
        />
      ))}
    </svg>
  );
}

function CosineDiagram({ cosine }: { cosine: number }) {
  const angle = Math.acos(Math.max(-1, Math.min(1, cosine)));
  const len = 78;
  const cx = 60;
  const cy = 110;
  return (
    <svg viewBox="0 0 220 140" className="edu-diagram" aria-hidden="true">
      <line className="edu-vector-a" x1={cx} y1={cy} x2={cx + len} y2={cy} />
      <line
        className="edu-vector-b"
        x1={cx}
        y1={cy}
        x2={cx + len * Math.cos(angle)}
        y2={cy - len * Math.sin(angle)}
      />
      <path
        className="edu-angle"
        d={`M ${cx + 26} ${cy} A 26 26 0 0 0 ${cx + 26 * Math.cos(angle)} ${cy - 26 * Math.sin(angle)}`}
      />
      <text className="edu-figure" x={cx + 54} y={cy - 26}>
        {cosine.toFixed(3)}
      </text>
    </svg>
  );
}

function DecisionDiagram({ probability, threshold }: { probability: number; threshold: number }) {
  return (
    <svg viewBox="0 0 220 140" className="edu-diagram" aria-hidden="true">
      <rect className="edu-track" x="14" y="60" width="192" height="14" />
      <rect
        className="edu-fill"
        x="14"
        y="60"
        width={192 * probability}
        height="14"
        style={{ "--target": `${192 * probability}px` } as CSSProperties}
      />
      <line
        className="edu-threshold"
        x1={14 + 192 * threshold}
        y1="48"
        x2={14 + 192 * threshold}
        y2="86"
      />
      <text className="edu-figure" x={14} y="40">
        0%
      </text>
      <text className="edu-figure" x={190} y="40">
        100%
      </text>
      <text className="edu-figure mark" x={14 + 192 * threshold} y="104" textAnchor="middle">
        THRESHOLD
      </text>
    </svg>
  );
}

function buildSteps(a: AnalysisResult): Step[] {
  const pct = (v: number) => `${(v * 100).toFixed(0)}%`;
  const comp = a.subjectA.featureComposition;

  return [
    {
      title: "FOOTAGE INGESTION",
      body:
        "Both clips are decoded and resampled to a common frame rate so the two subjects are measured on the same time base. Gait is a rhythm, so comparing footage recorded at different speeds would compare different rhythms.",
      figures: [
        ["FRAMES SAMPLED A", String(a.subjectA.poseQuality.framesSampled)],
        ["FRAMES SAMPLED B", String(a.subjectB.poseQuality.framesSampled)],
      ],
      diagram: <FrameStrip />,
    },
    {
      title: "POSE ACQUISITION",
      body:
        "A pose estimator locates 17 body joints in every frame, following the COCO convention the model was trained on. Appearance, clothing and background are discarded here: from this point the system only sees moving points.",
      figures: [
        ["JOINTS PER FRAME", String(a.model.joints)],
        ["POSE COVERAGE", `${pct(a.subjectA.poseQuality.coverage)} / ${pct(a.subjectB.poseQuality.coverage)}`],
      ],
      diagram: <JointCloud />,
    },
    {
      title: "SKELETON NORMALIZATION",
      body:
        "Every skeleton is re-centred on the pelvis and rescaled. This removes where the person stood and how close they were to the camera, leaving only the shape and motion of the body itself.",
      figures: [
        ["ORIGIN", "PELVIS"],
        ["SEQUENCE WINDOW", `${a.model.sequenceLength} FRAMES`],
      ],
      diagram: <CenteringDiagram />,
    },
    {
      title: "TEMPORAL FEATURE EXTRACTION",
      body:
        "Each joint is expanded into 8 channels: position, joint angles, body proportions, velocity and acceleration. Velocity and acceleration are what make this gait analysis rather than posture analysis.",
      figures: [
        ["INPUT CHANNELS", String(a.model.inputChannels)],
        ["TENSOR", `${a.model.inputChannels} × ${a.model.sequenceLength} × ${a.model.joints}`],
      ],
      diagram: (
        <ChannelBars
          values={[
            ["POS", comp.position],
            ["ANG", comp.angles],
            ["PRP", comp.proportions],
            ["VEL", comp.velocity],
            ["ACC", comp.acceleration],
          ]}
        />
      ),
    },
    {
      title: "GAIT ENCODING",
      body:
        "A spatio-temporal graph convolutional network treats the skeleton as a graph across time. It compresses the whole sequence into a single 128-dimensional signature describing how this person walks.",
      figures: [
        ["ARCHITECTURE", a.model.architecture],
        ["EMBEDDING", `${a.model.embeddingDimension}-D`],
      ],
      diagram: <EncodingDiagram />,
    },
    {
      title: "BIOMETRIC COMPARISON",
      body:
        "Both signatures are compared in that 128-dimensional space. Cosine similarity measures the angle between them, which is useful supporting evidence but is not what decides the outcome.",
      figures: [
        ["COSINE SIMILARITY", a.result.cosineSimilarity.toFixed(3)],
        ["COMPUTE DEVICE", a.model.device.toUpperCase()],
      ],
      diagram: <CosineDiagram cosine={a.result.cosineSimilarity} />,
    },
    {
      title: "CLASSIFICATION",
      body:
        "A trained classifier reads both signatures together and returns the probability that they came from the same person. Above the threshold it reports a likely match. This is a probability, not proof of identity.",
      figures: [
        ["MATCH PROBABILITY", pct(a.result.samePersonProbability)],
        ["VERDICT", a.result.verdict.replace("_", " ")],
      ],
      diagram: (
        <DecisionDiagram
          probability={a.result.samePersonProbability}
          threshold={a.result.threshold}
        />
      ),
    },
  ];
}

export function EducationFlow({ analysis, soundEnabled = true, onClose }: Props) {
  const steps = buildSteps(analysis);
  const [index, setIndex] = useState(0);
  const [paused, setPaused] = useState(false);
  const indexRef = useRef(0);

  useEffect(() => {
    if (index === indexRef.current) return;
    indexRef.current = index;
    if (soundEnabled) audio.debriefStep();
  }, [index, soundEnabled]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      if (e.key === "ArrowRight") setIndex((i) => Math.min(i + 1, steps.length - 1));
      if (e.key === "ArrowLeft") setIndex((i) => Math.max(i - 1, 0));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose, steps.length]);

  useEffect(() => {
    if (paused) return;
    const timer = window.setTimeout(() => {
      setIndex((i) => (i < steps.length - 1 ? i + 1 : i));
    }, STEP_MS);
    return () => window.clearTimeout(timer);
  }, [index, paused, steps.length]);

  const step = steps[index];
  const last = index === steps.length - 1;

  return (
    <div
      className="edu"
      role="dialog"
      aria-label="Pipeline debrief"
      onMouseEnter={() => setPaused(true)}
      onMouseLeave={() => setPaused(false)}
    >
      <div className="edu-head">
        <span className="edu-kicker">PIPELINE DEBRIEF</span>
        <span className="edu-count">
          {String(index + 1).padStart(2, "0")} / {String(steps.length).padStart(2, "0")}
        </span>
        <button type="button" className="ghost" onClick={onClose}>
          CLOSE
        </button>
      </div>

      <div className="edu-body">
        <ol className="edu-rail">
          {steps.map((s, i) => (
            <li
              key={s.title}
              className={i === index ? "is-active" : i < index ? "is-done" : ""}
              onClick={() => setIndex(i)}
            >
              <span className="edu-rail-node" />
              {s.title}
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
          <span
            className={paused ? "is-paused" : ""}
            style={{ animationDuration: `${STEP_MS}ms` }}
            key={`${index}-${paused}`}
          />
        </div>
        <div className="edu-controls">
          <button
            type="button"
            className="ghost"
            disabled={index === 0}
            onClick={() => setIndex((i) => Math.max(i - 1, 0))}
          >
            PREV
          </button>
          {last ? (
            <button type="button" className="ghost active" onClick={onClose}>
              DONE
            </button>
          ) : (
            <button
              type="button"
              className="ghost active"
              onClick={() => setIndex((i) => Math.min(i + 1, steps.length - 1))}
            >
              NEXT
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
