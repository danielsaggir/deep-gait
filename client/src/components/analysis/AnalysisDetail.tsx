import { useRef, useState, type CSSProperties, type ReactNode } from "react";
import type { AnalysisResult } from "../../types/analysis";
import { EmbeddingCompare } from "../charts/EmbeddingCompare";
import { FeatureComposition } from "../charts/FeatureComposition";
import { GaitSignature } from "../charts/GaitSignature";
import { FitToHeight } from "./FitToHeight";

type Props = {
  analysis: AnalysisResult;
  overlayEnabled: boolean;
  onToggleOverlay: () => void;
};

const TABS = [
  { id: "readout", label: "Readout", hint: "Scores and capture quality" },
  { id: "motion", label: "Motion", hint: "Per-frame velocity through the 64-frame window" },
  { id: "fingerprint", label: "Fingerprint", hint: "The 128-D signatures and what fed them" },
] as const;

type TabId = (typeof TABS)[number]["id"];

type Metric = { label: string; value: ReactNode; note: string };

function metricsFor(analysis: AnalysisResult): Metric[] {
  const coverage = (s: "subjectA" | "subjectB") =>
    `${(analysis[s].poseQuality.coverage * 100).toFixed(0)}%`;

  return [
    {
      label: "Cosine similarity",
      value: analysis.result.cosineSimilarity.toFixed(3),
      note: "Angle between the two signatures, −1 to 1",
    },
    {
      label: "Decision threshold",
      value: analysis.result.threshold.toFixed(3),
      note: "Above this the pair is called a match",
    },
    {
      label: "Pose coverage",
      value: (
        <span className="metric-pair">
          <span className="tone-a">{coverage("subjectA")}</span>
          <span className="metric-pair-sep" aria-hidden="true">
            /
          </span>
          <span className="tone-b">{coverage("subjectB")}</span>
        </span>
      ),
      note: "Frames with a usable skeleton, A / B",
    },
    {
      label: "Sequence window",
      value: `${analysis.model.sequenceLength} frames`,
      note: "Fixed window each clip is sampled onto",
    },
    {
      label: "Joints tracked",
      value: String(analysis.model.joints),
      note: "Kept from the 33 BlazePose returns",
    },
    {
      label: "Time taken",
      value: `${analysis.timing.total.toFixed(2)}s`,
      note: "Pose extraction plus inference",
    },
  ];
}

const STAGES = [
  { key: "poseExtraction", label: "Pose extraction", cls: "pose" },
  { key: "preprocessing", label: "Preprocessing", cls: "prep" },
  { key: "inference", label: "Inference", cls: "infer" },
] as const;

/** Where the wall-clock time actually went, as a proportional bar rather than
    a bare list of numbers — pose extraction dwarfs the other two stages on
    CPU, and the bar makes that legible at a glance. */
function TimingBreakdown({
  timing,
  model,
}: {
  timing: AnalysisResult["timing"];
  model: AnalysisResult["model"];
}) {
  const stageValue = (key: (typeof STAGES)[number]["key"]) => timing[key] ?? 0;
  const total = Math.max(timing.total ?? 0, 0.001);
  const pct = (v: number) => Math.max(0, (v / total) * 100);

  return (
    <div className="timing">
      <div className="timing-head">
        <h3>Pipeline timing</h3>
        <span className="timing-total">{total.toFixed(2)}s total</span>
      </div>

      <div
        className="timing-bar"
        role="img"
        aria-label={STAGES.map((s) => `${s.label} ${stageValue(s.key).toFixed(2)} seconds`).join(
          ", "
        )}
      >
        {STAGES.map((s) => (
          <span
            key={s.key}
            className={`timing-seg timing-seg-${s.cls}`}
            style={{ "--w": pct(stageValue(s.key)) } as CSSProperties}
          />
        ))}
      </div>

      <div className="timing-legend">
        {STAGES.map((s) => (
          <span key={s.key}>
            <i className={`timing-dot timing-dot-${s.cls}`} aria-hidden="true" />
            {s.label} <b>{stageValue(s.key).toFixed(2)}s</b>
          </span>
        ))}
      </div>

      <dl className="model-strip">
        <div>
          <dt>Architecture</dt>
          <dd>{model.architecture}</dd>
        </div>
        <div>
          <dt>Channels</dt>
          <dd>
            {model.inputChannels} → {model.embeddingDimension ?? "?"}-D
          </dd>
        </div>
        <div>
          <dt>Device</dt>
          <dd>{(model.device ?? "unknown").toUpperCase()}</dd>
        </div>
      </dl>
    </div>
  );
}

export function AnalysisDetail({ analysis, overlayEnabled, onToggleOverlay }: Props) {
  const [tab, setTab] = useState<TabId>("readout");
  const tabsRef = useRef<HTMLDivElement | null>(null);

  const onTabKey = (e: React.KeyboardEvent) => {
    const i = TABS.findIndex((t) => t.id === tab);
    let target: number;
    if (e.key === "ArrowRight") target = (i + 1) % TABS.length;
    else if (e.key === "ArrowLeft") target = (i - 1 + TABS.length) % TABS.length;
    else if (e.key === "Home") target = 0;
    else if (e.key === "End") target = TABS.length - 1;
    else return;

    e.preventDefault();
    const next = TABS[target];
    setTab(next.id);
    tabsRef.current?.querySelector<HTMLButtonElement>(`#tab-${next.id}`)?.focus();
  };

  const active = TABS.findIndex((t) => t.id === tab);

  return (
    <section className="detail" aria-label="Analysis detail">
      <header className="detail-head">
        <div className="detail-title">
          <h2>Analysis detail</h2>
          <p>{TABS[active].hint}</p>
        </div>

        {/* The overlay applies to the clips above, not to any one tab, so it
            lives beside the tabs instead of inside the readout panel. */}
        <button
          type="button"
          className="btn btn-quiet btn-sm"
          onClick={onToggleOverlay}
          aria-pressed={overlayEnabled}
        >
          <span className={`btn-dot ${overlayEnabled ? "is-live" : ""}`} aria-hidden="true" />
          Skeleton overlay
        </button>

        <div
          className="tabs"
          role="tablist"
          aria-label="Analysis detail sections"
          ref={tabsRef}
          onKeyDown={onTabKey}
          style={{ "--active": active, "--count": TABS.length } as CSSProperties}
        >
          <span className="tab-thumb" aria-hidden="true" />
          {TABS.map((t) => (
            <button
              key={t.id}
              id={`tab-${t.id}`}
              type="button"
              role="tab"
              className="tab"
              aria-selected={tab === t.id}
              aria-controls={`panel-${t.id}`}
              tabIndex={tab === t.id ? 0 : -1}
              onClick={() => setTab(t.id)}
            >
              {t.label}
            </button>
          ))}
        </div>
      </header>

      {tab === "readout" ? (
        <div
          className="detail-panel"
          id="panel-readout"
          role="tabpanel"
          aria-labelledby="tab-readout"
          tabIndex={0}
        >
          <div className="metrics">
            {metricsFor(analysis).map((m, i) => (
              <div className="metric" key={m.label} style={{ "--i": i } as CSSProperties}>
                <span className="metric-label">{m.label}</span>
                <b>{m.value}</b>
                <span className="metric-note">{m.note}</span>
              </div>
            ))}
          </div>

          <TimingBreakdown timing={analysis.timing} model={analysis.model} />
        </div>
      ) : null}

      {/* No panel holds focusable content of its own, so each takes a tab stop
          or a keyboard user skips straight past the charts. Motion and
          fingerprint scale their charts to the panel's actual height instead
          of scrolling — see FitToHeight. */}
      {tab === "motion" ? (
        <div
          className="detail-panel detail-panel-fit"
          id="panel-motion"
          role="tabpanel"
          aria-labelledby="tab-motion"
          tabIndex={0}
        >
          <FitToHeight>
            <GaitSignature
              a={analysis.subjectA.gaitSignature.velocityMagnitude}
              b={analysis.subjectB.gaitSignature.velocityMagnitude}
              subA={analysis.subjectA.gaitSignature.lowerBodyMotion}
              subB={analysis.subjectB.gaitSignature.lowerBodyMotion}
            />
          </FitToHeight>
        </div>
      ) : null}

      {tab === "fingerprint" ? (
        <div
          className="detail-panel detail-panel-fit"
          id="panel-fingerprint"
          role="tabpanel"
          aria-labelledby="tab-fingerprint"
          tabIndex={0}
        >
          <FitToHeight>
            <div className="detail-split">
              <EmbeddingCompare a={analysis.subjectA.embedding} b={analysis.subjectB.embedding} />
              <FeatureComposition values={analysis.subjectA.featureComposition} />
            </div>
          </FitToHeight>
        </div>
      ) : null}
    </section>
  );
}
