import { useRef, useState } from "react";
import { HudFrame } from "../hud/HudFrame";
import type { PoseFrame, SubjectSlot, VideoMetadata } from "../../types/analysis";
import { SkeletonOverlay } from "./SkeletonOverlay";

type Props = {
  label: string;
  slot: SubjectSlot;
  poseFrames?: PoseFrame[];
  edges?: Array<[number, number]>;
  overlayEnabled: boolean;
  onSelect: (file: File, objectUrl: string, metadata: VideoMetadata) => void;
  onClear: () => void;
};

function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds - m * 60;
  return `${String(m).padStart(2, "0")}:${s.toFixed(2).padStart(5, "0")}`;
}

export function SubjectPanel({
  label,
  slot,
  poseFrames,
  edges,
  overlayEnabled,
  onSelect,
  onClear,
}: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [videoEl, setVideoEl] = useState<HTMLVideoElement | null>(null);
  const [dragging, setDragging] = useState(false);

  const tracking = Boolean(poseFrames && edges);

  const onFile = (file: File) => {
    const url = URL.createObjectURL(file);
    const probe = document.createElement("video");
    probe.preload = "metadata";
    probe.src = url;
    probe.onloadedmetadata = () => {
      onSelect(file, url, {
        source: file.name,
        duration: probe.duration || 0,
        width: probe.videoWidth,
        height: probe.videoHeight,
        format: (file.name.split(".").pop() || "VIDEO").toUpperCase(),
      });
    };
  };

  return (
    <HudFrame className="subject-panel" active={Boolean(slot.file)}>
      <div className="panel-head">
        <strong>{label}</strong>
        <span className={`status-led ${slot.file ? "is-on" : ""}`} />
        <span className="panel-state">
          {tracking ? "POSE TRACKED" : slot.file ? "FOOTAGE ACQUIRED" : "STANDBY"}
        </span>
      </div>

      <div className={`video-stage ${tracking && overlayEnabled ? "is-tracking" : ""}`}>
        {slot.objectUrl ? (
          <>
            <video ref={setVideoEl} src={slot.objectUrl} controls playsInline />
            {poseFrames && edges ? (
              <SkeletonOverlay
                video={videoEl}
                frames={poseFrames}
                edges={edges}
                enabled={overlayEnabled}
              />
            ) : null}
            <span className="stage-reticle" aria-hidden="true" />
          </>
        ) : (
          <label
            className={`dropzone ${dragging ? "is-dragging" : ""}`}
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragging(false);
              const file = e.dataTransfer.files?.[0];
              if (file) onFile(file);
            }}
          >
            <input
              ref={inputRef}
              type="file"
              accept="video/*"
              hidden
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) onFile(file);
              }}
            />
            <svg viewBox="0 0 120 120" className="dropzone-mark" aria-hidden="true">
              <circle cx="60" cy="60" r="44" />
              <circle cx="60" cy="60" r="30" className="dashed" />
              <line x1="60" y1="8" x2="60" y2="22" />
              <line x1="60" y1="98" x2="60" y2="112" />
              <line x1="8" y1="60" x2="22" y2="60" />
              <line x1="98" y1="60" x2="112" y2="60" />
            </svg>
            <b>ACQUIRE FOOTAGE</b>
            <span>DROP OR SELECT A WALKING SEQUENCE</span>
          </label>
        )}
      </div>

      {slot.metadata ? (
        <>
          <dl className="meta">
            <div>
              <dt>SOURCE</dt>
              <dd title={slot.metadata.source}>{slot.metadata.source}</dd>
            </div>
            <div>
              <dt>DURATION</dt>
              <dd>{formatDuration(slot.metadata.duration)}</dd>
            </div>
            <div>
              <dt>RESOLUTION</dt>
              <dd>
                {slot.metadata.width} × {slot.metadata.height}
              </dd>
            </div>
            <div>
              <dt>FORMAT</dt>
              <dd>{slot.metadata.format}</dd>
            </div>
          </dl>
          <div className="video-controls">
            <button type="button" className="ghost" onClick={() => inputRef.current?.click()}>
              REPLACE
            </button>
            <button type="button" className="ghost" onClick={onClear}>
              REMOVE
            </button>
            <input
              ref={inputRef}
              type="file"
              accept="video/*"
              hidden
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) onFile(file);
              }}
            />
          </div>
        </>
      ) : null}
    </HudFrame>
  );
}
