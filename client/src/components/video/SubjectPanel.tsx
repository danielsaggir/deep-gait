import { useEffect, useId, useRef, useState } from "react";
import { audio } from "../../audio/engine";
import { HudFrame } from "../hud/HudFrame";
import type { PoseFrame, SubjectSlot, VideoMetadata } from "../../types/analysis";
import { SkeletonOverlay } from "./SkeletonOverlay";

type Props = {
  label: string;
  /** Colour identity shared with every chart that plots this subject. */
  tone: "a" | "b";
  slot: SubjectSlot;
  poseFrames?: PoseFrame[];
  edges?: Array<[number, number]>;
  overlayEnabled: boolean;
  /** Drives the ingest sweep and starts playback while the model reads. */
  scanning: boolean;
  /** Restarts both clips together once a verdict lands. */
  playing: boolean;
  onSelect: (file: File, objectUrl: string, metadata: VideoMetadata) => void;
  onClear: () => void;
};

function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds - m * 60;
  return `${String(m).padStart(2, "0")}:${s.toFixed(1).padStart(4, "0")}`;
}

function statusLabel(opts: {
  playback: boolean;
  scanning: boolean;
  tracked: boolean;
  hasFile: boolean;
}): string {
  if (opts.playback) return "Skeleton tracked";
  if (opts.scanning) return "Extracting pose";
  if (opts.tracked) return "Pose tracked";
  if (opts.hasFile) return "Loaded";
  return "Empty";
}

export function SubjectPanel({
  label,
  tone,
  slot,
  poseFrames,
  edges,
  overlayEnabled,
  scanning,
  playing,
  onSelect,
  onClear,
}: Props) {
  const inputId = useId();
  const inputRef = useRef<HTMLInputElement>(null);
  const [videoEl, setVideoEl] = useState<HTMLVideoElement | null>(null);
  const [dragging, setDragging] = useState(false);
  const [rejected, setRejected] = useState<string | null>(null);

  const tracking = Boolean(poseFrames && edges);
  const skeletonOn = tracking && overlayEnabled;
  const playback = playing && skeletonOn;

  // Playback starts with the run, not with the upload: both clips roll from
  // frame zero the moment the comparison begins, and again when it resolves.
  useEffect(() => {
    if (!videoEl || !(scanning || playing)) return;
    videoEl.currentTime = 0;
    void videoEl.play().catch(() => undefined);
  }, [scanning, playing, videoEl]);

  const onFile = (file: File) => {
    const url = URL.createObjectURL(file);
    const probe = document.createElement("video");
    probe.preload = "metadata";
    probe.src = url;

    probe.onloadedmetadata = () => {
      probe.onerror = null;
      onSelect(file, url, {
        source: file.name,
        duration: probe.duration || 0,
        width: probe.videoWidth,
        height: probe.videoHeight,
        format: (file.name.split(".").pop() || "video").toUpperCase(),
      });
    };

    // Without this a codec the browser cannot decode (an iPhone HEVC .mov is
    // the common one) drops the file on the floor: no slot, no message, and a
    // leaked object URL holding a decoder open.
    probe.onerror = () => {
      URL.revokeObjectURL(url);
      probe.removeAttribute("src");
      probe.load();
      setRejected(file.name);
    };
  };

  const pickFile = (file: File | undefined) => {
    if (!file) return;
    audio.resume();
    setRejected(null);
    onFile(file);
  };

  return (
    <HudFrame
      className={`subject-panel tone-${tone} ${playback ? "is-playback" : ""}`}
      active={Boolean(slot.file)}
    >
      <div className="panel-head">
        <strong className="panel-tag">{label}</strong>
        <span className={`status-led ${slot.file ? "is-on" : ""}`} aria-hidden="true" />
        {slot.metadata ? (
          <span className="panel-file" title={slot.metadata.source}>
            {slot.metadata.source}
          </span>
        ) : null}
        <span className="panel-state">
          {statusLabel({ playback, scanning, tracked: skeletonOn, hasFile: Boolean(slot.file) })}
        </span>
      </div>

      <div className={`video-stage ${skeletonOn ? "is-tracking" : ""}`}>
        {slot.objectUrl ? (
          <>
            <video ref={setVideoEl} src={slot.objectUrl} controls playsInline muted loop />
            {poseFrames && edges ? (
              <SkeletonOverlay
                video={videoEl}
                frames={poseFrames}
                edges={edges}
                enabled={overlayEnabled}
                tone={tone}
              />
            ) : null}
            <span className="stage-reticle" aria-hidden="true" />
            {scanning ? <span className="ingest-sweep" aria-hidden="true" /> : null}
          </>
        ) : (
          <label
            className={`dropzone ${dragging ? "is-dragging" : ""}`}
            htmlFor={inputId}
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragging(false);
              pickFile(e.dataTransfer.files?.[0]);
            }}
          >
            <svg viewBox="0 0 48 48" className="dropzone-mark" aria-hidden="true">
              <rect x="6" y="12" width="28" height="24" rx="4" />
              <path d="M34 22 L42 17 V31 L34 26 Z" />
              <path className="dropzone-plus" d="M20 20 V28 M16 24 H24" />
            </svg>
            <b>Drop a clip here</b>
            {rejected ? (
              <span className="dropzone-error" role="alert">
                This browser could not read {rejected}. Try an MP4 encoded with H.264.
              </span>
            ) : (
              <span>or click to browse · a few seconds of walking, filmed from the side</span>
            )}
          </label>
        )}
      </div>

      {slot.metadata ? (
        <div className="panel-foot">
          <dl className="meta">
            <div>
              <dt>Length</dt>
              <dd>{formatDuration(slot.metadata.duration)}</dd>
            </div>
            <div>
              <dt>Size</dt>
              <dd>
                {slot.metadata.width} × {slot.metadata.height}
              </dd>
            </div>
            <div>
              <dt>Format</dt>
              <dd>{slot.metadata.format}</dd>
            </div>
          </dl>
          <div className="video-controls">
            <button
              type="button"
              className="btn btn-quiet btn-sm"
              onClick={() => inputRef.current?.click()}
            >
              Replace
            </button>
            <button type="button" className="btn btn-quiet btn-sm" onClick={onClear}>
              Remove
            </button>
          </div>
        </div>
      ) : null}

      {/* One input for both the dropzone and the Replace button; two elements
          sharing a single ref only worked because they never rendered together. */}
      <input
        id={inputId}
        ref={inputRef}
        type="file"
        accept="video/*"
        hidden
        onChange={(e) => pickFile(e.target.files?.[0])}
      />
    </HudFrame>
  );
}
