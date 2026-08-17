/**
 * Procedural UI audio for the DeepGait workstation.
 *
 * Every cue is synthesised from one tonal family (A minor pentatonic) and runs
 * through a shared warmth filter and short reverb, so the set reads as one
 * instrument rather than a collection of beeps. Ascending intervals resolve
 * positively, descending intervals resolve negatively.
 */

const ROOT = 220; // A3
const WARMTH_HZ = 5200;
const REVERB_SECONDS = 1.1;

type Chain = {
  ctx: AudioContext;
  dry: GainNode;
  wet: GainNode;
};

let chain: Chain | null = null;
let hum: { osc: OscillatorNode[]; gain: GainNode; lfo: OscillatorNode } | null = null;

/** Semitones above the root, as a frequency. */
function note(semitones: number): number {
  return ROOT * Math.pow(2, semitones / 12);
}

/** Exponentially decaying noise, convolved to give every cue a short tail. */
function impulseResponse(ctx: AudioContext): AudioBuffer {
  const length = Math.floor(ctx.sampleRate * REVERB_SECONDS);
  const buffer = ctx.createBuffer(2, length, ctx.sampleRate);
  for (let c = 0; c < 2; c += 1) {
    const data = buffer.getChannelData(c);
    for (let i = 0; i < length; i += 1) {
      data[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / length, 2.6);
    }
  }
  return buffer;
}

function audioChain(): Chain | null {
  if (typeof window === "undefined") return null;
  if (chain) return chain;

  const Ctor =
    window.AudioContext ||
    (window as Window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
  if (!Ctor) return null;

  const ctx = new Ctor();

  const warmth = ctx.createBiquadFilter();
  warmth.type = "lowpass";
  warmth.frequency.value = WARMTH_HZ;

  const master = ctx.createGain();
  master.gain.value = 0.5;

  const dry = ctx.createGain();
  dry.gain.value = 1;

  const convolver = ctx.createConvolver();
  convolver.buffer = impulseResponse(ctx);

  const wet = ctx.createGain();
  wet.gain.value = 0.22;

  dry.connect(warmth);
  wet.connect(convolver);
  convolver.connect(warmth);
  warmth.connect(master);
  master.connect(ctx.destination);

  chain = { ctx, dry, wet };
  return chain;
}

type ChimeOptions = {
  delay?: number;
  duration?: number;
  gain?: number;
  /** Adds a quiet octave partial for a bell-like sheen. */
  shimmer?: boolean;
};

function chime(semitones: number, options: ChimeOptions = {}): void {
  const c = audioChain();
  if (!c) return;

  const { delay = 0, duration = 0.55, gain = 0.16, shimmer = true } = options;
  const t = c.ctx.currentTime + delay;
  const freq = note(semitones);

  const envelope = c.ctx.createGain();
  envelope.gain.setValueAtTime(0.0001, t);
  envelope.gain.exponentialRampToValueAtTime(gain, t + 0.012);
  envelope.gain.exponentialRampToValueAtTime(0.0001, t + duration);

  const tone = c.ctx.createBiquadFilter();
  tone.type = "lowpass";
  tone.frequency.value = 2600;

  const fundamental = c.ctx.createOscillator();
  fundamental.type = "sine";
  fundamental.frequency.value = freq;
  fundamental.connect(envelope);

  const voices: OscillatorNode[] = [fundamental];

  if (shimmer) {
    const partial = c.ctx.createOscillator();
    partial.type = "sine";
    partial.frequency.value = freq * 2;
    const partialGain = c.ctx.createGain();
    partialGain.gain.value = 0.16;
    partial.connect(partialGain);
    partialGain.connect(envelope);
    voices.push(partial);
  }

  envelope.connect(tone);
  tone.connect(c.dry);
  tone.connect(c.wet);

  voices.forEach((v) => {
    v.start(t);
    v.stop(t + duration + 0.05);
  });
}

type SweepOptions = {
  from: number;
  to: number;
  delay?: number;
  duration?: number;
  gain?: number;
};

/** Bandpass-filtered noise: the transitional whoosh between states. */
function sweep({ from, to, delay = 0, duration = 0.4, gain = 0.1 }: SweepOptions): void {
  const c = audioChain();
  if (!c) return;

  const t = c.ctx.currentTime + delay;
  const frames = Math.floor(c.ctx.sampleRate * duration);
  const buffer = c.ctx.createBuffer(1, frames, c.ctx.sampleRate);
  const data = buffer.getChannelData(0);
  for (let i = 0; i < frames; i += 1) data[i] = Math.random() * 2 - 1;

  const source = c.ctx.createBufferSource();
  source.buffer = buffer;

  const band = c.ctx.createBiquadFilter();
  band.type = "bandpass";
  band.Q.value = 5.5;
  band.frequency.setValueAtTime(from, t);
  band.frequency.exponentialRampToValueAtTime(to, t + duration);

  const envelope = c.ctx.createGain();
  envelope.gain.setValueAtTime(0.0001, t);
  envelope.gain.exponentialRampToValueAtTime(gain, t + duration * 0.35);
  envelope.gain.exponentialRampToValueAtTime(0.0001, t + duration);

  source.connect(band);
  band.connect(envelope);
  envelope.connect(c.dry);
  envelope.connect(c.wet);

  source.start(t);
  source.stop(t + duration + 0.02);
}

/** Low drifting bed that runs for the duration of an analysis. */
function startHum(): void {
  const c = audioChain();
  if (!c || hum) return;

  const t = c.ctx.currentTime;

  const gain = c.ctx.createGain();
  gain.gain.setValueAtTime(0.0001, t);
  gain.gain.exponentialRampToValueAtTime(0.05, t + 0.9);

  const body = c.ctx.createBiquadFilter();
  body.type = "lowpass";
  body.frequency.value = 320;
  body.Q.value = 1.4;

  // Slow filter drift keeps the bed alive instead of static.
  const lfo = c.ctx.createOscillator();
  lfo.frequency.value = 0.12;
  const lfoDepth = c.ctx.createGain();
  lfoDepth.gain.value = 90;
  lfo.connect(lfoDepth);
  lfoDepth.connect(body.frequency);

  const osc: OscillatorNode[] = [];
  [note(-24), note(-24) + 0.7, note(-17)].forEach((freq, i) => {
    const o = c.ctx.createOscillator();
    o.type = i === 2 ? "sine" : "triangle";
    o.frequency.value = freq;
    const g = c.ctx.createGain();
    g.gain.value = i === 2 ? 0.3 : 1;
    o.connect(g);
    g.connect(body);
    o.start(t);
    osc.push(o);
  });

  body.connect(gain);
  gain.connect(c.dry);
  gain.connect(c.wet);
  lfo.start(t);

  hum = { osc, gain, lfo };
}

function stopHum(): void {
  const c = audioChain();
  if (!c || !hum) return;

  const t = c.ctx.currentTime;
  const { osc, gain, lfo } = hum;
  hum = null;

  gain.gain.cancelScheduledValues(t);
  gain.gain.setValueAtTime(Math.max(gain.gain.value, 0.0001), t);
  gain.gain.exponentialRampToValueAtTime(0.0001, t + 0.6);

  osc.forEach((o) => o.stop(t + 0.7));
  lfo.stop(t + 0.7);
}

export const audio = {
  resume(): void {
    void audioChain()?.ctx.resume();
  },

  /** Footage accepted: a rising minor third, light and quick. */
  accepted(): void {
    sweep({ from: 900, to: 2400, duration: 0.22, gain: 0.05 });
    chime(7, { duration: 0.4, gain: 0.11 });
    chime(12, { delay: 0.07, duration: 0.5, gain: 0.09 });
  },

  /** Analysis begins: an upward sweep that hands off to the processing bed. */
  analyzeStart(): void {
    sweep({ from: 320, to: 3200, duration: 0.5, gain: 0.09 });
    chime(0, { duration: 0.9, gain: 0.1 });
    chime(7, { delay: 0.12, duration: 0.8, gain: 0.07 });
    startHum();
  },

  /** Near-subliminal marker as each pipeline stage advances. */
  stageAdvance(): void {
    chime(24, { duration: 0.18, gain: 0.022, shimmer: false });
  },

  analyzeEnd(): void {
    stopHum();
  },

  /** Match: ascending major triad, the only fully consonant cue in the set. */
  success(): void {
    sweep({ from: 1800, to: 5200, duration: 0.35, gain: 0.045 });
    chime(0, { duration: 0.9, gain: 0.12 });
    chime(4, { delay: 0.09, duration: 0.9, gain: 0.11 });
    chime(7, { delay: 0.18, duration: 1.2, gain: 0.1 });
    chime(12, { delay: 0.27, duration: 1.4, gain: 0.07 });
  },

  /** Different: descending, warm and neutral rather than punitive. */
  different(): void {
    chime(7, { duration: 0.7, gain: 0.1 });
    chime(3, { delay: 0.11, duration: 0.9, gain: 0.09 });
    chime(-5, { delay: 0.22, duration: 1.1, gain: 0.07 });
  },

  /** Failure: low descending semitone, soft enough not to punish the room. */
  failure(): void {
    sweep({ from: 1400, to: 260, duration: 0.5, gain: 0.05 });
    chime(-7, { duration: 0.8, gain: 0.09, shimmer: false });
    chime(-8, { delay: 0.14, duration: 1, gain: 0.07, shimmer: false });
  },

  /** Cut everything currently sounding, used when the user mutes mid-analysis. */
  silence(): void {
    stopHum();
  },

  /** Both clips rolling with skeleton overlays after classification. */
  playbackStart(): void {
    sweep({ from: 600, to: 2800, duration: 0.45, gain: 0.06 });
    chime(12, { duration: 0.55, gain: 0.08 });
    chime(19, { delay: 0.08, duration: 0.45, gain: 0.06, shimmer: false });
  },

  /** Each debrief step advance. */
  debriefStep(): void {
    chime(17, { duration: 0.22, gain: 0.035, shimmer: false });
  },

  getState(): "unsupported" | AudioContextState {
    const c = audioChain();
    if (!c) return "unsupported";
    return c.ctx.state;
  },
};
