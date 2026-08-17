let ctx: AudioContext | null = null;

function context(): AudioContext | null {
  if (typeof window === "undefined") return null;
  if (!ctx) {
    const Ctor = window.AudioContext || (window as Window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!Ctor) return null;
    ctx = new Ctor();
  }
  return ctx;
}

function tone(freq: number, duration: number, type: OscillatorType, gain = 0.04): void {
  const audio = context();
  if (!audio) return;
  const osc = audio.createOscillator();
  const g = audio.createGain();
  osc.type = type;
  osc.frequency.value = freq;
  g.gain.value = gain;
  g.gain.exponentialRampToValueAtTime(0.0001, audio.currentTime + duration);
  osc.connect(g);
  g.connect(audio.destination);
  osc.start();
  osc.stop(audio.currentTime + duration);
}

export const audio = {
  resume(): void {
    void context()?.resume();
  },
  accepted(): void {
    tone(880, 0.08, "sine", 0.03);
  },
  analyze(): void {
    tone(220, 0.18, "triangle", 0.035);
    setTimeout(() => tone(330, 0.12, "sine", 0.025), 90);
  },
  success(): void {
    tone(523, 0.12, "sine", 0.035);
    setTimeout(() => tone(784, 0.16, "sine", 0.03), 80);
  },
  different(): void {
    tone(196, 0.2, "triangle", 0.03);
  },
  failure(): void {
    tone(140, 0.22, "square", 0.02);
  },
};
