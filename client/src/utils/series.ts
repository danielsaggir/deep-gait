/**
 * Clips shorter than the sequence window are zero-padded during preprocessing,
 * so a short video arrives with a tail of exact zeros. Plotting that tail draws
 * a long flat run along the baseline that reads as missing data.
 */
export function trimPadding(values: number[]): number[] {
  let end = values.length;
  while (end > 0 && values[end - 1] === 0) end -= 1;
  // An all-zero series trims to nothing, and that is the honest answer. Padding
  // it back up to two points produced a flat line on the baseline — the exact
  // artefact this function exists to remove. Callers treat "fewer than two
  // samples" as no signal.
  return values.slice(0, end);
}
