import { AmbientCanvas } from "./AmbientCanvas";

type Props = {
  /** Speeds and brightens the field while the model runs. */
  charged?: boolean;
  /** Steps the field back so foreground content stays legible. */
  dimmed?: boolean;
};

/**
 * One field, one motif. Everything here is drawn from the system itself —
 * walking skeleton graphs, board traces and pipeline tokens — rather than a
 * stack of unrelated decorative layers.
 */
export function AmbientField({ charged = false, dimmed = false }: Props) {
  return (
    <div
      className={`ambient-field ${charged ? "is-charged" : ""} ${dimmed ? "is-dimmed" : ""}`}
      aria-hidden="true"
    >
      <AmbientCanvas charged={charged} />
      <div className="ambient-vignette" />
    </div>
  );
}
