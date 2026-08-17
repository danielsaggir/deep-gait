import type { ReactNode } from "react";

type Props = {
  children: ReactNode;
  className?: string;
  active?: boolean;
};

/** Floating glass shell — no corner brackets, just depth and light. */
export function HudFrame({ children, className = "", active = false }: Props) {
  return (
    <div className={`glass-panel ${active ? "is-active" : ""} ${className}`}>
      <div className="glass-panel-shimmer" aria-hidden="true" />
      {children}
    </div>
  );
}
