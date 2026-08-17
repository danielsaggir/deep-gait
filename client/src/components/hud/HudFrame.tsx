import type { ReactNode } from "react";

type Props = {
  children: ReactNode;
  className?: string;
  active?: boolean;
};

/** Thin corner brackets that frame a panel without adding a filled box. */
export function HudFrame({ children, className = "", active = false }: Props) {
  return (
    <div className={`hud-frame ${active ? "is-active" : ""} ${className}`}>
      <span className="bracket tl" />
      <span className="bracket tr" />
      <span className="bracket bl" />
      <span className="bracket br" />
      {children}
    </div>
  );
}
