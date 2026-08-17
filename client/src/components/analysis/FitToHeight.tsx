import { useLayoutEffect, useRef, useState, type ReactNode } from "react";

type Props = { children: ReactNode };

/**
 * Scales its content down — never up — so it always fits the height the
 * surrounding layout hands it, instead of growing the panel or leaving it to
 * scroll. Charts shrink uniformly (width with height) so nothing in them
 * (line peaks, bar heights) gets stretched out of proportion; the only cost
 * on a very short window is some unused space either side.
 *
 * scrollHeight/clientHeight are layout measurements and stay accurate even
 * while a `transform: scale()` is applied, so re-measuring after we've
 * already shrunk the content can't spiral into a feedback loop.
 */
export function FitToHeight({ children }: Props) {
  const outerRef = useRef<HTMLDivElement>(null);
  const innerRef = useRef<HTMLDivElement>(null);
  const [scale, setScale] = useState(1);

  useLayoutEffect(() => {
    const outer = outerRef.current;
    const inner = innerRef.current;
    if (!outer || !inner) return;

    const recompute = () => {
      const availableHeight = outer.clientHeight;
      const naturalHeight = inner.scrollHeight;
      if (availableHeight <= 0 || naturalHeight <= 0) return;
      const next = Math.min(1, availableHeight / naturalHeight);
      setScale((prev) => (Math.abs(prev - next) > 0.004 ? next : prev));
    };

    recompute();

    // Not implemented in the jsdom environment the unit tests run under.
    if (typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(recompute);
    observer.observe(outer);
    observer.observe(inner);
    return () => observer.disconnect();
  });

  return (
    <div ref={outerRef} className="fit-height">
      <div ref={innerRef} className="fit-height-inner" style={{ transform: `scale(${scale})` }}>
        {children}
      </div>
    </div>
  );
}
