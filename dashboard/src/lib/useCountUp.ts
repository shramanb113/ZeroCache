import { useEffect, useRef, useState } from "react";

const prefersReducedMotion = (): boolean => {
  try {
    return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  } catch {
    return false;
  }
};

/** Ease a displayed number toward `target`. One rAF loop lerps toward the latest
 *  target every frame, so it retargets seamlessly on each poll and can never lag
 *  a moving value. Snaps instantly under reduced motion or on a large jump (a
 *  counter reset). The signature motion of the hero readout — the only thing on
 *  the page that animates per poll. */
export function useCountUp(target: number): number {
  const [display, setDisplay] = useState(target);
  const targetRef = useRef(target);
  const curRef = useRef(target);
  const rafRef = useRef(0);

  useEffect(() => {
    targetRef.current = target;

    const jump = Math.abs(target - curRef.current);
    if (prefersReducedMotion() || jump > Math.max(1, Math.abs(curRef.current)) * 4) {
      curRef.current = target;
      cancelAnimationFrame(rafRef.current);
      rafRef.current = 0;
      setDisplay(target);
      return;
    }
    if (rafRef.current) return;

    const step = () => {
      const t = targetRef.current;
      const next = curRef.current + (t - curRef.current) * 0.24;
      if (Math.abs(t - next) < 0.005) {
        curRef.current = t;
        rafRef.current = 0;
        setDisplay(t);
        return;
      }
      curRef.current = next;
      setDisplay(next);
      rafRef.current = requestAnimationFrame(step);
    };
    rafRef.current = requestAnimationFrame(step);
  }, [target]);

  useEffect(() => () => cancelAnimationFrame(rafRef.current), []);

  return display;
}
