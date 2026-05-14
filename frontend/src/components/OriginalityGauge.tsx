import { useEffect, useState } from "react";
import { motion } from "motion/react";

interface Props {
  score: number;
  summary?: string;
}

export default function OriginalityGauge({ score, summary }: Props) {
  const [display, setDisplay] = useState(0);

  useEffect(() => {
    const reduced = window.matchMedia(
      "(prefers-reduced-motion: reduce)"
    ).matches;
    if (reduced) {
      const t = window.setTimeout(() => setDisplay(score), 0);
      return () => window.clearTimeout(t);
    }
    let raf: number;
    const t0 = performance.now();
    const ms = 1800;
    const step = (now: number) => {
      const p = Math.min((now - t0) / ms, 1);
      const ease = 1 - (1 - p) * (1 - p) * (1 - p);
      setDisplay(Math.round(ease * score));
      if (p < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [score]);

  let label: string;
  let verdict: string;
  let accent: string;
  if (score >= 70) {
    label = "High Originality";
    verdict = "the contribution stands apart";
    accent = "var(--color-moss)";
  } else if (score >= 40) {
    label = "Moderate Originality";
    verdict = "the work shares ground with extant literature";
    accent = "var(--color-ochre)";
  } else {
    label = "Low Originality";
    verdict = "the corpus already contains close kin";
    accent = "var(--color-vermillion)";
  }

  const clamped = Math.max(0, Math.min(100, score));
  const TICKS = 50;

  return (
    <div className="flex flex-col items-center w-full">
      {/* Top eyebrow */}
      <div className="flex items-center justify-center w-full max-w-[22rem] mb-3">
        <span className="small-caps text-[color:var(--color-ink-soft)] font-bold">
          the verdict
        </span>
      </div>

      {/* Double hairline */}
      <motion.span
        className="hairline-double w-full max-w-[22rem]"
        initial={{ scaleX: 0 }}
        animate={{ scaleX: 1 }}
        transition={{ duration: 0.9, ease: [0.7, 0, 0.25, 1] }}
        style={{ transformOrigin: "left" }}
      />

      {/* Hero numeral */}
      <div className="relative w-full max-w-[22rem] py-6 flex items-center justify-center">
        {/* Decorative side ornaments */}
        <motion.span
          aria-hidden
          initial={{ opacity: 0, x: 8 }}
          animate={{ opacity: 0.5, x: 0 }}
          transition={{ duration: 0.8, delay: 0.4 }}
          className="absolute left-2 top-1/2 -translate-y-1/2 font-display text-2xl"
          style={{ color: accent }}
        >
          ✦
        </motion.span>
        <motion.span
          aria-hidden
          initial={{ opacity: 0, x: -8 }}
          animate={{ opacity: 0.5, x: 0 }}
          transition={{ duration: 0.8, delay: 0.4 }}
          className="absolute right-2 top-1/2 -translate-y-1/2 font-display text-2xl"
          style={{ color: accent }}
        >
          ✦
        </motion.span>

        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7 }}
          className="flex items-baseline gap-2"
        >
          <span
            className="font-display leading-[0.85] numeric-tabular text-[clamp(5.5rem,16vw,8.5rem)]"
            style={{
              color: "var(--color-ink)",
              fontVariationSettings:
                '"opsz" 144, "SOFT" 30, "WONK" 1, "wght" 480',
            }}
          >
            {display}
          </span>
          <span className="font-display text-2xl text-[color:var(--color-ink-fade)] tracking-tight">
            /100
          </span>
        </motion.div>
      </div>

      <motion.span
        className="hairline w-full max-w-[22rem]"
        initial={{ scaleX: 0 }}
        animate={{ scaleX: 1 }}
        transition={{ duration: 0.9, ease: [0.7, 0, 0.25, 1], delay: 0.1 }}
        style={{ transformOrigin: "left" }}
      />

      {/* Scale strip · typeset histogram */}
      <div className="w-full max-w-[22rem] mt-5 mb-1 relative">
        <div className="flex items-end justify-between h-6 relative">
          {Array.from({ length: TICKS }, (_, i) => {
            const tickScore = ((i + 0.5) / TICKS) * 100;
            const filled = tickScore <= clamped;
            const isMajor = i % 10 === 0;
            return (
              <motion.span
                key={i}
                aria-hidden
                initial={{ scaleY: 0 }}
                animate={{ scaleY: 1 }}
                transition={{
                  duration: 0.5,
                  delay: 0.4 + i * 0.012,
                  ease: [0.2, 0.7, 0.25, 1],
                }}
                style={{
                  background: filled ? accent : "var(--color-ink-fade)",
                  width: "1.5px",
                  height: isMajor ? "100%" : filled ? "70%" : "45%",
                  transformOrigin: "bottom",
                  opacity: filled ? 1 : 0.85,
                }}
              />
            );
          })}
        </div>

        {/* Score pointer */}
        <motion.div
          aria-hidden
          initial={{ opacity: 0, y: -4 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, delay: 1.2 }}
          className="absolute top-0 -translate-x-1/2"
          style={{ left: `${clamped}%` }}
        >
          <span
            className="block w-px h-7"
            style={{ background: accent }}
          />
          <span
            className="block font-mono text-[10px] mt-1 numeric-tabular -translate-x-1/2 ml-1"
            style={{ color: accent }}
          >
            {display}
          </span>
        </motion.div>

        {/* Scale endpoints */}
        <div className="flex items-center justify-between mt-7 font-mono text-[10px] text-[color:var(--color-ink-soft)] numeric-tabular font-bold">
          <span>000</span>
          <span className="small-caps">similarity with the corpus →</span>
          <span>100</span>
        </div>
      </div>

      {/* Severity stamp */}
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5, delay: 0.9 }}
        className="mt-6 flex items-center gap-3"
      >
        <span
          className="block w-8 h-px"
          style={{ background: accent }}
        />
        <p
          className="small-caps tracking-[0.24em] text-sm font-bold"
          style={{ color: accent }}
        >
          {label}
        </p>
        <span
          className="block w-8 h-px"
          style={{ background: accent }}
        />
      </motion.div>

      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.6, delay: 1.1 }}
        className="mt-3 font-display italic text-[color:var(--color-ink-soft)] text-lg text-center max-w-md"
      >
        {verdict}
      </motion.p>

      {/* Why it matches? */}
      {summary && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 1.3 }}
          className="mt-7 w-full max-w-2xl border-l-2 pl-5 py-1"
          style={{ borderColor: accent }}
        >
          <p
            className="small-caps text-[color:var(--color-ink-fade)] font-bold mb-1"
            style={{ color: accent }}
          >
            why it matches?
          </p>
          <p className="font-display italic text-[color:var(--color-ink)] text-base leading-relaxed">
            “{summary}”
          </p>
        </motion.div>
      )}
    </div>
  );
}
