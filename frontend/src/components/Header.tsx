import { motion } from "motion/react";

const TITLE = "HYPOTHETICA";

export default function Header() {
  return (
    <header className="relative pt-2 pb-10 mb-10">
      <motion.span
        className="hairline rule-draw block"
        initial={{ scaleX: 0 }}
        animate={{ scaleX: 1 }}
        transition={{ duration: 0.9, ease: [0.7, 0, 0.25, 1], delay: 0.1 }}
      />

      {/* Masthead title */}
      <h1 className="font-display text-[clamp(3.2rem,12vw,7.2rem)] leading-[0.86] tracking-[-0.03em] text-[color:var(--color-ink)] mt-6 mb-3 text-center">
        <span className="word-rise inline-block">
          {TITLE.split("").map((c, i) => (
            <span key={i} style={{ animationDelay: `${0.25 + i * 0.04}s` }}>
              {c}
            </span>
          ))}
        </span>
      </h1>

      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.8, delay: 0.85 }}
        className="font-display italic text-center text-[clamp(1rem,2.2vw,1.4rem)] text-[color:var(--color-ink-soft)] mb-5"
      >
        your hypothesis,{" "}
        <span className="text-[color:var(--color-vermillion)]">
          cross-examined
        </span>{" "}
        against prior art
      </motion.p>

      <motion.span
        className="hairline-double rule-draw block"
        initial={{ scaleX: 0 }}
        animate={{ scaleX: 1 }}
        transition={{ duration: 0.9, ease: [0.7, 0, 0.25, 1], delay: 0.95 }}
      />
    </header>
  );
}
