import { motion } from "motion/react";
import type { CriteriaScores } from "../types/api";

interface Props {
  criteria: CriteriaScores;
}

const CRITERIA = [
  {
    key: "problem_similarity" as const,
    label: "Problem",
    glyph: "❡",
    description: "the question itself, as framed.",
  },
  {
    key: "method_similarity" as const,
    label: "Method",
    glyph: "§",
    description: "the techniques and apparatus employed.",
  },
  {
    key: "domain_similarity" as const,
    label: "Domain",
    glyph: "✦",
    description: "the field in which the work is situated.",
  },
  {
    key: "contribution_similarity" as const,
    label: "Contribution",
    glyph: "✶",
    description: "the claimed result, the so-what.",
  },
];

function toLikert(s: number): number {
  if (s >= 1.0) return 5;
  if (s >= 0.75) return 4;
  if (s >= 0.5) return 3;
  if (s >= 0.25) return 2;
  return 1;
}

function severityColor(likert: number): string {
  if (likert >= 4) return "var(--color-vermillion)";
  if (likert >= 3) return "var(--color-ochre)";
  return "var(--color-moss)";
}

function severityWord(likert: number): string {
  if (likert >= 5) return "identical";
  if (likert >= 4) return "high similarity";
  if (likert >= 3) return "moderate";
  if (likert >= 2) return "slight";
  return "distinct";
}

export default function CriteriaBreakdown({ criteria }: Props) {
  return (
    <div>
      <h3 className="font-display text-2xl tracking-tight mb-1">
        Criteria, Assessed
      </h3>
      <p className="font-body text-[15px] text-[color:var(--color-ink-soft)] mb-5 leading-snug">
        each criterion weighed against the corpus · 1 distinct ↔ 5 identical
      </p>
      <ol className="divide-y divide-[color:var(--color-rule)]">
        {CRITERIA.map(({ key, label, glyph, description }, idx) => {
          const score = criteria[key];
          const likert = toLikert(score);
          const accent = severityColor(likert);
          return (
            <motion.li
              key={key}
              initial={{ opacity: 0, x: -8 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.4, delay: idx * 0.06 }}
              className="py-3"
            >
              <div className="flex items-baseline justify-between mb-1.5">
                <div className="flex items-baseline gap-2">
                  <span
                    className="font-display text-lg"
                    style={{ color: accent }}
                  >
                    {glyph}
                  </span>
                  <span className="font-display text-xl tracking-tight">
                    {label}
                  </span>
                </div>
                <div className="flex items-baseline gap-3">
                  <span
                    className="font-mono uppercase tracking-[0.18em] text-[11px] font-bold"
                    style={{ color: accent }}
                  >
                    {severityWord(likert)}
                  </span>
                  <span
                    className="font-display text-2xl numeric-tabular font-semibold"
                    style={{
                      color: accent,
                      fontVariationSettings: '"opsz" 144, "SOFT" 30, "WONK" 1',
                    }}
                  >
                    {likert}
                    <span className="text-[color:var(--color-ink-fade)] text-xs font-normal">
                      /5
                    </span>
                  </span>
                </div>
              </div>
              <div className="h-px bg-[color:var(--color-rule)] overflow-hidden">
                <motion.div
                  className="h-full"
                  style={{ background: accent, height: "3px" }}
                  initial={{ width: 0 }}
                  animate={{ width: `${(likert / 5) * 100}%` }}
                  transition={{ duration: 0.9, delay: 0.2 + idx * 0.06, ease: [0.2, 0.7, 0.25, 1] }}
                />
              </div>
              <p className="font-body text-[14px] text-[color:var(--color-ink-soft)] mt-1.5 leading-snug">
                {description}
              </p>
            </motion.li>
          );
        })}
      </ol>
    </div>
  );
}
