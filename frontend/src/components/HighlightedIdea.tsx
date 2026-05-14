import { useState, useEffect } from "react";
import { AnimatePresence, motion } from "motion/react";
import type { SentenceAnnotation, MatchedSection } from "../types/api";
import { getSentenceMatches } from "../lib/api";

interface Props {
  annotations: SentenceAnnotation[];
  jobId: string;
}

const CRITERION_DISPLAY: Record<string, string> = {
  problem_similarity: "Problem",
  method_similarity: "Method",
  domain_similarity: "Domain",
  contribution_similarity: "Contribution",
};

function labelClass(label: string): string {
  if (label === "low") return "ink-mark ink-mark-low";
  if (label === "medium") return "ink-mark ink-mark-medium";
  return "";
}

function critAccent(lbl: string): string {
  if (lbl === "low") return "var(--color-vermillion)";
  if (lbl === "medium") return "var(--color-ochre)";
  return "var(--color-moss)";
}

function severityWord(lbl: string): string {
  if (lbl === "low") return "high";
  if (lbl === "medium") return "moderate";
  return "distinct";
}

export default function HighlightedIdea({ annotations, jobId }: Props) {
  const [selectedAnn, setSelectedAnn] = useState<SentenceAnnotation | null>(
    null
  );
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const [matches, setMatches] = useState<MatchedSection[]>([]);
  const [loadingMatches, setLoadingMatches] = useState(false);
  const [selectedMatchIndex, setSelectedMatchIndex] = useState(0);

  useEffect(() => {
    let cancelled = false;
    if (!selectedAnn) {
      const t = window.setTimeout(() => {
        if (!cancelled) setMatches([]);
      }, 0);
      return () => {
        cancelled = true;
        window.clearTimeout(t);
      };
    }
    const linked = selectedAnn.linked_sections ?? [];
    if (linked.length > 0) {
      const t = window.setTimeout(() => {
        if (cancelled) return;
        setMatches(linked);
        setSelectedMatchIndex(0);
      }, 0);
      return () => {
        cancelled = true;
        window.clearTimeout(t);
      };
    }
    setLoadingMatches(true);
    getSentenceMatches(jobId, selectedAnn.sentence)
      .then((result) => {
        if (cancelled) return;
        setMatches(result);
        setSelectedMatchIndex(0);
      })
      .catch(() => {
        if (!cancelled) setMatches([]);
      })
      .finally(() => {
        if (!cancelled) setLoadingMatches(false);
      });
    return () => {
      cancelled = true;
    };
  }, [selectedAnn, jobId]);

  const hasLowOrMed = (ann: SentenceAnnotation) =>
    ann.label === "low" ||
    ann.label === "medium" ||
    Object.values(ann.criteria_labels ?? {}).some(
      (lbl) => lbl === "low" || lbl === "medium"
    );

  const handleSentenceClick = (ann: SentenceAnnotation) => {
    if (!hasLowOrMed(ann)) return;
    if (selectedAnn?.index === ann.index) {
      setSelectedAnn(null);
    } else {
      setSelectedAnn(ann);
    }
  };

  const currentMatch = matches[selectedMatchIndex];

  return (
    <div>
      {/* Manuscript text */}
      <div className="relative font-body text-[17px] md:text-[18px] leading-[1.85] text-[color:var(--color-ink)]">
        <p className="drop-cap">
          {annotations.map((ann, idx) => {
            const clickable = hasLowOrMed(ann);
            const isSelected = selectedAnn?.index === ann.index;
            const cls = labelClass(ann.label);
            return (
              <span
                key={idx}
                className={`relative ${cls} ${
                  clickable ? "" : "cursor-default"
                } ${
                  isSelected
                    ? "rounded-sm px-1 -mx-1 py-0.5 font-semibold"
                    : ""
                }`}
                style={
                  isSelected
                    ? {
                        background: "rgba(200, 48, 24, 0.14)",
                        color: "var(--color-vermillion)",
                        boxDecorationBreak: "clone",
                        WebkitBoxDecorationBreak: "clone",
                      }
                    : undefined
                }
                onClick={() => clickable && handleSentenceClick(ann)}
                onMouseEnter={() => setHoveredIndex(ann.index)}
                onMouseLeave={() => setHoveredIndex(null)}
              >
                {ann.sentence}
                {idx < annotations.length - 1 && " "}
                {/* Hover preview */}
                {hoveredIndex === ann.index && !selectedAnn && clickable && (
                  <span
                    className="absolute left-0 top-full mt-2 z-20 block w-80 max-w-[80vw]"
                    style={{ pointerEvents: "none" }}
                  >
                    <span className="block bg-[color:var(--color-paper)] border border-[color:var(--color-rule-strong)] shadow-lg px-4 py-3">
                      <span className="font-mono text-[10px] uppercase tracking-wider text-[color:var(--color-ink-soft)] font-bold block mb-2">
                        click to consult the source
                      </span>
                      {ann.linked_sections[0]?.text_snippet && (
                        <span className="block font-body text-sm text-[color:var(--color-ink-soft)] line-clamp-3">
                          “{ann.linked_sections[0].text_snippet.slice(0, 180)}…”
                        </span>
                      )}
                    </span>
                  </span>
                )}
              </span>
            );
          })}
        </p>

        {/* Legend */}
        <div className="mt-8 pt-4 border-t border-[color:var(--color-rule)] flex flex-wrap items-center gap-x-5 gap-y-2 font-mono text-[12px] text-[color:var(--color-ink-soft)]">
          <span className="small-caps font-bold">marginalia</span>
          <span className="inline-flex items-center gap-1.5">
            <span
              className="inline-block w-6"
              style={{
                background: "var(--color-vermillion)",
                height: "3px",
              }}
            />
            high similarity
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span
              className="inline-block w-6"
              style={{ background: "var(--color-ochre)", height: "3px" }}
            />
            moderate
          </span>
          <span className="text-[color:var(--color-ink-fade)]">
            click any mark to inspect
          </span>
        </div>
      </div>

      {/* Evidence panel — appears below the manuscript */}
      <AnimatePresence>
        {selectedAnn && (
          <motion.section
            key="evidence"
            initial={{ opacity: 0, y: 12, height: 0 }}
            animate={{ opacity: 1, y: 0, height: "auto" }}
            exit={{ opacity: 0, y: 12, height: 0 }}
            transition={{ duration: 0.45, ease: [0.2, 0.7, 0.25, 1] }}
            className="overflow-hidden"
          >
            <div className="mt-8 bg-[color:var(--color-paper)] border-2 border-[color:var(--color-vermillion)]">
              {/* Top bar */}
              <header className="flex items-center justify-between px-6 py-3 border-b-2 border-[color:var(--color-vermillion)] bg-[color:var(--color-vermillion)] text-[color:var(--color-paper)]">
                <span className="small-caps font-bold tracking-[0.22em]">
                  evidence
                </span>
                <button
                  onClick={() => setSelectedAnn(null)}
                  className="small-caps font-bold hover:underline underline-offset-4"
                >
                  ✕ close
                </button>
              </header>

              {/* Selected sentence */}
              <div className="px-6 pt-5 pb-4 border-b border-[color:var(--color-rule)]">
                <p className="small-caps text-[color:var(--color-ink-soft)] font-bold mb-2 text-[11px]">
                  your idea
                </p>
                <p className="font-display text-lg md:text-xl text-[color:var(--color-ink)] leading-snug">
                  “{selectedAnn.sentence}”
                </p>

                {/* Criteria chips */}
                <div className="flex flex-wrap items-center gap-x-6 gap-y-2 mt-4">
                  {Object.entries(selectedAnn.criteria_labels ?? {}).map(
                    ([crit, lbl]) => (
                      <span
                        key={crit}
                        className="inline-flex items-center gap-2 small-caps text-[11px] tracking-[0.16em]"
                        style={{ color: critAccent(lbl) }}
                      >
                        <span
                          aria-hidden
                          className="inline-block w-1.5 h-1.5 rounded-full"
                          style={{ background: critAccent(lbl) }}
                        />
                        <span className="font-bold">
                          {CRITERION_DISPLAY[crit] ?? crit}
                        </span>
                        <span className="opacity-75">{severityWord(lbl)}</span>
                      </span>
                    )
                  )}
                </div>
              </div>

              {/* Body */}
              <div className="px-6 py-5">
                {loadingMatches ? (
                  <div className="flex items-center gap-2 font-mono text-[12px] text-[color:var(--color-ink-soft)]">
                    <span className="inline-block w-3 h-3 border border-[color:var(--color-vermillion)] border-t-transparent rounded-full animate-spin" />
                    consulting the corpus…
                  </div>
                ) : matches.length === 0 ? (
                  <p className="font-body text-sm text-[color:var(--color-ink-soft)]">
                    No detailed passage available.
                  </p>
                ) : (
                  <div>
                    {matches.length > 1 && (
                      <div className="mb-5">
                        <p className="small-caps text-[color:var(--color-ink-soft)] font-bold mb-2 text-[11px]">
                          inspect by criterion
                        </p>
                        <div className="flex flex-wrap gap-1.5">
                          {matches.map((m, i) => {
                            const isActive = selectedMatchIndex === i;
                            return (
                              <button
                                key={i}
                                onClick={() => setSelectedMatchIndex(i)}
                                className={`small-caps font-bold px-3 py-1.5 transition-colors ${
                                  isActive
                                    ? "bg-[color:var(--color-ink)] text-[color:var(--color-paper)]"
                                    : "border border-[color:var(--color-rule-strong)] text-[color:var(--color-ink-soft)] hover:text-[color:var(--color-ink)] hover:border-[color:var(--color-ink)]"
                                }`}
                              >
                                {m.criterion
                                  ? CRITERION_DISPLAY[m.criterion] ??
                                    m.criterion
                                  : `#${i + 1}`}
                              </button>
                            );
                          })}
                        </div>
                      </div>
                    )}

                    {currentMatch && (
                      <div className="grid md:grid-cols-12 gap-6">
                        {/* Source paper meta */}
                        <div className="md:col-span-4">
                          <p className="small-caps text-[color:var(--color-ink-soft)] font-bold mb-2 text-[11px]">
                            from the source
                          </p>
                          <p className="font-display text-base text-[color:var(--color-ink)] leading-snug mb-2">
                            {currentMatch.paper_title}
                          </p>
                          {currentMatch.heading && (
                            <p className="font-mono text-[11px] text-[color:var(--color-ink-soft)] uppercase tracking-wider font-bold">
                              § {currentMatch.heading}
                            </p>
                          )}
                        </div>

                        {/* Snippet + reason */}
                        <div className="md:col-span-8 space-y-4">
                          {currentMatch.text_snippet && (
                            <div>
                              <p className="small-caps text-[color:var(--color-ink-soft)] font-bold mb-2 text-[11px]">
                                matching passage
                              </p>
                              <blockquote className="border-l-2 border-[color:var(--color-rule-strong)] pl-4 font-body text-[15px] leading-relaxed text-[color:var(--color-ink)]">
                                “{currentMatch.text_snippet}”
                              </blockquote>
                            </div>
                          )}

                          {currentMatch.reason && (
                            <div>
                              <p className="small-caps text-[color:var(--color-vermillion)] font-bold mb-2 text-[11px]">
                                why it matches?
                              </p>
                              <p className="font-body text-[15px] text-[color:var(--color-ink)] leading-relaxed">
                                {currentMatch.reason}
                              </p>
                            </div>
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          </motion.section>
        )}
      </AnimatePresence>
    </div>
  );
}
