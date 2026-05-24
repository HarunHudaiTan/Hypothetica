import { useState, useEffect, useMemo } from "react";
import { AnimatePresence, motion } from "motion/react";
import type { SentenceAnnotation, MatchedSection } from "../types/api";
import { getSentenceMatches } from "../lib/api";

interface Props {
  annotation: SentenceAnnotation;
  jobId: string;
  onClose: () => void;
}

const CRITERION_DISPLAY: Record<string, string> = {
  problem_similarity: "Problem",
  method_similarity: "Method",
  domain_similarity: "Domain",
  contribution_similarity: "Contribution",
};

function critAccent(label: string): string {
  if (label === "low") return "var(--color-vermillion)";
  if (label === "medium") return "var(--color-ochre)";
  return "var(--color-moss)";
}

export default function MatchesModal({
  annotation,
  jobId,
  onClose,
}: Props) {
  const [ragMatches, setRagMatches] = useState<MatchedSection[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedIndex, setSelectedIndex] = useState(0);

  const accent = critAccent(annotation.label);

  useEffect(() => {
    let cancelled = false;
    async function fetchMatches() {
      try {
        const result = await getSentenceMatches(jobId, annotation.sentence);
        if (!cancelled) setRagMatches(result);
      } catch {
        // ignore
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    fetchMatches();
    return () => {
      cancelled = true;
    };
  }, [jobId, annotation.sentence]);

  const allMatches = useMemo(() => {
    const linked = annotation.linked_sections ?? [];
    if (linked.length > 0) return linked;
    return ragMatches;
  }, [annotation.linked_sections, ragMatches]);

  const currentMatch = allMatches[selectedIndex];

  return (
    <AnimatePresence>
      <motion.div
        key="overlay"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        transition={{ duration: 0.25 }}
        className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-[color:var(--color-night)]/55 backdrop-blur-[2px]"
        onClick={(e) => e.target === e.currentTarget && onClose()}
      >
        <motion.div
          key="modal"
          initial={{ opacity: 0, y: 24, scale: 0.98 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: 16, scale: 0.98 }}
          transition={{ duration: 0.35, ease: [0.2, 0.7, 0.25, 1] }}
          className="relative w-full max-w-5xl max-h-[88vh] bg-[color:var(--color-paper)] border border-[color:var(--color-rule-strong)] shadow-[0_30px_80px_-30px_rgba(0,0,0,0.5)] overflow-hidden flex flex-col"
        >
          {/* Header */}
          <div className="px-7 py-5 border-b border-[color:var(--color-rule)] flex items-start justify-between gap-4 flex-shrink-0">
            <div>
              <p
                className="small-caps mb-1"
                style={{ color: accent }}
              >
                evidence bridge · folio detail
              </p>
              <h2 className="font-display text-3xl tracking-tight text-[color:var(--color-ink)]">
                The Author ↔ The Corpus
              </h2>
            </div>
            <button
              onClick={onClose}
              className="font-mono text-sm text-[color:var(--color-ink-fade)] hover:text-[color:var(--color-vermillion)] transition-colors"
            >
              ✕ close
            </button>
          </div>

          {/* Body */}
          <div className="flex-1 overflow-auto">
            {loading ? (
              <div className="flex items-center justify-center py-24 gap-3">
                <span className="inline-block w-4 h-4 border-2 border-[color:var(--color-vermillion)] border-t-transparent rounded-full animate-spin" />
                <span className="font-display italic text-[color:var(--color-ink-fade)]">
                  consulting the corpus…
                </span>
              </div>
            ) : allMatches.length === 0 ? (
              <div className="text-center py-20">
                <p className="font-display italic text-[color:var(--color-ink-fade)] text-lg">
                  no matching passages were located for this sentence.
                </p>
              </div>
            ) : (
              <div className="grid grid-cols-1 lg:grid-cols-2 min-h-[420px]">
                {/* LEFT */}
                <div className="p-7 border-b lg:border-b-0 lg:border-r border-[color:var(--color-rule)] bg-[color:var(--color-paper-shade)]">
                  <p className="small-caps text-[color:var(--color-ink-fade)] mb-3">
                    your idea
                  </p>
                  <div
                    className="border-l-2 pl-4 py-1 mb-5"
                    style={{ borderColor: accent }}
                  >
                    <p className="font-display text-xl leading-snug text-[color:var(--color-ink)]">
                      “{annotation.sentence}”
                    </p>
                  </div>

                  <div className="flex flex-wrap gap-3 small-caps mb-6">
                    {Object.entries(annotation.criteria_labels ?? {})
                      .filter(([, lbl]) => lbl !== "high")
                      .map(([crit, lbl]) => (
                        <span
                          key={crit}
                          style={{
                            color: critAccent(lbl),
                            borderBottom: `1px solid ${critAccent(lbl)}`,
                          }}
                        >
                          {CRITERION_DISPLAY[crit] ?? crit} · {lbl}
                        </span>
                      ))}
                  </div>

                  {allMatches.length > 1 && (
                    <div>
                      <p className="small-caps text-[color:var(--color-ink-fade)] mb-2">
                        match {selectedIndex + 1} / {allMatches.length}
                      </p>
                      <div className="flex flex-wrap gap-2">
                        {allMatches.map((m, i) => (
                          <button
                            key={i}
                            onClick={() => setSelectedIndex(i)}
                            className={`small-caps px-2 py-1 transition-colors ${
                              selectedIndex === i
                                ? "text-[color:var(--color-paper)] bg-[color:var(--color-ink)]"
                                : "text-[color:var(--color-ink-fade)] hover:text-[color:var(--color-ink)] border border-[color:var(--color-rule)]"
                            }`}
                          >
                            {m.criterion
                              ? CRITERION_DISPLAY[m.criterion] ?? m.criterion
                              : `#${i + 1}`}
                            <span className="ml-1 opacity-70 font-mono">
                              · {Math.round(m.similarity * 100)}%
                            </span>
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                {/* RIGHT */}
                <div className="p-7 bg-[color:var(--color-paper)]">
                  <p className="small-caps text-[color:var(--color-ink-fade)] mb-3">
                    the corpus replies
                  </p>
                  {currentMatch ? (
                    <div className="space-y-5">
                      <div>
                        <p className="font-display italic text-xl text-[color:var(--color-ink)] leading-snug">
                          {currentMatch.paper_title || "Unknown source"}
                        </p>
                        {currentMatch.heading && (
                          <p className="font-mono text-[10px] text-[color:var(--color-ink-fade)] uppercase tracking-wider mt-1">
                            § {currentMatch.heading}
                          </p>
                        )}
                      </div>

                      <div
                        className="border-l-2 pl-4 py-2"
                        style={{ borderColor: accent }}
                      >
                        <p className="small-caps text-[color:var(--color-ink-fade)] mb-1">
                          matching passage
                        </p>
                        {currentMatch.text_snippet ? (
                          <p className="font-body text-[15px] leading-relaxed text-[color:var(--color-ink-soft)]">
                            “{currentMatch.text_snippet}”
                          </p>
                        ) : (
                          <p className="font-display italic text-[color:var(--color-ink-fade)]">
                            no text snippet · similarity detected semantically.
                          </p>
                        )}
                      </div>

                      {currentMatch.reason && (
                        <div>
                          <p className="small-caps text-[color:var(--color-vermillion)] mb-1">
                            why it matches?
                          </p>
                          <p className="font-display italic text-[color:var(--color-ink)] leading-relaxed">
                            {currentMatch.reason}
                          </p>
                        </div>
                      )}
                    </div>
                  ) : (
                    <p className="font-display italic text-[color:var(--color-ink-fade)]">
                      Select a match to view its evidence.
                    </p>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Footer nav */}
          {allMatches.length > 1 && (
            <div className="px-7 py-3 border-t border-[color:var(--color-rule)] flex items-center justify-between flex-shrink-0 bg-[color:var(--color-paper-shade)]">
              <button
                onClick={() =>
                  setSelectedIndex((p) =>
                    p > 0 ? p - 1 : allMatches.length - 1
                  )
                }
                className="small-caps text-[color:var(--color-ink-fade)] hover:text-[color:var(--color-ink)] transition-colors"
              >
                ← previous match
              </button>
              <span className="font-mono text-[11px] text-[color:var(--color-ink-fade)]">
                {selectedIndex + 1} of {allMatches.length}
              </span>
              <button
                onClick={() =>
                  setSelectedIndex((p) =>
                    p < allMatches.length - 1 ? p + 1 : 0
                  )
                }
                className="small-caps text-[color:var(--color-ink-fade)] hover:text-[color:var(--color-ink)] transition-colors"
              >
                next match →
              </button>
            </div>
          )}
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}
