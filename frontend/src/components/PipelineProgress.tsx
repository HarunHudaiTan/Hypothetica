import { useMemo } from "react";
import { AnimatePresence, motion } from "motion/react";

interface RealityCheckInfo {
  warning: string | null;
  result: Record<string, unknown>;
}

interface Props {
  progress: number;
  message: string;
  error: string | null;
  realityCheck: RealityCheckInfo | null;
  onRetry?: () => void;
  evidenceDisplayName?: string;
  evidenceNounSingular?: string;
  userIdea?: string;
}

function prettifyMessage(raw: string): string {
  if (!raw) return raw;
  return raw
    .replace(/Layer 1 analysis/gi, "Candidate Originality Scoring")
    .replace(/Layer 2 aggregation/gi, "Idea Originality Scoring")
    .replace(/\bLayer 1\b/gi, "Candidate Originality Scoring")
    .replace(/\bLayer 2\b/gi, "Idea Originality Scoring")
    .replace(/PDF Processing & Indexing/gi, "Parsing Phase")
    .replace(/Embedding Search & Reranking/gi, "Candidate Selection")
    .replace(/Query Variants & .* Search/gi, "Retrieval Phase")
    .replace(/LLM .* Selection/gi, "Candidate Selection · Relevance")
    .replace(/Follow-up Questions/gi, "User Interaction Phase");
}

const ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII"];

function buildSteps(displayName: string, _nounSingular: string) {
  return [
    {
      label: "User Interaction Phase",
      blurb: "follow-up questions enrich the idea",
      threshold: 0.1,
    },
    {
      label: "Reality Check",
      blurb: "advisory glance at the public record",
      threshold: 0.05,
    },
    {
      label: "Retrieval Phase",
      blurb: `query variants canvass ${displayName}`,
      threshold: 0.25,
    },
    {
      label: "Candidate Selection · Embedding",
      blurb: "vector similarity narrows the pool",
      threshold: 0.4,
    },
    {
      label: "Candidate Selection · Reranking",
      blurb: "cross-encoder + LLM judge pick the shortlist",
      threshold: 0.5,
    },
    {
      label: "Parsing Phase",
      blurb: "documents fetched, chunked, indexed",
      threshold: 0.75,
    },
    {
      label: "Candidate Originality Scoring",
      blurb: "per-candidate, criterion-by-criterion",
      threshold: 0.9,
    },
    {
      label: "Idea Originality Scoring",
      blurb: "aggregation into the global verdict",
      threshold: 0.98,
    },
  ];
}

export default function PipelineProgress({
  progress,
  message,
  error,
  realityCheck,
  onRetry,
  evidenceDisplayName = "arXiv",
  evidenceNounSingular = "paper",
  userIdea,
}: Props) {
  const steps = useMemo(
    () => buildSteps(evidenceDisplayName, evidenceNounSingular),
    [evidenceDisplayName, evidenceNounSingular]
  );
  const pct = Math.round(progress * 100);
  const ideaPreview = (userIdea ?? "").trim();
  const firstSentence =
    ideaPreview.split(/(?<=[.!?])\s+/)[0]?.slice(0, 180) ?? "";
  const rcResult = realityCheck?.result as Record<string, unknown> | undefined;
  const rcExists = rcResult?.already_exists === true;
  const rcExamples = (rcResult?.existing_examples ?? []) as Array<{
    name: string;
    similarity: number;
    description: string;
  }>;
  const rcAssessment =
    typeof rcResult?.assessment === "string" ? rcResult.assessment : "";
  const rcRecommendation =
    typeof rcResult?.recommendation === "string" ? rcResult.recommendation : "";

  return (
    <div className="grid grid-cols-12 gap-6">
      {/* Marginalia column */}
      <aside className="hidden lg:flex col-span-3 flex-col gap-5 pt-2">
        {/* What you submitted */}
        {firstSentence && (
          <div className="bg-[color:var(--color-paper-shade)] border-l-2 border-[color:var(--color-ink)] pl-3 pr-2 py-3">
            <p className="small-caps text-[color:var(--color-ink-soft)] font-bold mb-2 text-[10px]">
              your submission
            </p>
            <p className="font-body text-[13px] text-[color:var(--color-ink)] leading-snug line-clamp-4">
              “{firstSentence}
              {ideaPreview.length > firstSentence.length ? "…" : ""}”
            </p>
            <p className="mt-2 font-mono text-[10px] uppercase tracking-wider text-[color:var(--color-ink-soft)] font-bold">
              source · {evidenceDisplayName}
            </p>
          </div>
        )}

        {/* Progress */}
        <div className="font-mono text-[12px] text-[color:var(--color-ink-fade)]">
          <div className="flex items-baseline justify-between">
            <span className="small-caps text-[color:var(--color-ink-soft)] font-bold">
              progress
            </span>
            <span className="font-display text-3xl text-[color:var(--color-ink)] numeric-tabular">
              {pct}
              <span className="text-base text-[color:var(--color-ink-fade)]">
                %
              </span>
            </span>
          </div>
          <div className="mt-2 h-px bg-[color:var(--color-rule)] overflow-hidden">
            <motion.div
              className="h-full bg-[color:var(--color-vermillion)]"
              animate={{ width: `${pct}%`, height: "3px" }}
              transition={{ duration: 0.6, ease: [0.2, 0.7, 0.25, 1] }}
            />
          </div>
          <p className="mt-3 font-body text-[12px] text-[color:var(--color-ink-soft)] leading-snug">
            {prettifyMessage(message) || "preparing the bench…"}
          </p>
        </div>

        {/* Reality check · under progress */}
        <AnimatePresence>
          {realityCheck && rcExists && (
            <motion.div
              key="rc-exists"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.5 }}
              className="border-l-2 border-[color:var(--color-ochre)] pl-3 py-2 bg-[color:var(--color-paper-shade)]"
            >
              <p className="small-caps text-[color:var(--color-ochre)] font-bold mb-1 text-[10px]">
                AI overview
              </p>
              <p className="font-display text-sm text-[color:var(--color-ink)] mb-2 leading-snug">
                {rcAssessment ||
                  "similar concepts already inhabit the public record."}
              </p>
              {rcExamples.length > 0 && (
                <ul className="space-y-1 mb-2">
                  {rcExamples.slice(0, 3).map((ex, i) => (
                    <li
                      key={i}
                      className="flex items-baseline gap-2 font-body text-[11px] leading-snug"
                    >
                      <span className="font-mono text-[10px] text-[color:var(--color-ochre)] flex-shrink-0 numeric-tabular">
                        {Math.round(ex.similarity * 100)}%
                      </span>
                      <span>
                        <span className="font-display italic text-[color:var(--color-ink)]">
                          {ex.name}
                        </span>
                        <span className="text-[color:var(--color-ink-fade)]">
                          {" "}· {ex.description}
                        </span>
                      </span>
                    </li>
                  ))}
                </ul>
              )}
              {rcRecommendation && (
                <p className="font-display italic text-[11px] text-[color:var(--color-ink-soft)] mb-2 leading-snug">
                  {rcRecommendation}
                </p>
              )}
              <p className="font-mono text-[9px] text-[color:var(--color-ink-fade)] border-t border-[color:var(--color-rule)] pt-1.5 leading-snug">
                advisory only · does not affect final score.
              </p>
            </motion.div>
          )}
          {realityCheck && !rcExists && (
            <motion.div
              key="rc-clear"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="border-l-2 border-[color:var(--color-moss)] pl-3 py-2 bg-[color:var(--color-paper-shade)]"
            >
              <p className="small-caps text-[color:var(--color-moss)] font-bold mb-1 text-[10px]">
                AI overview
              </p>
              <p className="font-display italic text-xs text-[color:var(--color-ink-soft)] leading-snug">
                advisory scan surfaced no obvious extant solutions.
              </p>
            </motion.div>
          )}
        </AnimatePresence>
      </aside>

      {/* Main */}
      <article className="col-span-12 lg:col-span-9">
        <h2 className="font-display text-4xl md:text-5xl tracking-tight mb-2">
          Assessment in Progress
        </h2>
        <p className="font-display italic text-[color:var(--color-ink-fade)] mb-1">
          sit tight · this usually takes a couple of minutes
        </p>

        <div className="lg:hidden mt-4 mb-6 font-mono text-xs text-[color:var(--color-ink-fade)] flex items-center gap-3">
          <span>{pct}%</span>
          <div className="flex-1 h-px bg-[color:var(--color-rule)]">
            <motion.div
              className="h-full bg-[color:var(--color-vermillion)]"
              animate={{ width: `${pct}%`, height: "3px" }}
              transition={{ duration: 0.6 }}
            />
          </div>
        </div>

        <div className="hairline-double mt-6 mb-6" />

        {error ? (
          <div className="border-l-2 border-[color:var(--color-vermillion)] pl-5 py-4 bg-[color:var(--color-paper-shade)]">
            <p className="small-caps text-[color:var(--color-vermillion)] mb-2">
              the assessment failed
            </p>
            <p className="font-body text-[color:var(--color-ink-soft)] mb-3">
              {error}
            </p>
            {onRetry && (
              <button
                onClick={onRetry}
                className="small-caps text-[color:var(--color-vermillion)] hover:underline underline-offset-4"
              >
                ↺ begin afresh
              </button>
            )}
          </div>
        ) : (
          <ol className="space-y-1">
            {steps.map((step, idx) => {
              const done = progress >= step.threshold;
              const active = !done && progress >= step.threshold - 0.1;
              return (
                <motion.li
                  key={step.label}
                  initial={{ opacity: 0, x: -8 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.4, delay: idx * 0.05 }}
                  className={`relative grid grid-cols-[3rem_1fr_auto] gap-4 items-baseline py-4 border-b border-[color:var(--color-rule)] transition-colors ${
                    active
                      ? "bg-[color:var(--color-paper-shade)]/60"
                      : ""
                  }`}
                >
                  <span
                    className={`font-display text-2xl md:text-3xl tracking-tight ${
                      done
                        ? "text-[color:var(--color-moss)]"
                        : active
                          ? "text-[color:var(--color-vermillion)]"
                          : "text-[color:var(--color-ink-mute)]"
                    }`}
                  >
                    {ROMAN[idx] ?? idx + 1}
                  </span>
                  <span className="min-w-0">
                    <span
                      className={`block font-display text-xl md:text-2xl tracking-tight leading-tight ${
                        done || active
                          ? "text-[color:var(--color-ink)]"
                          : "text-[color:var(--color-ink-mute)]"
                      }`}
                    >
                      {step.label}
                    </span>
                    <span className="block font-display italic text-sm text-[color:var(--color-ink-fade)] mt-0.5">
                      · {step.blurb}
                    </span>
                  </span>
                  <span className="flex-shrink-0 small-caps">
                    {done ? (
                      <span className="text-[color:var(--color-moss)]">
                        ✓ entered
                      </span>
                    ) : active ? (
                      <span className="flex items-center gap-2 text-[color:var(--color-vermillion)]">
                        <span className="w-1.5 h-1.5 rounded-full bg-[color:var(--color-vermillion)] pulse-dot" />
                        in session
                      </span>
                    ) : (
                      <span className="text-[color:var(--color-ink-mute)]">
                        pending
                      </span>
                    )}
                  </span>
                </motion.li>
              );
            })}
          </ol>
        )}

        {/* Reality check shown in left marginalia (desktop). Mobile fallback below. */}
        <div className="lg:hidden">
          <AnimatePresence>
            {realityCheck && rcExists && (
              <motion.div
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.5 }}
                className="mt-8 border-l-2 border-[color:var(--color-ochre)] pl-5 py-4 bg-[color:var(--color-paper-shade)]"
              >
                <p className="small-caps text-[color:var(--color-ochre)] font-bold mb-1">
                  AI overview
                </p>
                <p className="font-display text-lg text-[color:var(--color-ink)] mb-3 leading-snug">
                  {rcAssessment ||
                    "similar concepts already inhabit the public record."}
                </p>
                {rcExamples.length > 0 && (
                  <ul className="space-y-2 mb-3">
                    {rcExamples.slice(0, 3).map((ex, i) => (
                      <li
                        key={i}
                        className="flex items-baseline gap-3 font-body text-sm"
                      >
                        <span className="font-mono text-xs text-[color:var(--color-ochre)] w-10 flex-shrink-0 numeric-tabular">
                          {Math.round(ex.similarity * 100)}%
                        </span>
                        <span>
                          <span className="font-display italic text-[color:var(--color-ink)]">
                            {ex.name}
                          </span>
                          <span className="text-[color:var(--color-ink-fade)]">
                            {" "}· {ex.description}
                          </span>
                        </span>
                      </li>
                    ))}
                  </ul>
                )}
                {rcRecommendation && (
                  <p className="font-display italic text-sm text-[color:var(--color-ink-soft)] mb-2">
                    {rcRecommendation}
                  </p>
                )}
                <p className="font-mono text-[10px] text-[color:var(--color-ink-fade)] border-t border-[color:var(--color-rule)] pt-2">
                  advisory only · does not affect the final score.
                </p>
              </motion.div>
            )}
            {realityCheck && !rcExists && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="mt-8 border-l-2 border-[color:var(--color-moss)] pl-5 py-3 bg-[color:var(--color-paper-shade)]"
              >
                <p className="small-caps text-[color:var(--color-moss)] font-bold mb-1">
                  AI overview
                </p>
                <p className="font-display italic text-[color:var(--color-ink-soft)]">
                  advisory scan surfaced no obvious extant solutions.
                </p>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </article>
    </div>
  );
}
