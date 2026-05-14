import { useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import type { PaperDetail } from "../types/api";
import { evidenceTableHeading, sourceLinkLabel } from "../lib/evidenceLabels";

interface Props {
  papers: PaperDetail[];
  originalityScore?: number;
}

type SortKey = "overall" | "problem" | "method" | "domain" | "contribution";

const SORT_OPTIONS: { key: SortKey; label: string }[] = [
  { key: "overall", label: "overall" },
  { key: "problem", label: "problem" },
  { key: "method", label: "method" },
  { key: "domain", label: "domain" },
  { key: "contribution", label: "contribution" },
];

function toLikert(s: number): number {
  if (s >= 1.0) return 5;
  if (s >= 0.75) return 4;
  if (s >= 0.5) return 3;
  if (s >= 0.25) return 2;
  return 1;
}

function severityColor(score: number): string {
  if (score >= 0.7) return "var(--color-vermillion)";
  if (score >= 0.4) return "var(--color-ochre)";
  return "var(--color-moss)";
}

function getScore(paper: PaperDetail, key: SortKey): number {
  if (key === "overall") return paper.paper_similarity_score ?? 0;
  const c = paper.criteria_scores;
  if (!c) return 0;
  if (key === "problem") return c.problem_similarity;
  if (key === "method") return c.method_similarity;
  if (key === "domain") return c.domain_similarity;
  return c.contribution_similarity;
}

function parseReasons(reason: string | undefined): Record<string, string> {
  if (!reason) return {};
  const map: Record<string, string> = {};
  reason.split(" | ").forEach((part) => {
    const colonIdx = part.indexOf(":");
    if (colonIdx > -1) {
      map[part.slice(0, colonIdx).trim().toLowerCase()] = part
        .slice(colonIdx + 1)
        .trim();
    }
  });
  return map;
}

export default function PaperTable({ papers }: Props) {
  const [sortKey, setSortKey] = useState<SortKey>("overall");
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const validPapers = papers.filter(
    (p) => p.paper_similarity_score !== undefined
  );
  const sorted = [...validPapers].sort(
    (a, b) => getScore(b, sortKey) - getScore(a, sortKey)
  );

  const primarySource = papers[0]?.source;
  const { title: tableTitle, itemWord } = evidenceTableHeading(primarySource);

  return (
    <div className="relative">
      <div className="flex items-baseline justify-between mb-1">
        <div>
          <h3 className="font-display text-3xl tracking-tight">
            {tableTitle}
          </h3>
        </div>
        <span className="font-mono text-[11px] text-[color:var(--color-ink-soft)] font-bold">
          {sorted.length} {itemWord}
          {sorted.length !== 1 ? "s" : ""}
        </span>
      </div>
      <p className="font-body text-[15px] text-[color:var(--color-ink-soft)] mb-5 leading-snug">
        ranked by similarity with the submission · click any entry for the
        criterion-by-criterion assessment
      </p>

      {/* Sort ribbon */}
      <div className="flex items-center gap-1 flex-wrap mb-6 border-y border-[color:var(--color-rule)] py-2">
        <span className="small-caps text-[color:var(--color-ink-fade)] mr-2">
          sort by
        </span>
        {SORT_OPTIONS.map(({ key, label }) => (
          <button
            key={key}
            onClick={() => setSortKey(key)}
            className={`small-caps px-3 py-1 transition-colors ${
              sortKey === key
                ? "text-[color:var(--color-vermillion)] border-b border-[color:var(--color-vermillion)]"
                : "text-[color:var(--color-ink-fade)] hover:text-[color:var(--color-ink)]"
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Bibliography entries */}
      <ol className="divide-y divide-[color:var(--color-rule)]">
        {sorted.map((paper, idx) => {
          const isExpanded = expandedId === paper.paper_id;
          const overall = paper.paper_similarity_score ?? 0;
          const overallPct = Math.round(overall * 100);
          const accent = severityColor(overall);

          return (
            <motion.li
              key={paper.paper_id}
              initial={{ opacity: 0, y: 8 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-50px" }}
              transition={{ duration: 0.5, delay: idx * 0.04 }}
              className={isExpanded ? "bg-[color:var(--color-paper-shade)]/60" : ""}
            >
              <button
                type="button"
                onClick={() =>
                  setExpandedId(isExpanded ? null : paper.paper_id)
                }
                className="block w-full text-left py-5 hover:bg-[color:var(--color-paper-shade)]/40 transition-colors"
              >
                <div className="grid grid-cols-[2.2rem_1fr_auto] gap-4">
                  <span className="font-mono text-[11px] text-[color:var(--color-ink-fade)] pt-1.5 numeric-tabular">
                    [{(idx + 1).toString().padStart(2, "0")}]
                  </span>

                  <div className="min-w-0">
                    {/* Authors · Year */}
                    {paper.authors.length > 0 && (
                      <p className="font-body text-sm text-[color:var(--color-ink-soft)] mb-1">
                        {paper.authors.slice(0, 3).join(", ")}
                        {paper.authors.length > 3 &&
                          ` et al. (+${paper.authors.length - 3})`}
                      </p>
                    )}
                    <p className="font-display text-lg md:text-xl text-[color:var(--color-ink)] leading-snug">
                      {paper.title}
                    </p>
                    <div className="flex items-center gap-4 flex-wrap mt-3">
                      {paper.categories.slice(0, 3).map((cat) => (
                        <span
                          key={cat}
                          className="font-mono uppercase tracking-wider text-[12px] text-[color:var(--color-ink-soft)] font-bold"
                        >
                          {cat}
                        </span>
                      ))}
                      {paper.url && (
                        <a
                          href={paper.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          onClick={(e) => e.stopPropagation()}
                          className="inline-flex items-center gap-1 px-2.5 py-1 border border-[color:var(--color-rule-strong)] font-mono text-[12px] small-caps text-[color:var(--color-ink)] hover:bg-[color:var(--color-ink)] hover:text-[color:var(--color-paper)] hover:border-[color:var(--color-ink)] transition-colors font-bold"
                        >
                          {paper.source === "github"
                            ? "github"
                            : sourceLinkLabel(paper.source).toLowerCase()}
                          <span aria-hidden>↗</span>
                        </a>
                      )}
                      {paper.pdf_url && paper.source !== "github" && (
                        <a
                          href={paper.pdf_url}
                          target="_blank"
                          rel="noopener noreferrer"
                          onClick={(e) => e.stopPropagation()}
                          className="inline-flex items-center gap-1 px-2.5 py-1 border border-[color:var(--color-rule-strong)] font-mono text-[12px] small-caps text-[color:var(--color-ink)] hover:bg-[color:var(--color-vermillion)] hover:text-[color:var(--color-paper)] hover:border-[color:var(--color-vermillion)] transition-colors font-bold"
                        >
                          pdf
                          <span aria-hidden>↗</span>
                        </a>
                      )}
                    </div>
                  </div>

                  <div className="text-right flex flex-col items-end gap-1 pt-1">
                    <span
                      className="font-display text-3xl leading-none numeric-tabular"
                      style={{ color: accent }}
                    >
                      {overallPct}
                      <span className="text-sm text-[color:var(--color-ink-fade)]">
                        %
                      </span>
                    </span>
                    <span className="font-mono text-[9px] small-caps text-[color:var(--color-ink-fade)]">
                      similarity
                    </span>
                    <div className="w-20 h-px bg-[color:var(--color-rule)] mt-1">
                      <div
                        className="h-full"
                        style={{
                          width: `${overallPct}%`,
                          background: accent,
                          height: "3px",
                        }}
                      />
                    </div>
                  </div>
                </div>
              </button>

              <AnimatePresence>
                {isExpanded && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: "auto", opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.35 }}
                    className="overflow-hidden"
                  >
                    <PaperDetailPanel paper={paper} />
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.li>
          );
        })}
      </ol>

      {sorted.length === 0 && (
        <p className="text-center py-12 font-display italic text-[color:var(--color-ink-fade)]">
          No entries with analysis data available.
        </p>
      )}
    </div>
  );
}

function PaperDetailPanel({ paper }: { paper: PaperDetail }) {
  const [expandedCriterion, setExpandedCriterion] = useState<string | null>(
    null
  );
  const c = paper.criteria_scores;
  const reasons = parseReasons(paper.reason);

  const criteria = c
    ? [
        {
          label: "Problem",
          reasonKey: "problem",
          score: c.problem_similarity,
          desc: "alignment of the research question.",
        },
        {
          label: "Method",
          reasonKey: "method",
          score: c.method_similarity,
          desc: "alignment of approach and apparatus.",
        },
        {
          label: "Domain",
          reasonKey: "domain",
          score: c.domain_similarity,
          desc: "alignment of field and application.",
        },
        {
          label: "Contribution",
          reasonKey: "contribution",
          score: c.contribution_similarity,
          desc: "alignment of claimed outcome.",
        },
      ]
    : [];

  return (
    <div className="pl-12 pr-2 pb-6">
      {/* Abstract */}
      <div className="mb-5 border-l-2 border-[color:var(--color-rule)] pl-4">
        <p className="small-caps text-[color:var(--color-ink-fade)] mb-2">
          abstract
        </p>
        <p className="font-body text-[14px] leading-relaxed text-[color:var(--color-ink-soft)] line-clamp-4">
          {paper.abstract}
        </p>
      </div>

      {criteria.length > 0 && (
        <div>
          <p className="small-caps text-[color:var(--color-ink-fade)] mb-3">
            criterion findings
            {Object.keys(reasons).length > 0 && (
              <span className="ml-2 font-mono text-[10px] normal-case tracking-normal">
                · click for the why it matches?
              </span>
            )}
          </p>
          <ol className="space-y-3">
            {criteria.map(({ label, reasonKey, score, desc }) => {
              const likert = toLikert(score);
              const accent = severityColor(score);
              const isExp = expandedCriterion === reasonKey;
              const reasonText = reasons[reasonKey];

              return (
                <li
                  key={label}
                  onClick={() =>
                    reasonText && setExpandedCriterion(isExp ? null : reasonKey)
                  }
                  className={`border-l border-[color:var(--color-rule)] pl-4 py-2 transition-all ${
                    reasonText ? "cursor-pointer hover:border-l-2" : ""
                  } ${isExp ? "bg-[color:var(--color-paper)]" : ""}`}
                  style={isExp ? { borderLeftColor: accent, borderLeftWidth: "2px" } : {}}
                >
                  <div className="flex items-baseline justify-between mb-1">
                    <span className="font-display text-lg tracking-tight">
                      {label}
                    </span>
                    <span className="font-mono text-xs flex items-baseline gap-2">
                      <span className="text-[color:var(--color-ink-fade)] small-caps">
                        {likert}/5
                      </span>
                      <span
                        className="font-display text-lg numeric-tabular"
                        style={{ color: accent }}
                      >
                        {Math.round(score * 100)}%
                      </span>
                      {reasonText && (
                        <span className="text-[color:var(--color-ink-fade)]">
                          {isExp ? "▴" : "▾"}
                        </span>
                      )}
                    </span>
                  </div>
                  <p
                    className={`font-body text-[14px] leading-relaxed ${
                      isExp && reasonText
                        ? "text-[color:var(--color-ink)]"
                        : "text-[color:var(--color-ink-soft)]"
                    }`}
                  >
                    {isExp && reasonText ? reasonText : desc}
                  </p>
                </li>
              );
            })}
          </ol>
        </div>
      )}
    </div>
  );
}
