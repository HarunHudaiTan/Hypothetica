import { useState } from "react";
import type { PaperDetail } from "../types/api";

interface Props {
  papers: PaperDetail[];
  originalityScore?: number;
}

type SortKey = "overall" | "problem" | "method" | "domain" | "contribution";

const SORT_OPTIONS: { key: SortKey; label: string; short: string }[] = [
  { key: "overall", label: "Overall Similarity", short: "Overall" },
  { key: "problem", label: "Problem", short: "Problem" },
  { key: "method", label: "Method", short: "Method" },
  { key: "domain", label: "Domain", short: "Domain" },
  { key: "contribution", label: "Contribution", short: "Contrib." },
];


function toLikert(s: number): number {
  if (s >= 1.0) return 5;
  if (s >= 0.75) return 4;
  if (s >= 0.5) return 3;
  if (s >= 0.25) return 2;
  return 1;
}

function barColor(score: number): string {
  if (score >= 0.7) return "bg-rose-500";
  if (score >= 0.4) return "bg-amber-400";
  return "bg-emerald-500";
}

function likertColor(likert: number): { text: string; bg: string; border: string } {
  if (likert >= 4) return { text: "text-rose-700", bg: "bg-rose-50", border: "border-rose-200" };
  if (likert >= 3) return { text: "text-amber-700", bg: "bg-amber-50", border: "border-amber-200" };
  return { text: "text-emerald-700", bg: "bg-emerald-50", border: "border-emerald-200" };
}

function overallColor(score: number): { text: string; bg: string } {
  if (score >= 0.7) return { text: "text-rose-700", bg: "bg-rose-50" };
  if (score >= 0.4) return { text: "text-amber-700", bg: "bg-amber-50" };
  return { text: "text-emerald-700", bg: "bg-emerald-50" };
}

function originalityColor(score: number): string {
  if (score >= 70) return "text-emerald-600";
  if (score >= 40) return "text-amber-600";
  return "text-rose-600";
}

function getScore(paper: PaperDetail, key: SortKey): number {
  if (key === "overall") return paper.paper_similarity_score ?? 0;
  const c = paper.criteria_scores;
  if (!c) return 0;
  if (key === "problem") return c.problem_similarity;
  if (key === "method") return c.method_similarity;
  if (key === "domain") return c.domain_overlap;
  return c.contribution_similarity;
}

function parseReasons(reason: string | undefined): Record<string, string> {
  if (!reason) return {};
  const map: Record<string, string> = {};
  reason.split(" | ").forEach((part) => {
    const colonIdx = part.indexOf(":");
    if (colonIdx > -1) {
      map[part.slice(0, colonIdx).trim().toLowerCase()] = part.slice(colonIdx + 1).trim();
    }
  });
  return map;
}

export default function PaperTable({ papers, originalityScore }: Props) {
  const [sortKey, setSortKey] = useState<SortKey>("overall");
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const validPapers = papers.filter((p) => p.paper_similarity_score !== undefined);
  const sorted = [...validPapers].sort((a, b) => getScore(b, sortKey) - getScore(a, sortKey));

  // Detect if analyzing GitHub repos
  const isGitHubSource = papers[0]?.source === "github";

  return (
    <div className="rounded-2xl overflow-hidden border border-slate-200 bg-white shadow-sm">

      {/* ── Header ── */}
      <div className="px-6 pt-5 pb-4 border-b border-slate-100">
        <div className="flex items-start justify-between gap-4 flex-wrap">
          <div>
            <h3 className="text-base font-semibold text-slate-900 tracking-tight">
              {isGitHubSource ? "Analyzed Repositories" : "Analyzed Papers"}
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              {sorted.length} {isGitHubSource ? "repo" : "paper"}{sorted.length !== 1 ? "s" : ""} · click any row to expand
            </p>
          </div>
          {originalityScore !== undefined && (
            <div className="flex items-center gap-3 bg-slate-50 border border-slate-200 rounded-xl px-4 py-2.5">
              <div className="text-right">
                <p className="text-[10px] font-semibold text-slate-400 uppercase tracking-widest">Global Originality</p>
                <p className={`text-2xl font-bold leading-none mt-0.5 ${originalityColor(originalityScore)}`}>
                  {originalityScore}
                  <span className="text-sm font-medium text-slate-400 ml-0.5">/100</span>
                </p>
              </div>
              <OriginArc score={originalityScore} />
            </div>
          )}
        </div>

        {/* Sort controls */}
        <div className="flex items-center gap-1.5 mt-4 flex-wrap">
          <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-widest mr-1">Sort</span>
          {SORT_OPTIONS.map(({ key, short }) => (
            <button
              key={key}
              onClick={() => setSortKey(key)}
              className={`text-xs px-3 py-1.5 rounded-lg font-medium transition-all ${
                sortKey === key
                  ? "bg-slate-900 text-white shadow-sm"
                  : "bg-slate-100 text-slate-500 hover:bg-slate-200 hover:text-slate-700"
              }`}
            >
              {short}
            </button>
          ))}
        </div>
      </div>

      {/* ── Paper list ── */}
      <div className="divide-y divide-slate-100">
        {sorted.map((paper, idx) => {
          const isExpanded = expandedId === paper.paper_id;
          const overall = paper.paper_similarity_score ?? 0;
          const oc = overallColor(overall);

          return (
            <div key={paper.paper_id} className={isExpanded ? "bg-slate-50/50" : ""}>
              {/* Row */}
              <div
                className="px-6 py-5 cursor-pointer hover:bg-slate-50/80 transition-colors"
                onClick={() => setExpandedId(isExpanded ? null : paper.paper_id)}
              >
                <div className="flex items-start gap-4">
                  {/* Rank */}
                  <div className="hidden sm:flex w-7 h-7 rounded-full bg-slate-100 text-slate-400 text-xs font-bold items-center justify-center flex-shrink-0 mt-0.5">
                    {idx + 1}
                  </div>

                  {/* Content */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-start justify-between gap-3">
                      <div className="min-w-0 flex-1">
                        <p className="text-base font-semibold text-slate-800 leading-snug line-clamp-2">
                          {paper.title}
                        </p>
                        <div className="flex items-center gap-2 mt-1.5 flex-wrap">
                          {paper.authors.length > 0 && (
                            <span className="text-[11px] text-slate-500">
                              {paper.authors[0]}{paper.authors.length > 1 && ` +${paper.authors.length - 1}`}
                            </span>
                          )}
                          {paper.categories.slice(0, 2).map((cat) => (
                            <span key={cat} className="text-[10px] px-1.5 py-0.5 bg-slate-100 text-slate-500 rounded font-mono">
                              {cat}
                            </span>
                          ))}
                        </div>
                      </div>

                      {/* Overall score + expand indicator */}
                      <div className="flex items-center gap-2 flex-shrink-0">
                        <span className={`text-base font-bold px-3 py-1.5 rounded-xl ${oc.text} ${oc.bg}`}>
                          {Math.round(overall * 100)}%
                        </span>
                        <span className="text-slate-300 text-sm">{isExpanded ? "▲" : "▼"}</span>
                      </div>
                    </div>

                    {/* Links */}
                    <div className="flex items-center gap-3 mt-2">
                      {paper.source === "github" ? (
                        <a
                          href={paper.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          onClick={(e) => e.stopPropagation()}
                          className="text-xs text-slate-400 hover:text-slate-700 font-medium underline underline-offset-2 decoration-slate-200 hover:decoration-slate-500 transition-colors"
                        >
                          GitHub ↗
                        </a>
                      ) : (
                        <>
                          <a
                            href={paper.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            onClick={(e) => e.stopPropagation()}
                            className="text-xs text-slate-400 hover:text-slate-700 font-medium underline underline-offset-2 decoration-slate-200 hover:decoration-slate-500 transition-colors"
                          >
                            arXiv ↗
                          </a>
                          <a
                            href={paper.pdf_url}
                            target="_blank"
                            rel="noopener noreferrer"
                            onClick={(e) => e.stopPropagation()}
                            className="text-xs text-slate-400 hover:text-slate-700 font-medium underline underline-offset-2 decoration-slate-200 hover:decoration-slate-500 transition-colors"
                          >
                            PDF ↗
                          </a>
                        </>
                      )}
                    </div>
                  </div>
                </div>
              </div>

              {/* Expanded panel */}
              {isExpanded && <PaperDetailPanel paper={paper} />}
            </div>
          );
        })}
      </div>

      {sorted.length === 0 && (
        <div className="px-6 py-12 text-center text-slate-400 text-sm">
          No papers with analysis data available.
        </div>
      )}
    </div>
  );
}

/** Small SVG arc showing originality score */
function OriginArc({ score }: { score: number }) {
  const r = 18;
  const circ = 2 * Math.PI * r;
  const dash = (score / 100) * circ;
  const color = score >= 70 ? "#10b981" : score >= 40 ? "#f59e0b" : "#f43f5e";
  return (
    <svg width="44" height="44" viewBox="0 0 44 44" className="-rotate-90">
      <circle cx="22" cy="22" r={r} fill="none" stroke="#e2e8f0" strokeWidth="3.5" />
      <circle
        cx="22" cy="22" r={r}
        fill="none"
        stroke={color}
        strokeWidth="3.5"
        strokeDasharray={`${dash} ${circ}`}
        strokeLinecap="round"
      />
    </svg>
  );
}

function PaperDetailPanel({ paper }: { paper: PaperDetail }) {
  const [expandedCriterion, setExpandedCriterion] = useState<string | null>(null);
  const c = paper.criteria_scores;
  const reasons = parseReasons(paper.reason);

  const criteria = c
    ? [
        { label: "Problem Similarity", reasonKey: "problem", score: c.problem_similarity, desc: "How closely the research problem matches your idea." },
        { label: "Method Similarity", reasonKey: "method", score: c.method_similarity, desc: "Overlap in approach, algorithms, or techniques." },
        { label: "Domain Overlap", reasonKey: "domain", score: c.domain_overlap, desc: "Overlap in the application area or field of study." },
        { label: "Contribution Similarity", reasonKey: "contribution", score: c.contribution_similarity, desc: "How similar the claimed contributions are." },
      ]
    : [];

  return (
    <div className="mx-4 mb-4 rounded-xl border border-slate-200 bg-white overflow-hidden shadow-sm">
      {/* Abstract */}
      <div className="px-6 py-5 border-b border-slate-100">
        <p className="text-xs font-semibold text-slate-400 uppercase tracking-widest mb-3">Abstract</p>
        <p className="text-sm text-slate-600 leading-relaxed line-clamp-4">{paper.abstract}</p>
      </div>

      {/* Criteria breakdown */}
      {criteria.length > 0 && (
        <div className="px-6 py-5">
          <p className="text-xs font-semibold text-slate-400 uppercase tracking-widest mb-4">
            Criteria Breakdown
            {Object.keys(reasons).length > 0 && (
              <span className="ml-2 font-normal normal-case text-slate-400">· click a card for details</span>
            )}
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {criteria.map(({ label, reasonKey, score, desc }) => {
              const likert = toLikert(score);
              const lc = likertColor(likert);
              const isExp = expandedCriterion === reasonKey;
              const reasonText = reasons[reasonKey];

              return (
                <div
                  key={label}
                  onClick={() => reasonText && setExpandedCriterion(isExp ? null : reasonKey)}
                  className={`rounded-xl border p-4 transition-all ${
                    reasonText ? "cursor-pointer" : ""
                  } ${isExp ? `${lc.border} ${lc.bg}` : "border-slate-100 bg-slate-50 hover:border-slate-200"}`}
                >
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-sm font-semibold text-slate-700">{label}</span>
                    <div className="flex items-center gap-2.5">
                      <div className="flex items-center gap-1">
                        {[1, 2, 3, 4, 5].map((d) => (
                          <div
                            key={d}
                            className={`w-2.5 h-2.5 rounded-full transition-colors ${d <= likert ? barColor(score) : "bg-slate-200"}`}
                          />
                        ))}
                      </div>
                      <span className={`text-sm font-bold ${lc.text}`}>{likert}/5</span>
                      {reasonText && (
                        <span className="text-xs text-slate-400">{isExp ? "▲" : "▼"}</span>
                      )}
                    </div>
                  </div>
                  <p className={`text-sm leading-relaxed ${isExp && reasonText ? "text-slate-700" : "text-slate-400"}`}>
                    {isExp && reasonText ? reasonText : desc}
                  </p>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
