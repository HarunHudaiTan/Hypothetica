import { useState } from "react";
import type { PaperDetail } from "../types/api";

interface Props {
  papers: PaperDetail[];
}

function overlapColor(score: number): string {
  if (score >= 0.7) return "text-red-600 bg-red-50";
  if (score >= 0.4) return "text-amber-600 bg-amber-50";
  return "text-green-600 bg-green-50";
}

function overlapBarColor(score: number): string {
  if (score >= 0.7) return "bg-red-500";
  if (score >= 0.4) return "bg-amber-500";
  return "bg-green-500";
}

export default function PaperTable({ papers }: Props) {
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const sorted = [...papers]
    .filter((p) => p.idea_similarity_score !== undefined)
    .sort((a, b) => (b.idea_similarity_score ?? 0) - (a.idea_similarity_score ?? 0));

  return (
    <div className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden">
      <div className="px-6 py-4 border-b border-slate-100">
        <h3 className="text-sm font-semibold text-slate-700 uppercase tracking-wider">
          Analyzed Papers
        </h3>
        <p className="text-xs text-slate-400 mt-1">
          Sorted by similarity — click a row to see per-criteria breakdown
        </p>
      </div>

      {/* Table header */}
      <div className="hidden sm:grid grid-cols-12 gap-2 px-6 py-2 bg-slate-50 text-xs font-semibold text-slate-500 uppercase tracking-wider border-b border-slate-100">
        <div className="col-span-6">Paper</div>
        <div className="col-span-3 text-center">Similarity</div>
        <div className="col-span-3 text-center">Links</div>
      </div>

      {/* Rows */}
      <div className="divide-y divide-slate-100">
        {sorted.map((paper) => {
          const overlap = paper.idea_similarity_score ?? 0;
          const pct = Math.round(overlap * 100);
          const isExpanded = expandedId === paper.paper_id;

          return (
            <div key={paper.paper_id}>
              {/* Main row */}
              <div
                className="grid grid-cols-1 sm:grid-cols-12 gap-2 px-6 py-4 hover:bg-slate-50 cursor-pointer transition-colors items-center"
                onClick={() =>
                  setExpandedId(isExpanded ? null : paper.paper_id)
                }
              >
                {/* Title + meta */}
                <div className="sm:col-span-6 min-w-0">
                  <p className="text-sm font-medium text-slate-800 leading-snug line-clamp-2">
                    {paper.title}
                  </p>
                  <div className="flex items-center gap-2 mt-1">
                    {paper.categories.slice(0, 2).map((cat) => (
                      <span
                        key={cat}
                        className="text-[10px] px-1.5 py-0.5 bg-slate-100 text-slate-500 rounded"
                      >
                        {cat}
                      </span>
                    ))}
                    {paper.authors.length > 0 && (
                      <span className="text-[10px] text-slate-400 truncate">
                        {paper.authors[0]}
                        {paper.authors.length > 1 &&
                          ` +${paper.authors.length - 1}`}
                      </span>
                    )}
                  </div>
                </div>

                {/* Similarity bar + number */}
                <div className="sm:col-span-3 flex items-center gap-3 mt-2 sm:mt-0 justify-center">
                  <div className="flex-1 max-w-[100px] h-2 bg-slate-100 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all ${overlapBarColor(overlap)}`}
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                  <span
                    className={`text-xs font-bold px-2 py-0.5 rounded ${overlapColor(overlap)}`}
                  >
                    {pct}%
                  </span>
                </div>

                {/* Links */}
                <div className="sm:col-span-3 flex items-center gap-2 mt-2 sm:mt-0 justify-center">
                  <a
                    href={paper.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    onClick={(e) => e.stopPropagation()}
                    className="text-xs text-indigo-600 hover:text-indigo-800 font-medium hover:underline"
                  >
                    arXiv
                  </a>
                  <span className="text-slate-300">·</span>
                  <a
                    href={paper.pdf_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    onClick={(e) => e.stopPropagation()}
                    className="text-xs text-indigo-600 hover:text-indigo-800 font-medium hover:underline"
                  >
                    PDF
                  </a>
                  <span className="text-slate-300">·</span>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      setExpandedId(isExpanded ? null : paper.paper_id);
                    }}
                    className="text-xs text-slate-500 hover:text-slate-700"
                  >
                    {isExpanded ? "▲ Less" : "▼ More"}
                  </button>
                </div>
              </div>

              {/* Expanded detail */}
              {isExpanded && (
                <PaperDetailPanel paper={paper} />
              )}
            </div>
          );
        })}
      </div>

      {sorted.length === 0 && (
        <div className="px-6 py-8 text-center text-slate-400 text-sm">
          No papers with analysis data available.
        </div>
      )}
    </div>
  );
}

function PaperDetailPanel({ paper }: { paper: PaperDetail }) {
  const c = paper.criteria_scores;

  const criteria = c
    ? [
        {
          key: "problem_similarity",
          label: "Problem Similarity",
          icon: "🎯",
          score: c.problem_similarity,
          desc: "How closely the research problem or question matches your idea.",
        },
        {
          key: "method_similarity",
          label: "Method Similarity",
          icon: "⚙️",
          score: c.method_similarity,
          desc: "Overlap in the proposed approach, algorithms, or techniques.",
        },
        {
          key: "domain_overlap",
          label: "Domain Overlap",
          icon: "🌐",
          score: c.domain_overlap,
          desc: "Overlap in the application area or field of study.",
        },
        {
          key: "contribution_similarity",
          label: "Contribution Similarity",
          icon: "💡",
          score: c.contribution_similarity,
          desc: "How similar the claimed contributions and findings are.",
        },
      ]
    : [];

  return (
    <div className="px-6 pb-5 bg-slate-50 border-t border-slate-100">
      {/* Abstract */}
      <div className="pt-4 mb-4">
        <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1">
          Abstract
        </p>
        <p className="text-xs text-slate-600 leading-relaxed line-clamp-4">
          {paper.abstract}
        </p>
      </div>

      {/* Per-criteria bars */}
      {criteria.length > 0 && (
        <div>
          <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">
            Criteria Breakdown for this Paper
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {criteria.map(({ key, label, icon, score, desc }) => {
              const pct = Math.round(score * 100);
              return (
                <div
                  key={key}
                  className="bg-white rounded-lg border border-slate-200 p-3"
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-xs font-semibold text-slate-700">
                      {icon} {label}
                    </span>
                    <span
                      className={`text-xs font-bold px-1.5 py-0.5 rounded ${overlapColor(score)}`}
                    >
                      {pct}%
                    </span>
                  </div>
                  <div className="w-full h-1.5 bg-slate-100 rounded-full overflow-hidden mb-1.5">
                    <div
                      className={`h-full rounded-full ${overlapBarColor(score)}`}
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                  <p className="text-[11px] text-slate-400 leading-snug">
                    {desc}
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
