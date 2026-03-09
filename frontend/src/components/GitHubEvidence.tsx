import { useState } from "react";
import type { GitHubAnalysis, RepoRelevanceResult } from "../types/api";

interface Props {
  analysis: GitHubAnalysis;
}

const VERDICT_CONFIG = {
  pursue_as_is: {
    label: "Pursue As-Is",
    color: "text-green-700",
    bg: "bg-green-50",
    border: "border-green-200",
    icon: "✅",
  },
  refine_scope: {
    label: "Refine Scope",
    color: "text-amber-700",
    bg: "bg-amber-50",
    border: "border-amber-200",
    icon: "🔧",
  },
  reconsider: {
    label: "Reconsider",
    color: "text-red-700",
    bg: "bg-red-50",
    border: "border-red-200",
    icon: "⚠️",
  },
};

const REPO_VERDICT_STYLES: Record<string, { bg: string; text: string; dot: string }> = {
  strong_overlap: { bg: "bg-red-100", text: "text-red-700", dot: "bg-red-500" },
  partial_overlap: { bg: "bg-amber-100", text: "text-amber-700", dot: "bg-amber-500" },
  tangential: { bg: "bg-blue-100", text: "text-blue-700", dot: "bg-blue-500" },
  unrelated: { bg: "bg-slate-100", text: "text-slate-500", dot: "bg-slate-400" },
};

function formatVerdict(v: string): string {
  return v.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

export default function GitHubEvidence({ analysis }: Props) {
  const [expanded, setExpanded] = useState(false);
  const verdict = VERDICT_CONFIG[analysis.verdict] ?? VERDICT_CONFIG.pursue_as_is;
  const relevant = analysis.repo_results.filter((r) => r.verdict !== "unrelated");

  return (
    <div className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden">
      {/* Header */}
      <div className="px-6 py-5 border-b border-slate-100">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2.5">
            <svg className="w-5 h-5 text-slate-700" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z" />
            </svg>
            <h3 className="text-sm font-semibold text-slate-700 uppercase tracking-wider">
              GitHub Evidence
            </h3>
          </div>
          <div className={`flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold ${verdict.bg} ${verdict.color} ${verdict.border} border`}>
            <span>{verdict.icon}</span>
            <span>{verdict.label}</span>
          </div>
        </div>

        <p className="text-sm text-slate-600 leading-relaxed">{analysis.synthesis}</p>

        <div className="flex items-center gap-4 mt-3 text-xs text-slate-400">
          <span>{analysis.repos_analyzed} repos analyzed</span>
          <span className="text-slate-300">·</span>
          <span>{analysis.repos_relevant} relevant</span>
        </div>
      </div>

      {/* Collapsible repo list */}
      {relevant.length > 0 && (
        <div>
          <button
            onClick={() => setExpanded(!expanded)}
            className="w-full flex items-center justify-between px-6 py-3 text-sm text-slate-600 hover:bg-slate-50 transition-colors"
          >
            <span className="font-medium">
              {expanded ? "Hide" : "Show"} repository details ({relevant.length})
            </span>
            <svg
              className={`w-4 h-4 text-slate-400 transition-transform duration-200 ${expanded ? "rotate-180" : ""}`}
              fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}
            >
              <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
            </svg>
          </button>

          {expanded && (
            <div className="border-t border-slate-100 divide-y divide-slate-100">
              {relevant.map((repo) => (
                <RepoCard key={repo.repo_full_name} repo={repo} />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function RepoCard({ repo }: { repo: RepoRelevanceResult }) {
  const style = REPO_VERDICT_STYLES[repo.verdict] ?? REPO_VERDICT_STYLES.unrelated;
  const overlapPct = Math.round(repo.overlap_score * 100);

  return (
    <div className="px-6 py-4">
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2 mb-1">
            <a
              href={repo.repo_url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm font-semibold text-indigo-600 hover:text-indigo-800 hover:underline truncate"
            >
              {repo.repo_full_name}
            </a>
            <span className="flex items-center gap-1 text-xs text-slate-400 flex-shrink-0">
              <svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 24 24">
                <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z" />
              </svg>
              {repo.stars.toLocaleString()}
            </span>
          </div>

          {repo.description && (
            <p className="text-xs text-slate-500 mb-2 line-clamp-1">{repo.description}</p>
          )}

          <div className="space-y-1">
            <p className="text-xs text-slate-600">
              <span className="font-medium text-slate-500">Covers:</span> {repo.what_it_covers}
            </p>
            <p className="text-xs text-slate-600">
              <span className="font-medium text-emerald-600">Gap:</span> {repo.what_it_misses}
            </p>
          </div>
        </div>

        <div className="flex flex-col items-end gap-2 flex-shrink-0">
          <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-bold ${style.bg} ${style.text}`}>
            <span className={`w-1.5 h-1.5 rounded-full ${style.dot}`} />
            {formatVerdict(repo.verdict)}
          </span>
          <div className="flex items-center gap-1.5">
            <div className="w-16 h-1.5 bg-slate-100 rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full ${overlapPct >= 70 ? "bg-red-500" : overlapPct >= 40 ? "bg-amber-500" : "bg-green-500"}`}
                style={{ width: `${overlapPct}%` }}
              />
            </div>
            <span className="text-[10px] font-bold text-slate-500">{overlapPct}%</span>
          </div>
        </div>
      </div>

      {repo.topics.length > 0 && (
        <div className="flex items-center gap-1 mt-2 flex-wrap">
          {repo.topics.slice(0, 5).map((t) => (
            <span key={t} className="text-[10px] px-1.5 py-0.5 bg-slate-100 text-slate-500 rounded">
              {t}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
