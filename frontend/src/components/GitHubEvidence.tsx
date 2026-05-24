import { useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import type { GitHubAnalysis, RepoRelevanceResult } from "../types/api";

interface Props {
  analysis: GitHubAnalysis;
}

const VERDICT: Record<
  string,
  { label: string; accent: string; glyph: string }
> = {
  pursue_as_is: {
    label: "Pursue as written",
    accent: "var(--color-moss)",
    glyph: "✓",
  },
  refine_scope: {
    label: "Refine the scope",
    accent: "var(--color-ochre)",
    glyph: "✦",
  },
  reconsider: {
    label: "Reconsider the work",
    accent: "var(--color-vermillion)",
    glyph: "⚠",
  },
};

const REPO_VERDICT: Record<string, { label: string; accent: string }> = {
  strong_similarity: {
    label: "strong similarity",
    accent: "var(--color-vermillion)",
  },
  partial_similarity: { label: "partial similarity", accent: "var(--color-ochre)" },
  tangential: { label: "tangential", accent: "var(--color-gold)" },
  unrelated: { label: "unrelated", accent: "var(--color-ink-mute)" },
};

export default function GitHubEvidence({ analysis }: Props) {
  const [expanded, setExpanded] = useState(false);
  const v = VERDICT[analysis.verdict] ?? VERDICT.pursue_as_is;
  const relevant = analysis.repo_results.filter(
    (r) => r.verdict !== "unrelated"
  );

  return (
    <div className="border border-[color:var(--color-rule)] bg-[color:var(--color-paper-shade)]">
      <div className="px-6 py-5 border-b border-[color:var(--color-rule)]">
        <div className="flex items-start justify-between gap-4 flex-wrap">
          <div>
            <p className="small-caps text-[color:var(--color-ink-fade)] mb-1">
              appendix B
            </p>
            <h3 className="font-display text-2xl tracking-tight">
              Implementation Evidence
            </h3>
            <p className="font-display italic text-sm text-[color:var(--color-ink-fade)]">
              what the open-source archive already contains
            </p>
          </div>
          <div
            className="border-l-2 pl-3 py-1"
            style={{ borderColor: v.accent }}
          >
            <p
              className="small-caps"
              style={{ color: v.accent }}
            >
              editorial verdict
            </p>
            <p
              className="font-display text-lg leading-tight"
              style={{ color: v.accent }}
            >
              {v.glyph} {v.label}
            </p>
          </div>
        </div>

        <p className="font-body text-[15px] leading-relaxed text-[color:var(--color-ink-soft)] mt-4">
          {analysis.synthesis}
        </p>

        <p className="mt-3 font-mono text-[11px] text-[color:var(--color-ink-fade)]">
          {analysis.repos_analyzed} repos canvassed ·{" "}
          {analysis.repos_relevant} held relevant
        </p>
      </div>

      {relevant.length > 0 && (
        <>
          <button
            onClick={() => setExpanded(!expanded)}
            className="w-full px-6 py-3 text-left small-caps text-[color:var(--color-ink-fade)] hover:text-[color:var(--color-ink)] hover:bg-[color:var(--color-paper)] transition-colors flex items-center justify-between"
          >
            <span>
              <span className="font-mono mr-2">
                {expanded ? "–" : "+"}
              </span>
              repository roster ({relevant.length})
            </span>
            <span className="font-mono text-xs">
              {expanded ? "▴" : "▾"}
            </span>
          </button>
          <AnimatePresence>
            {expanded && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: "auto", opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.4 }}
                className="overflow-hidden border-t border-[color:var(--color-rule)]"
              >
                <ol className="divide-y divide-[color:var(--color-rule)]">
                  {relevant.map((repo) => (
                    <RepoEntry key={repo.repo_full_name} repo={repo} />
                  ))}
                </ol>
              </motion.div>
            )}
          </AnimatePresence>
        </>
      )}
    </div>
  );
}

function RepoEntry({ repo }: { repo: RepoRelevanceResult }) {
  const v = REPO_VERDICT[repo.verdict] ?? REPO_VERDICT.unrelated;
  const pct = Math.round(repo.similarity_score * 100);

  return (
    <li className="px-6 py-4">
      <div className="grid grid-cols-[1fr_auto] gap-4">
        <div className="min-w-0">
          <div className="flex items-baseline gap-3 flex-wrap mb-1">
            <a
              href={repo.repo_url}
              target="_blank"
              rel="noopener noreferrer"
              className="font-display italic text-lg text-[color:var(--color-ink)] hover:text-[color:var(--color-vermillion)] underline underline-offset-2 decoration-1 decoration-[color:var(--color-rule)] hover:decoration-[color:var(--color-vermillion)]"
            >
              {repo.repo_full_name}
            </a>
            <span className="font-mono text-[10px] text-[color:var(--color-ink-fade)]">
              ★ {repo.stars.toLocaleString()}
            </span>
          </div>
          {repo.description && (
            <p className="font-body text-sm text-[color:var(--color-ink-soft)] mb-2 line-clamp-2">
              {repo.description}
            </p>
          )}
          <div className="font-body text-xs space-y-1">
            <p>
              <span className="small-caps text-[color:var(--color-ink-fade)] mr-2">
                covers
              </span>
              <span className="text-[color:var(--color-ink-soft)]">
                {repo.what_it_covers}
              </span>
            </p>
            <p>
              <span
                className="small-caps mr-2"
                style={{ color: "var(--color-moss)" }}
              >
                gap
              </span>
              <span className="text-[color:var(--color-ink-soft)]">
                {repo.what_it_misses}
              </span>
            </p>
          </div>
          {repo.topics.length > 0 && (
            <div className="mt-2 flex flex-wrap gap-2 font-mono text-[10px] text-[color:var(--color-ink-fade)]">
              {repo.topics.slice(0, 6).map((t) => (
                <span key={t} className="lowercase">
                  · {t}
                </span>
              ))}
            </div>
          )}
        </div>
        <div className="text-right flex flex-col items-end gap-1">
          <span
            className="small-caps"
            style={{ color: v.accent }}
          >
            {v.label}
          </span>
          <span
            className="font-display text-2xl leading-none numeric-tabular"
            style={{ color: v.accent }}
          >
            {pct}%
          </span>
          <div className="w-20 h-px bg-[color:var(--color-rule)] mt-1">
            <div
              className="h-full"
              style={{
                width: `${pct}%`,
                background: v.accent,
                height: "3px",
              }}
            />
          </div>
        </div>
      </div>
    </li>
  );
}
