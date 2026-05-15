import { useState } from "react";
import { motion } from "motion/react";
import type { AnalysisResults, SentenceAnnotation } from "../types/api";
import OriginalityGauge from "./OriginalityGauge";
import CriteriaBreakdown from "./CriteriaBreakdown";
import HighlightedIdea from "./HighlightedIdea";
import MatchesModal from "./MatchesModal";
import PaperTable from "./PaperTable";
import GitHubEvidence from "./GitHubEvidence";
import { evidenceStatsLabels } from "../lib/evidenceLabels";

interface RealityCheckInfo {
  warning: string | null;
  result: Record<string, unknown>;
}

interface Props {
  results: AnalysisResults;
  jobId: string;
  realityCheck: RealityCheckInfo | null;
  onNewAnalysis: () => void;
}

export default function ResultsView({
  results,
  jobId,
  realityCheck,
  onNewAnalysis,
}: Props) {
  const [selectedSentence, setSelectedSentence] =
    useState<SentenceAnnotation | null>(null);

  const evidenceSource = results.papers?.[0]?.source;
  const statLabels = evidenceStatsLabels(evidenceSource);
  const githubEvidence = results.github_result;

  const reportText = [
    `# Originality Assessment Report\n`,
    `## Score: ${results.originality_score}/100\n`,
    `## Summary\n${results.comprehensive_report || results.summary}\n`,
    `## Sentence Analysis`,
    ...results.sentence_annotations.map((ann) => {
      const emoji =
        ann.label === "high" ? "🟢" : ann.label === "medium" ? "🟡" : "🔴";
      return `${emoji} [${Math.round(ann.similarity_score * 100)}% similarity] ${ann.sentence}`;
    }),
  ].join("\n");

  return (
    <>
      {/* Issue header */}
      <div className="grid grid-cols-12 gap-6 mb-12 items-stretch">
        <div className="col-span-12 md:col-span-7 flex flex-col justify-between gap-8">
          <div>
            <p className="small-caps text-[color:var(--color-ink-fade)] mb-2">
              Issue № {jobId.slice(0, 6)}
            </p>
            <h2 className="font-display text-[clamp(2.6rem,6vw,4.5rem)] tracking-tight leading-[0.92] text-[color:var(--color-ink)]">
              The Verdict
            </h2>
            <p className="font-body text-[color:var(--color-ink-soft)] text-base mt-2 max-w-md leading-snug">
              assessed against {results.papers?.length ?? 0} sources of
              prior art
            </p>
          </div>

          <div className="hairline" />

          <dl className="grid grid-cols-2 gap-x-6 gap-y-6">
            <HeaderStat
              label="prior art consulted"
              value={(results.papers?.length ?? 0).toString().padStart(2, "0")}
              hint={
                results.stats
                  ? `from ${results.stats.total_fetched.toLocaleString()} fetched`
                  : undefined
              }
            />
            <HeaderStat
              label="criteria assessed"
              value="04"
              hint="problem · method · domain · contribution"
            />
            <HeaderStat
              label="processing time"
              value={formatDuration(results.total_processing_time)}
              hint={
                results.stats
                  ? `${results.stats.total_chunks.toLocaleString()} chunks indexed`
                  : undefined
              }
            />
            <HeaderStat
              label="cost of assessment"
              value={`$${results.cost.estimated_cost_usd.toFixed(3)}`}
              hint="across all agents"
            />
          </dl>
        </div>
        <div className="col-span-12 md:col-span-5">
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7 }}
            className="border border-[color:var(--color-rule)] bg-[color:var(--color-paper-shade)] p-6 h-full"
          >
            <OriginalityGauge
              score={results.originality_score}
              summary={results.summary}
            />
          </motion.div>
        </div>
      </div>

      <span className="hairline-double block mb-10" />

      {/* The manuscript */}
      <section className="grid grid-cols-12 gap-6 mb-14">
        <aside className="hidden lg:block col-span-3 pt-2">
          <p className="small-caps text-[color:var(--color-ink-fade)] mb-2">
            chapter iii
          </p>
          <p className="font-display italic text-xl text-[color:var(--color-ink-soft)] leading-tight">
            The manuscript, annotated.
          </p>
          <div className="hairline my-4" />
          <p className="font-mono text-[11px] text-[color:var(--color-ink-fade)] leading-relaxed">
            Vermillion underlines mark<br />
            <span className="text-[color:var(--color-vermillion)]">high similarity</span>;{" "}
            ochre marks<br />
            <span className="text-[color:var(--color-ochre)]">moderate similarity</span>.<br /><br />
            Click any mark to see<br />
            the matching passage<br />
            in the source.
          </p>
        </aside>
        <div className="col-span-12 lg:col-span-9">
          <h3 className="font-display text-3xl tracking-tight mb-1">
            Author's Submission, with Annotation
          </h3>
          <p className="font-display italic text-[color:var(--color-ink-fade)] mb-5 text-sm">
            Hypothetica marks where the corpus knows your idea
          </p>

          <div className="relative bg-[color:var(--color-paper-shade)] border border-[color:var(--color-rule)] p-7 md:p-9">
            <span className="absolute top-2 right-3 font-mono text-[10px] text-[color:var(--color-ink-fade)]">
              ✦
            </span>
            <HighlightedIdea
              annotations={results.sentence_annotations}
              jobId={jobId}
            />
          </div>
        </div>
      </section>

      <span className="hairline-double block mb-10" />

      {/* Apparatus + bibliography */}
      <div className="grid grid-cols-12 gap-x-6 gap-y-10">
        <aside className="col-span-12 lg:col-span-4 space-y-6">
          {/* Criteria */}
          {results.aggregated_criteria && (
            <section className="bg-[color:var(--color-paper-shade)]/70 border border-[color:var(--color-rule)] p-6">
              <CriteriaBreakdown criteria={results.aggregated_criteria} />
            </section>
          )}

          {/* Stats */}
          {results.stats && (
            <section className="bg-[color:var(--color-paper-shade)]/70 border border-[color:var(--color-rule)] p-6">
              <h3 className="font-display text-2xl tracking-tight mb-1">
                Statistics
              </h3>
              <p className="font-body text-[14px] text-[color:var(--color-ink-soft)] mb-4 leading-snug">
                retrieval and indexing metrics
              </p>
              <dl className="grid grid-cols-2 gap-x-4 gap-y-4 mb-3">
                <Stat
                  label={statLabels.analyzed}
                  value={results.stats.papers_processed}
                />
                <Stat
                  label="Chunks Indexed"
                  value={results.stats.total_chunks}
                />
                <Stat
                  label="Query Variants"
                  value={results.stats.query_variants}
                />
                <Stat
                  label={statLabels.fetched}
                  value={results.stats.total_fetched}
                />
              </dl>
              <SearchQueriesList stats={results.stats} />
            </section>
          )}

          {/* Cost */}
          <section className="bg-[color:var(--color-paper-shade)]/70 border border-[color:var(--color-rule)] p-6">
            <h3 className="font-display text-2xl tracking-tight mb-1">
              Cost of Assessment
            </h3>
            <p className="font-body text-[14px] text-[color:var(--color-ink-soft)] mb-4 leading-snug">
              token spend across every agent
            </p>
            <dl className="font-mono text-[12px] divide-y divide-[color:var(--color-rule)]">
              {Object.entries(results.cost.breakdown).map(([key, val]) => (
                <div key={key} className="flex items-baseline justify-between py-2">
                  <dt className="text-[color:var(--color-ink-soft)] uppercase tracking-wider font-bold">
                    {key.replace(/_/g, " ")}
                  </dt>
                  <dd className="text-[color:var(--color-ink)] numeric-tabular">
                    ${val.toFixed(4)}
                  </dd>
                </div>
              ))}
              <div className="flex items-baseline justify-between py-2 border-t-[3px] border-double border-[color:var(--color-ink)] mt-1">
                <dt className="small-caps text-[color:var(--color-ink)] font-bold">total</dt>
                <dd className="font-display text-xl text-[color:var(--color-vermillion)] numeric-tabular">
                  ${results.cost.estimated_cost_usd.toFixed(4)}
                </dd>
              </div>
            </dl>
          </section>
        </aside>

        <div className="col-span-12 lg:col-span-8 space-y-8">
          {githubEvidence && <GitHubEvidence analysis={githubEvidence} />}
          {results.papers && results.papers.length > 0 && (
            <PaperTable
              papers={results.papers}
              originalityScore={results.originality_score}
            />
          )}
        </div>
      </div>

      {/* Actions */}
      <div className="mt-16 flex flex-col md:flex-row items-start md:items-center gap-4">
        <button
          onClick={onNewAnalysis}
          className="group relative inline-flex items-center gap-3 px-7 py-3.5 bg-[color:var(--color-ink)] text-[color:var(--color-paper)] font-display text-lg tracking-tight shadow-[0_5px_0_0_var(--color-vermillion)] hover:shadow-[0_2px_0_0_var(--color-vermillion)] hover:translate-y-[3px] transition-all"
        >
          <span className="font-mono text-sm opacity-70 group-hover:rotate-180 transition-transform duration-500">
            ↺
          </span>
          <span>Submit Another</span>
        </button>
        <a
          href={`data:text/markdown;charset=utf-8,${encodeURIComponent(reportText)}`}
          download="originality_report.md"
          className="group inline-flex items-center gap-2 px-5 py-3 border border-[color:var(--color-ink)] text-[color:var(--color-ink)] small-caps font-bold hover:bg-[color:var(--color-ink)] hover:text-[color:var(--color-paper)] transition-colors"
        >
          <span className="font-mono group-hover:translate-y-0.5 transition-transform">↓</span>
          <span>download .md</span>
        </a>
      </div>

      {selectedSentence && (
        <MatchesModal
          annotation={selectedSentence}
          jobId={jobId}
          onClose={() => setSelectedSentence(null)}
        />
      )}
    </>
  );
}

function HeaderStat({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  hint?: string;
}) {
  return (
    <div className="flex flex-col">
      <span className="small-caps text-[color:var(--color-ink-soft)] font-bold order-2 mt-1">
        {label}
      </span>
      <span
        className="font-display text-[clamp(2rem,4vw,2.75rem)] leading-none text-[color:var(--color-ink)] numeric-tabular order-1"
        style={{ fontVariationSettings: '"opsz" 144, "SOFT" 30, "WONK" 1' }}
      >
        {value}
      </span>
      {hint && (
        <span className="font-body text-[13px] text-[color:var(--color-ink-soft)] order-3 mt-1 leading-snug">
          {hint}
        </span>
      )}
    </div>
  );
}

function formatDuration(seconds: number): string {
  if (!seconds || seconds < 0) return "·";
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return `${m}m ${s.toString().padStart(2, "0")}s`;
}

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div className="flex flex-col">
      <dt className="small-caps text-[color:var(--color-ink-soft)] font-bold order-2 mt-1">
        {label}
      </dt>
      <dd className="font-display text-3xl text-[color:var(--color-ink)] numeric-tabular order-1">
        {value.toLocaleString()}
      </dd>
    </div>
  );
}

function SearchQueriesList({
  stats,
}: {
  stats: NonNullable<AnalysisResults["stats"]>;
}) {
  const raw =
    stats.query_variant_strings && stats.query_variant_strings.length > 0
      ? stats.query_variant_strings
      : (stats.query_variants_list?.map((v) => v.query).filter(Boolean) as
          | string[]
          | undefined) ?? [];
  if (raw.length === 0) return null;
  return (
    <details className="mt-4 group border-t border-[color:var(--color-rule)] pt-3">
      <summary className="small-caps text-[color:var(--color-ink-soft)] font-bold cursor-pointer hover:text-[color:var(--color-vermillion)] transition-colors list-none flex items-center gap-2">
        <span className="font-mono">+</span>
        retrieval queries ({raw.length})
      </summary>
      <ol className="mt-3 space-y-2 font-mono text-[11px] text-[color:var(--color-ink-soft)]">
        {raw.map((q, i) => (
          <li key={i} className="flex gap-3">
            <span className="text-[color:var(--color-ink-fade)] flex-shrink-0">
              {(i + 1).toString().padStart(2, "0")}.
            </span>
            <span className="break-words">{q}</span>
          </li>
        ))}
      </ol>
    </details>
  );
}

