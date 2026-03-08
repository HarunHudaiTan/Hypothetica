import { useState } from "react";
import type { AnalysisResults, SentenceAnnotation } from "../types/api";
import OriginalityGauge from "./OriginalityGauge";
import CriteriaBreakdown from "./CriteriaBreakdown";
import SentenceHighlighting from "./SentenceHighlighting";
import HighlightedIdea from "./HighlightedIdea";
import MatchesModal from "./MatchesModal";
import PaperTable from "./PaperTable";

interface RealityCheckInfo {
  warning: string | null;
  result: Record<string, unknown>;
}

interface Props {
  results: AnalysisResults;
  jobId: string;
  userIdea: string;
  realityCheck: RealityCheckInfo | null;
  onNewAnalysis: () => void;
}

export default function ResultsView({
  results,
  jobId,
  userIdea,
  realityCheck,
  onNewAnalysis,
}: Props) {
  const [selectedSentence, setSelectedSentence] =
    useState<SentenceAnnotation | null>(null);

  const reportText = [
    `# Originality Assessment Report\n`,
    `## Score: ${results.global_originality_score}/100\n`,
    `## Summary\n${results.comprehensive_report || results.summary}\n`,
    `## Sentence Analysis`,
    ...results.sentence_annotations.map((ann) => {
      const emoji =
        ann.label === "high" ? "🟢" : ann.label === "medium" ? "🟡" : "🔴";
      return `${emoji} [${Math.round(ann.overlap_score * 100)}% overlap] ${ann.sentence}`;
    }),
  ].join("\n");

  return (
    <>
      {/* Reality check disclaimer — advisory only, does not affect score */}
      <RealityCheckBanner realityCheck={realityCheck} results={results} />

      {/* User Idea + Score Section */}
      <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6 mb-6">
        {/* User's Idea with highlighting */}
        <div className="mb-6">
          <h3 className="text-sm font-semibold text-slate-700 uppercase tracking-wider mb-3">
            Your Research Idea
            <span className="text-xs font-normal text-slate-400 ml-2">
              (highlighted areas show overlap with existing research)
            </span>
          </h3>
          <div className="bg-slate-50 rounded-xl p-4 border border-slate-100">
            <HighlightedIdea 
              annotations={results.sentence_annotations}
              jobId={jobId}
            />
          </div>
          {/* Legend */}
          <div className="flex items-center gap-4 mt-2 text-xs text-slate-500">
            <span className="flex items-center gap-1">
              <span className="w-3 h-2 bg-red-200 rounded-sm border-b-2 border-red-400" />
              High overlap
            </span>
            <span className="flex items-center gap-1">
              <span className="w-3 h-2 bg-amber-200 rounded-sm border-b-2 border-amber-400" />
              Moderate overlap
            </span>
            <span className="text-slate-400">Click highlighted text to see evidence</span>
          </div>
        </div>

        {/* Originality Gauge with Summary */}
        <OriginalityGauge 
          score={results.global_originality_score} 
          summary={results.summary}
        />
      </div>

      {/* Main results grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left column: Criteria + Stats + Cost */}
        <div className="space-y-6">

          {/* Stats */}
          {results.stats && (
            <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
              <h3 className="text-sm font-semibold text-slate-700 uppercase tracking-wider mb-3">
                Statistics
              </h3>
              <div className="grid grid-cols-2 gap-3">
                <Stat
                  label="Papers Analyzed"
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
                  label="Papers Fetched"
                  value={results.stats.total_fetched}
                />
              </div>
            </div>
          )}

          {/* Criteria */}
          {results.aggregated_criteria && (
            <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
              <CriteriaBreakdown criteria={results.aggregated_criteria} />
            </div>
          )}

          {/* Cost */}
          <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
            <h3 className="text-sm font-semibold text-slate-700 uppercase tracking-wider mb-3">
              Cost Breakdown
            </h3>
            <div className="space-y-2 text-sm">
              {Object.entries(results.cost.breakdown).map(([key, val]) => (
                <div key={key} className="flex justify-between">
                  <span className="text-slate-500 capitalize">{key}</span>
                  <span className="text-slate-700 font-medium">
                    ${val.toFixed(4)}
                  </span>
                </div>
              ))}
              <div className="border-t border-slate-100 pt-2 flex justify-between font-semibold">
                <span className="text-slate-700">Total</span>
                <span className="text-indigo-600">
                  ${results.cost.estimated_cost_usd.toFixed(4)}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Right column: Sentence analysis (spans 2 cols) */}
        <div className="lg:col-span-2">
          <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className="text-lg font-semibold text-slate-800">
                  Your Idea Analysis
                </h3>
                <p className="text-xs text-slate-500 mt-1">
                  Click 🔍 on highlighted sentences to see matching sources
                </p>
              </div>
            </div>

            <SentenceHighlighting
              annotations={results.sentence_annotations}
              onSentenceClick={(ann) => setSelectedSentence(ann)}
            />
          </div>
        </div>
      </div>

      {/* Paper comparison table */}
      {results.papers && results.papers.length > 0 && (
        <div className="mt-6">
          <PaperTable papers={results.papers} />
        </div>
      )}

      {/* Actions */}
      <div className="flex items-center gap-4 mt-8">
        <button
          onClick={onNewAnalysis}
          className="px-5 py-2.5 bg-gradient-to-r from-indigo-600 to-purple-600 text-white font-semibold rounded-lg shadow-md hover:shadow-lg hover:-translate-y-0.5 transition-all"
        >
          New Analysis
        </button>
        <a
          href={`data:text/markdown;charset=utf-8,${encodeURIComponent(reportText)}`}
          download="originality_report.md"
          className="px-5 py-2.5 border border-slate-300 text-slate-700 font-medium rounded-lg hover:bg-slate-50 transition-colors"
        >
          Download Report
        </a>
      </div>

      {/* Modal */}
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

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div className="bg-slate-50 rounded-lg p-3 text-center">
      <div className="text-lg font-bold text-slate-800">{value}</div>
      <div className="text-xs text-slate-500">{label}</div>
    </div>
  );
}

function RealityCheckBanner({
  realityCheck,
  results,
}: {
  realityCheck: RealityCheckInfo | null;
  results: AnalysisResults;
}) {
  const rc =
    (realityCheck?.result as Record<string, unknown>) ??
    (results.reality_check as Record<string, unknown> | undefined);
  if (!rc) return null;

  const exists = rc.already_exists === true;
  const examples = (rc.existing_examples ?? []) as Array<{
    name: string;
    similarity: number;
    description: string;
  }>;
  const assessment = rc.assessment as string | undefined;
  const recommendation = rc.recommendation as string | undefined;
  const noveltyAspects = (rc.novelty_aspects ?? []) as string[];

  if (!exists) return null;

  return (
    <div className="bg-amber-50 border border-amber-200 rounded-2xl p-5 mb-6">
      <div className="flex items-start gap-3">
        <span className="text-xl flex-shrink-0">⚠️</span>
        <div className="flex-1">
          <h3 className="text-sm font-bold text-amber-800 mb-1">
            This idea may already exist
          </h3>
          {assessment && (
            <p className="text-sm text-amber-700 leading-relaxed mb-3">
              {assessment}
            </p>
          )}

          {examples.length > 0 && (
            <details className="mb-3">
              <summary className="text-sm text-amber-700 cursor-pointer font-medium hover:text-amber-900">
                Similar existing products/research ({examples.length})
              </summary>
              <div className="mt-2 space-y-2">
                {examples.slice(0, 5).map((ex, i) => (
                  <div
                    key={i}
                    className="bg-white rounded-lg px-3 py-2 border border-amber-100 flex items-start gap-2"
                  >
                    <span className="text-xs font-bold text-amber-600 flex-shrink-0 mt-0.5">
                      {Math.round(ex.similarity * 100)}%
                    </span>
                    <div>
                      <span className="text-sm font-medium text-slate-800">
                        {ex.name}
                      </span>
                      <p className="text-xs text-slate-500 mt-0.5">
                        {ex.description}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </details>
          )}

          {noveltyAspects.length > 0 && (
            <div className="mb-3">
              <p className="text-xs font-semibold text-green-700 mb-1">
                Potentially novel aspects:
              </p>
              <ul className="text-xs text-green-600 list-disc list-inside space-y-0.5">
                {noveltyAspects.map((a, i) => (
                  <li key={i}>{a}</li>
                ))}
              </ul>
            </div>
          )}

          {recommendation && (
            <p className="text-xs text-amber-600 italic mb-3">
              {recommendation}
            </p>
          )}

          <p className="text-[11px] text-amber-400 border-t border-amber-200 pt-2">
            This is an advisory check based on LLM pretraining data and does not
            affect the originality score above. The score is calculated solely
            from grounded analysis of retrieved academic papers.
          </p>
        </div>
      </div>
    </div>
  );
}
