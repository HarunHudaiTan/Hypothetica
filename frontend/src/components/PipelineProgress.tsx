import { useMemo } from "react";

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
  /** Short adapter label, e.g. "arXiv", "Google Patents" */
  evidenceDisplayName?: string;
  /** e.g. "paper", "patent", "repository" — used in step labels */
  evidenceNounSingular?: string;
}

function buildSteps(displayName: string, nounSingular: string) {
  const capNoun = nounSingular.charAt(0).toUpperCase() + nounSingular.slice(1);
  return [
    { label: "Follow-up Questions", threshold: 0.1 },
    { label: "Reality Check (advisory)", threshold: 0.05 },
    { label: `Query Variants & ${displayName} Search`, threshold: 0.25 },
    { label: "Embedding Search & Reranking", threshold: 0.4 },
    { label: `LLM ${capNoun} Selection`, threshold: 0.5 },
    { label: "PDF Processing & Indexing", threshold: 0.75 },
    { label: "Layer 1 Analysis", threshold: 0.9 },
    { label: "Layer 2 Aggregation", threshold: 0.98 },
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
}: Props) {
  const steps = useMemo(
    () => buildSteps(evidenceDisplayName, evidenceNounSingular),
    [evidenceDisplayName, evidenceNounSingular]
  );
  const pct = Math.round(progress * 100);
  const rcResult = realityCheck?.result as Record<string, unknown> | undefined;
  const rcExists = rcResult?.already_exists === true;
  const rcExamples = (rcResult?.existing_examples ?? []) as Array<{
    name: string;
    similarity: number;
    description: string;
  }>;
  const rcAssessment = typeof rcResult?.assessment === "string" ? rcResult.assessment : "";
  const rcRecommendation = typeof rcResult?.recommendation === "string" ? rcResult.recommendation : "";

  return (
    <div className="space-y-4">
      <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-8">
        <h2 className="text-xl font-semibold text-slate-800 mb-6">
          Analyzing Your Research Idea
        </h2>

        {error ? (
          <div className="bg-red-50 border border-red-200 rounded-xl p-4">
            <p className="text-red-700 font-medium mb-2">Analysis failed</p>
            <p className="text-red-600 text-sm">{error}</p>
            {onRetry && (
              <button
                onClick={onRetry}
                className="mt-3 px-4 py-2 bg-red-100 text-red-700 rounded-lg text-sm font-medium hover:bg-red-200 transition-colors"
              >
                ← Start Over
              </button>
            )}
          </div>
        ) : (
          <>
            {/* Progress bar */}
            <div className="mb-6">
              <div className="flex justify-between text-sm mb-2">
                <span className="text-slate-600 font-medium">{message}</span>
                <span className="text-indigo-600 font-semibold">{pct}%</span>
              </div>
              <div className="w-full h-3 bg-slate-100 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-indigo-500 to-purple-500 rounded-full transition-all duration-500 ease-out"
                  style={{ width: `${pct}%` }}
                />
              </div>
            </div>

            {/* Step indicators */}
            <div className="space-y-2">
              {steps.map((step) => {
                const done = progress >= step.threshold;
                const active = !done && progress >= step.threshold - 0.1;

                return (
                  <div key={step.label} className="flex items-center gap-3">
                    <div
                      className={`w-5 h-5 rounded-full flex items-center justify-center flex-shrink-0 transition-colors ${
                        done
                          ? "bg-green-500"
                          : active
                            ? "bg-indigo-500 animate-pulse"
                            : "bg-slate-200"
                      }`}
                    >
                      {done && (
                        <svg
                          className="w-3 h-3 text-white"
                          fill="none"
                          viewBox="0 0 24 24"
                          stroke="currentColor"
                          strokeWidth={3}
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            d="M5 13l4 4L19 7"
                          />
                        </svg>
                      )}
                    </div>
                    <span
                      className={`text-sm ${
                        done
                          ? "text-green-700 font-medium"
                          : active
                            ? "text-indigo-700 font-medium"
                            : "text-slate-400"
                      }`}
                    >
                      {step.label}
                    </span>
                  </div>
                );
              })}
            </div>
          </>
        )}
      </div>

      {/* Reality check banner — shows as soon as it arrives */}
      {realityCheck && rcExists && (
        <div className="bg-amber-50 border border-amber-200 rounded-2xl p-5 animate-in fade-in">
          <div className="flex items-start gap-3">
            <span className="text-xl flex-shrink-0 mt-0.5">⚠️</span>
            <div className="flex-1 min-w-0">
              <h3 className="text-sm font-semibold text-amber-800 mb-1">
                This idea may already exist
              </h3>
              <p className="text-sm text-amber-700 leading-relaxed">
                {rcAssessment ||
                  "Our initial check suggests similar concepts already exist. This does not affect the final score — the score is based solely on grounded analysis of retrieved evidence."}
              </p>

              {rcExamples.length > 0 && (
                <div className="mt-3 space-y-2">
                  {rcExamples.slice(0, 3).map((ex, i) => (
                    <div
                      key={i}
                      className="flex items-center gap-2 bg-white rounded-lg px-3 py-2 border border-amber-100"
                    >
                      <span className="text-xs font-bold text-amber-600 flex-shrink-0">
                        {Math.round(ex.similarity * 100)}%
                      </span>
                      <div className="min-w-0">
                        <span className="text-sm font-medium text-slate-800">
                          {ex.name}
                        </span>
                        <span className="text-xs text-slate-500 ml-1.5">
                          — {ex.description}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {rcRecommendation && (
                <p className="mt-3 text-xs text-amber-600 italic">
                  {rcRecommendation}
                </p>
              )}

              <p className="mt-3 text-[11px] text-amber-500 border-t border-amber-200 pt-2">
                This is an advisory check based on LLM pretraining data. It does
                not affect your originality score, which is calculated solely
                from grounded analysis of retrieved evidence.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Non-existing — show a green note */}
      {realityCheck && !rcExists && (
        <div className="bg-green-50 border border-green-200 rounded-2xl p-4">
          <div className="flex items-center gap-2">
            <span className="text-lg">✅</span>
            <p className="text-sm text-green-700">
              No obvious existing products or well-known solutions found for this
              idea.
            </p>
          </div>
          <p className="mt-1 text-[11px] text-green-500 ml-8">
            Advisory check based on LLM pretraining data — does not affect the
            final score.
          </p>
        </div>
      )}
    </div>
  );
}
