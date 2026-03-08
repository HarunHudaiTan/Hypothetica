import { useState, useEffect, useMemo } from "react";
import type { SentenceAnnotation, MatchedSection } from "../types/api";
import { getSentenceMatches } from "../lib/api";

interface Props {
  annotation: SentenceAnnotation;
  jobId: string;
  onClose: () => void;
}

function getLabelColors(label: string) {
  switch (label) {
    case "low":
      return {
        bg: "bg-red-50",
        border: "border-red-400",
        text: "text-red-900",
        badge: "bg-red-500",
        lightBg: "bg-red-100",
        accent: "#ef4444",
      };
    case "medium":
      return {
        bg: "bg-amber-50",
        border: "border-amber-400",
        text: "text-amber-900",
        badge: "bg-amber-500",
        lightBg: "bg-amber-100",
        accent: "#f59e0b",
      };
    default:
      return {
        bg: "bg-green-50",
        border: "border-green-400",
        text: "text-green-900",
        badge: "bg-green-500",
        lightBg: "bg-green-100",
        accent: "#22c55e",
      };
  }
}

export default function MatchesModal({ annotation, jobId, onClose }: Props) {
  const [ragMatches, setRagMatches] = useState<MatchedSection[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedIndex, setSelectedIndex] = useState(0);

  const colors = getLabelColors(annotation.label);

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
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm"
      onClick={(e) => e.target === e.currentTarget && onClose()}
    >
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-5xl max-h-[85vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-200 bg-slate-50 flex-shrink-0">
          <div className="flex items-center gap-3">
            <div
              className="w-4 h-4 rounded-full"
              style={{ backgroundColor: colors.accent }}
            />
            <div>
              <h2 className="text-lg font-bold text-slate-800">
                Evidence Bridge
              </h2>
              <p className="text-xs text-slate-500">
                Your idea ↔ Matching paper passages
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 text-slate-400 hover:text-slate-600 hover:bg-slate-200 rounded-lg transition-colors"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-auto">
          {loading ? (
            <div className="flex items-center justify-center py-20">
              <div className="w-8 h-8 border-3 border-indigo-500 border-t-transparent rounded-full animate-spin" />
              <span className="ml-3 text-slate-500">Loading evidence...</span>
            </div>
          ) : allMatches.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-20 text-slate-400">
              <span className="text-4xl mb-3">📭</span>
              <p>No matching passages found for this sentence.</p>
            </div>
          ) : (
            <div className="grid grid-cols-1 lg:grid-cols-2 min-h-[400px]">
              {/* LEFT: User's Sentence */}
              <div className="p-6 bg-slate-50 border-b lg:border-b-0 lg:border-r border-slate-200">
                <div className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-4 flex items-center gap-2">
                  <span className="text-lg">💡</span> Your Sentence
                </div>
                
                <div className={`${colors.bg} border-2 ${colors.border} rounded-xl p-5`}>
                  <p className={`text-base leading-relaxed font-medium ${colors.text}`}>
                    "{annotation.sentence}"
                  </p>
                  <div className="mt-4 pt-3 border-t border-slate-200/50 flex items-center gap-3">
                    <span className={`px-3 py-1 rounded-full text-xs font-bold text-white ${colors.badge}`}>
                      {Math.round(annotation.overlap_score * 100)}% overlap
                    </span>
                    <span className="text-xs text-slate-500">
                      {allMatches.length} match{allMatches.length !== 1 ? "es" : ""}
                    </span>
                  </div>
                </div>

                {/* Match selector */}
                {allMatches.length > 1 && (
                  <div className="mt-6">
                    <div className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-3">
                      Select Match ({selectedIndex + 1}/{allMatches.length})
                    </div>
                    <div className="flex flex-wrap gap-2">
                      {allMatches.map((match, idx) => (
                        <button
                          key={idx}
                          onClick={() => setSelectedIndex(idx)}
                          className={`px-3 py-2 rounded-lg text-xs font-medium transition-all ${
                            selectedIndex === idx
                              ? `${colors.badge} text-white shadow-md`
                              : "bg-white border border-slate-200 text-slate-600 hover:bg-slate-100"
                          }`}
                        >
                          {Math.round(match.similarity * 100)}% • {match.heading?.slice(0, 15) || `#${idx + 1}`}
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {/* RIGHT: Paper Evidence */}
              <div className="p-6 bg-white">
                <div className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-4 flex items-center gap-2">
                  <span className="text-lg">📄</span> Matching Paper Evidence
                </div>

                {currentMatch ? (
                  <div className="space-y-4">
                    {/* Paper info */}
                    <div className="bg-slate-100 rounded-xl p-4">
                      <div className="text-xs text-slate-500 mb-1">From Paper:</div>
                      <h4 className="text-sm font-bold text-slate-800 leading-snug">
                        {currentMatch.paper_title || "Unknown Paper"}
                      </h4>
                      {currentMatch.heading && (
                        <div className="mt-2 text-xs text-slate-600">
                          Section: <span className="font-semibold">{currentMatch.heading}</span>
                        </div>
                      )}
                    </div>

                    {/* Similarity bar */}
                    <div className="flex items-center gap-3">
                      <div className="flex-1 h-3 bg-slate-200 rounded-full overflow-hidden">
                        <div
                          className="h-full rounded-full transition-all"
                          style={{
                            width: `${currentMatch.similarity * 100}%`,
                            backgroundColor: colors.accent,
                          }}
                        />
                      </div>
                      <span className={`px-2 py-1 rounded text-xs font-bold text-white ${colors.badge}`}>
                        {Math.round(currentMatch.similarity * 100)}%
                      </span>
                    </div>

                    {/* The actual matching passage from paper */}
                    <div className={`${colors.lightBg} border-2 ${colors.border} rounded-xl p-5`}>
                      <div className="text-xs font-bold text-slate-600 uppercase tracking-wider mb-3 flex items-center gap-2">
                        <div className="w-2 h-2 rounded-full" style={{ backgroundColor: colors.accent }} />
                        Matching Passage
                      </div>
                      {currentMatch.text_snippet ? (
                        <p className={`text-sm leading-relaxed ${colors.text}`}>
                          "{currentMatch.text_snippet}"
                        </p>
                      ) : (
                        <p className="text-sm text-slate-400 italic">
                          No text snippet available. The similarity was detected through semantic analysis.
                        </p>
                      )}
                    </div>

                    {/* Why it matches */}
                    {currentMatch.reason && (
                      <div className="bg-blue-50 border border-blue-200 rounded-xl p-5">
                        <div className="text-xs font-bold text-blue-600 uppercase tracking-wider mb-2 flex items-center gap-2">
                          🔗 Why This Matches
                        </div>
                        <p className="text-sm text-blue-900 leading-relaxed">
                          {currentMatch.reason}
                        </p>
                      </div>
                    )}
                  </div>
                ) : (
                  <p className="text-slate-400 text-sm">Select a match to view details.</p>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Footer navigation */}
        {allMatches.length > 1 && (
          <div className="flex items-center justify-between px-6 py-3 border-t border-slate-200 bg-slate-50 flex-shrink-0">
            <button
              onClick={() => setSelectedIndex((p) => (p > 0 ? p - 1 : allMatches.length - 1))}
              className="flex items-center gap-2 px-4 py-2 text-sm text-slate-600 hover:bg-white rounded-lg transition-colors"
            >
              ← Previous
            </button>
            <span className="text-sm text-slate-500">
              {selectedIndex + 1} of {allMatches.length}
            </span>
            <button
              onClick={() => setSelectedIndex((p) => (p < allMatches.length - 1 ? p + 1 : 0))}
              className="flex items-center gap-2 px-4 py-2 text-sm text-slate-600 hover:bg-white rounded-lg transition-colors"
            >
              Next →
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
