import { useState, useEffect } from "react";
import type { SentenceAnnotation, MatchedSection } from "../types/api";
import { getSentenceMatches } from "../lib/api";

interface Props {
  annotations: SentenceAnnotation[];
  jobId: string;
}

function getLabelStyle(label: string) {
  switch (label) {
    case "low":
      return {
        bg: "bg-red-200/70",
        border: "border-b-2 border-red-400",
        hoverBg: "hover:bg-red-300/70",
        panelBg: "bg-red-50",
        panelBorder: "border-red-300",
        badge: "bg-red-500",
        text: "text-red-900",
        accent: "#ef4444",
      };
    case "medium":
      return {
        bg: "bg-amber-200/70",
        border: "border-b-2 border-amber-400",
        hoverBg: "hover:bg-amber-300/70",
        panelBg: "bg-amber-50",
        panelBorder: "border-amber-300",
        badge: "bg-amber-500",
        text: "text-amber-900",
        accent: "#f59e0b",
      };
    default:
      return {
        bg: "",
        border: "",
        hoverBg: "",
        panelBg: "bg-green-50",
        panelBorder: "border-green-300",
        badge: "bg-green-500",
        text: "text-green-900",
        accent: "#22c55e",
      };
  }
}

export default function HighlightedIdea({ annotations, jobId }: Props) {
  const [selectedAnn, setSelectedAnn] = useState<SentenceAnnotation | null>(null);
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const [matches, setMatches] = useState<MatchedSection[]>([]);
  const [loadingMatches, setLoadingMatches] = useState(false);
  const [selectedMatchIndex, setSelectedMatchIndex] = useState(0);

  const isExpanded = selectedAnn !== null;

  useEffect(() => {
    if (!selectedAnn) {
      setMatches([]);
      return;
    }

    const linked = selectedAnn.linked_sections ?? [];
    if (linked.length > 0) {
      setMatches(linked);
      setSelectedMatchIndex(0);
      return;
    }

    setLoadingMatches(true);
    getSentenceMatches(jobId, selectedAnn.sentence)
      .then((result) => {
        setMatches(result);
        setSelectedMatchIndex(0);
      })
      .catch(() => setMatches([]))
      .finally(() => setLoadingMatches(false));
  }, [selectedAnn, jobId]);

  const handleSentenceClick = (ann: SentenceAnnotation) => {
    if (selectedAnn?.index === ann.index) {
      setSelectedAnn(null);
    } else {
      setSelectedAnn(ann);
    }
  };

  const currentMatch = matches[selectedMatchIndex];
  const selectedStyle = selectedAnn ? getLabelStyle(selectedAnn.label) : null;

  return (
    <div className="relative">
      <div className={`flex transition-all duration-300 ease-out ${isExpanded ? "gap-4" : ""}`}>
        {/* Left: Research Idea Text */}
        <div className={`transition-all duration-300 ease-out ${isExpanded ? "w-1/2" : "w-full"}`}>
          <div className={`text-sm text-slate-700 leading-relaxed ${isExpanded ? "space-y-2" : ""}`}>
            {annotations.map((ann, idx) => {
              const style = getLabelStyle(ann.label);
              const hasMatches = ann.linked_sections.length > 0;
              const isClickable = ann.label !== "high" || hasMatches;
              const isSelected = selectedAnn?.index === ann.index;
              const isHovered = hoveredIndex === ann.index;

              if (ann.label === "high" && !hasMatches) {
                return (
                  <span
                    key={idx}
                    className={`transition-all duration-300 ${isExpanded ? "block py-1 px-2 bg-slate-50 rounded" : "inline"}`}
                  >
                    {ann.sentence}{!isExpanded && " "}
                  </span>
                );
              }

              return (
                <span
                  key={idx}
                  className={`relative transition-all duration-300 ${isExpanded ? "block" : "inline"}`}
                >
                  <span
                    className={`${style.bg} ${style.border} ${style.hoverBg} ${isClickable ? "cursor-pointer" : ""} rounded-sm px-0.5 transition-all ${isSelected ? "ring-2 ring-offset-1 ring-slate-400" : ""} ${isExpanded ? "block py-1.5 px-2" : ""}`}
                    onClick={() => isClickable && handleSentenceClick(ann)}
                    onMouseEnter={() => hasMatches && setHoveredIndex(ann.index)}
                    onMouseLeave={() => setHoveredIndex(null)}
                    title={`${Math.round(ann.overlap_score * 100)}% overlap`}
                  >
                    {ann.sentence}
                  </span>

                  {/* Hover preview tooltip */}
                  {isHovered && !isExpanded && ann.linked_sections[0] && (
                    <div className="absolute left-0 top-full mt-1 z-30 w-72 animate-fadeIn">
                      <div className="bg-white rounded-lg shadow-xl border border-slate-200 p-3">
                        <div className="flex items-center gap-2 mb-2">
                          <span className={`px-2 py-0.5 rounded text-xs font-bold text-white ${style.badge}`}>
                            {Math.round(ann.linked_sections[0].similarity * 100)}%
                          </span>
                          <span className="text-xs text-slate-500 truncate">
                            {ann.linked_sections[0].paper_title}
                          </span>
                        </div>
                        {ann.linked_sections[0].text_snippet && (
                          <p className="text-xs text-slate-600 line-clamp-2">
                            "{ann.linked_sections[0].text_snippet.slice(0, 150)}..."
                          </p>
                        )}
                        <p className="text-xs text-slate-400 mt-1">Click to expand</p>
                      </div>
                    </div>
                  )}
                  {!isExpanded && " "}
                </span>
              );
            })}
          </div>
        </div>

        {/* Right: Evidence Panel (slides in when expanded) */}
        <div
          className={`transition-all duration-300 ease-out overflow-hidden ${
            isExpanded ? "w-1/2 opacity-100" : "w-0 opacity-0"
          }`}
        >
          {selectedAnn && selectedStyle && (
            <div className={`${selectedStyle.panelBg} border ${selectedStyle.panelBorder} rounded-xl p-4 h-full`}>
              {/* Header */}
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <div
                    className="w-3 h-3 rounded-full"
                    style={{ backgroundColor: selectedStyle.accent }}
                  />
                  <span className="text-xs font-bold text-slate-600 uppercase">
                    Matching Evidence
                  </span>
                </div>
                <button
                  onClick={() => setSelectedAnn(null)}
                  className="p-1 text-slate-400 hover:text-slate-600 hover:bg-white/50 rounded transition-colors"
                >
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>

              {/* Selected sentence */}
              <div className={`${selectedStyle.panelBg} border ${selectedStyle.panelBorder} rounded-lg p-3 mb-3`}>
                <p className={`text-sm font-medium ${selectedStyle.text}`}>
                  "{selectedAnn.sentence}"
                </p>
                <span className={`inline-block mt-2 px-2 py-0.5 rounded text-xs font-bold text-white ${selectedStyle.badge}`}>
                  {Math.round(selectedAnn.overlap_score * 100)}% overlap
                </span>
              </div>

              {/* Evidence content */}
              {loadingMatches ? (
                <div className="flex items-center justify-center py-6">
                  <div className="w-5 h-5 border-2 border-slate-400 border-t-transparent rounded-full animate-spin" />
                  <span className="ml-2 text-xs text-slate-500">Loading...</span>
                </div>
              ) : matches.length === 0 ? (
                <p className="text-xs text-slate-500 text-center py-4">
                  No detailed evidence available.
                </p>
              ) : (
                <div className="space-y-3">
                  {/* Match selector if multiple */}
                  {matches.length > 1 && (
                    <div className="flex items-center gap-1 flex-wrap">
                      {matches.map((_, idx) => (
                        <button
                          key={idx}
                          onClick={() => setSelectedMatchIndex(idx)}
                          className={`w-6 h-6 rounded text-xs font-bold transition-all ${
                            selectedMatchIndex === idx
                              ? `${selectedStyle.badge} text-white`
                              : "bg-white text-slate-600 hover:bg-slate-100"
                          }`}
                        >
                          {idx + 1}
                        </button>
                      ))}
                    </div>
                  )}

                  {/* Current match details */}
                  {currentMatch && (
                    <div className="space-y-2">
                      {/* Paper info */}
                      <div className="bg-white/60 rounded-lg p-2">
                        <div className="text-xs text-slate-500">Paper:</div>
                        <div className="text-xs font-semibold text-slate-800 truncate">
                          {currentMatch.paper_title}
                        </div>
                        {currentMatch.heading && (
                          <div className="text-xs text-slate-500 mt-1">
                            Section: {currentMatch.heading}
                          </div>
                        )}
                      </div>

                      {/* Matching passage */}
                      {currentMatch.text_snippet && (
                        <div className="bg-white rounded-lg p-3 border border-slate-200">
                          <div className="text-xs font-bold text-slate-500 mb-1 flex items-center gap-1">
                            <span className="w-2 h-2 rounded-full" style={{ backgroundColor: selectedStyle.accent }} />
                            Matching Passage
                          </div>
                          <p className="text-xs text-slate-700 leading-relaxed">
                            "{currentMatch.text_snippet}"
                          </p>
                        </div>
                      )}

                      {/* Similarity */}
                      <div className="flex items-center gap-2">
                        <div className="flex-1 h-1.5 bg-slate-200 rounded-full overflow-hidden">
                          <div
                            className="h-full rounded-full"
                            style={{
                              width: `${currentMatch.similarity * 100}%`,
                              backgroundColor: selectedStyle.accent,
                            }}
                          />
                        </div>
                        <span className="text-xs font-bold text-slate-600">
                          {Math.round(currentMatch.similarity * 100)}%
                        </span>
                      </div>

                      {/* Reason */}
                      {currentMatch.reason && (
                        <div className="bg-blue-50 rounded-lg p-2 border border-blue-100">
                          <div className="text-xs font-bold text-blue-600 mb-1">Why:</div>
                          <p className="text-xs text-blue-800">{currentMatch.reason}</p>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      <style>{`
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(-4px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .animate-fadeIn {
          animation: fadeIn 0.15s ease-out;
        }
        .line-clamp-2 {
          display: -webkit-box;
          -webkit-line-clamp: 2;
          -webkit-box-orient: vertical;
          overflow: hidden;
        }
      `}</style>
    </div>
  );
}
