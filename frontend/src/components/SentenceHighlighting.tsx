import { useState } from "react";
import type { SentenceAnnotation } from "../types/api";

interface Props {
  annotations: SentenceAnnotation[];
  onSentenceClick: (ann: SentenceAnnotation) => void;
}

const LABEL_STYLES: Record<string, { bg: string; border: string; text: string; hover: string; badge: string }> = {
  high: {
    bg: "bg-green-50",
    border: "border-l-green-500",
    text: "text-green-800",
    hover: "hover:bg-green-100",
    badge: "bg-green-500",
  },
  medium: {
    bg: "bg-amber-50",
    border: "border-l-amber-500",
    text: "text-amber-800",
    hover: "hover:bg-amber-100",
    badge: "bg-amber-500",
  },
  low: {
    bg: "bg-red-50",
    border: "border-l-red-500",
    text: "text-red-800",
    hover: "hover:bg-red-100",
    badge: "bg-red-500",
  },
};

export default function SentenceHighlighting({
  annotations,
  onSentenceClick,
}: Props) {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

  return (
    <div>
      {/* Legend */}
      <div className="flex items-center justify-center gap-5 mb-4 py-2 bg-slate-50 rounded-lg">
        <span className="flex items-center gap-1.5 text-xs text-slate-600">
          <span className="w-2.5 h-2.5 rounded-full bg-green-500" />
          High Originality
        </span>
        <span className="flex items-center gap-1.5 text-xs text-slate-600">
          <span className="w-2.5 h-2.5 rounded-full bg-amber-500" />
          Moderate
        </span>
        <span className="flex items-center gap-1.5 text-xs text-slate-600">
          <span className="w-2.5 h-2.5 rounded-full bg-red-500" />
          Low Originality
        </span>
      </div>

      {/* Sentences */}
      <div className="space-y-3">
        {annotations.map((ann) => {
          const style = LABEL_STYLES[ann.label] ?? LABEL_STYLES.high;
          const hasMatches = ann.linked_sections.length > 0;
          const isClickable = ann.label !== "high" || hasMatches;
          const isHovered = hoveredIndex === ann.index;
          const topMatch = ann.linked_sections[0];

          return (
            <div key={ann.index} className="relative">
              {/* Main sentence card */}
              <div
                className={`flex items-start gap-2 border-l-4 rounded-r-lg px-4 py-3 transition-all ${style.bg} ${style.border} ${style.text} ${isClickable ? `cursor-pointer ${style.hover}` : ""} ${isHovered ? "shadow-md" : ""}`}
                onClick={() => isClickable && onSentenceClick(ann)}
                onMouseEnter={() => hasMatches && setHoveredIndex(ann.index)}
                onMouseLeave={() => setHoveredIndex(null)}
              >
                <p className="flex-1 text-sm leading-relaxed">{ann.sentence}</p>
                <div className="flex items-center gap-2 flex-shrink-0 mt-0.5">
                  <span className="text-xs opacity-70 font-medium">
                    {Math.round(ann.overlap_score * 100)}%
                  </span>
                  {isClickable && (
                    <span
                      className="text-base opacity-70 hover:opacity-100 transition-opacity"
                      title="View matching evidence"
                    >
                      🔍
                    </span>
                  )}
                </div>
              </div>

              {/* Hover preview of matching evidence */}
              {isHovered && topMatch && (
                <div className="absolute left-4 right-4 top-full mt-1 z-20 animate-fadeIn">
                  <div className="bg-white rounded-xl shadow-xl border border-slate-200 overflow-hidden">
                    {/* Arrow */}
                    <div className="absolute -top-2 left-6 w-4 h-4 bg-white border-l border-t border-slate-200 transform rotate-45" />
                    
                    {/* Content */}
                    <div className="relative p-4">
                      <div className="flex items-center gap-2 mb-2">
                        <span className={`px-2 py-0.5 rounded text-xs font-bold text-white ${style.badge}`}>
                          {Math.round(topMatch.similarity * 100)}% match
                        </span>
                        <span className="text-xs text-slate-500 truncate flex-1">
                          {topMatch.paper_title}
                        </span>
                      </div>
                      
                      {topMatch.text_snippet ? (
                        <p className="text-sm text-slate-700 leading-relaxed line-clamp-3">
                          "{topMatch.text_snippet.slice(0, 200)}{topMatch.text_snippet.length > 200 ? "..." : ""}"
                        </p>
                      ) : topMatch.reason ? (
                        <p className="text-sm text-slate-600 italic">
                          {topMatch.reason}
                        </p>
                      ) : (
                        <p className="text-sm text-slate-400 italic">
                          Semantic similarity detected
                        </p>
                      )}
                      
                      {ann.linked_sections.length > 1 && (
                        <div className="mt-2 pt-2 border-t border-slate-100 text-xs text-slate-500">
                          +{ann.linked_sections.length - 1} more match{ann.linked_sections.length > 2 ? "es" : ""} • Click to view all
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* CSS for animation */}
      <style>{`
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(-4px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .animate-fadeIn {
          animation: fadeIn 0.15s ease-out;
        }
        .line-clamp-3 {
          display: -webkit-box;
          -webkit-line-clamp: 3;
          -webkit-box-orient: vertical;
          overflow: hidden;
        }
      `}</style>
    </div>
  );
}
