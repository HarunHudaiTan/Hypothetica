import type { SentenceAnnotation } from "../types/api";

interface Props {
  annotations: SentenceAnnotation[];
  onSentenceClick: (ann: SentenceAnnotation) => void;
}

const LABEL_STYLES: Record<string, string> = {
  high: "bg-green-50 border-l-green-500 text-green-800 hover:bg-green-100",
  medium: "bg-amber-50 border-l-amber-500 text-amber-800 hover:bg-amber-100",
  low: "bg-red-50 border-l-red-500 text-red-800 hover:bg-red-100",
};

export default function SentenceHighlighting({
  annotations,
  onSentenceClick,
}: Props) {
  return (
    <div>
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

      <div className="space-y-2">
        {annotations.map((ann) => {
          const style = LABEL_STYLES[ann.label] ?? LABEL_STYLES.high;
          const hasMatches = ann.linked_sections.length > 0;
          const isClickable = ann.label !== "high" || hasMatches;

          return (
            <div
              key={ann.index}
              className={`flex items-start gap-2 border-l-4 rounded-r-lg px-4 py-3 transition-colors ${style} ${isClickable ? "cursor-pointer" : ""}`}
              onClick={() => isClickable && onSentenceClick(ann)}
            >
              <p className="flex-1 text-sm leading-relaxed">{ann.sentence}</p>
              <div className="flex items-center gap-2 flex-shrink-0 mt-0.5">
                <span className="text-xs opacity-60">
                  {Math.round(ann.overlap_score * 100)}%
                </span>
                {isClickable && (
                  <span
                    className="text-lg opacity-70 hover:opacity-100 transition-opacity"
                    title="View matching sources"
                  >
                    🔍
                  </span>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
