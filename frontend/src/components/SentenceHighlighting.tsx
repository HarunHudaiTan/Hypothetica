import { useState } from "react";
import type { SentenceAnnotation } from "../types/api";

interface Props {
  annotations: SentenceAnnotation[];
  onSentenceClick: (ann: SentenceAnnotation) => void;
}

const CRITERION_DISPLAY: Record<string, string> = {
  problem_similarity: "Problem",
  method_similarity: "Method",
  domain_similarity: "Domain",
  contribution_similarity: "Contribution",
};

function critAccent(lbl: string): string {
  if (lbl === "low") return "var(--color-vermillion)";
  if (lbl === "medium") return "var(--color-ochre)";
  return "var(--color-moss)";
}

function rowAccent(label: string): string {
  if (label === "low") return "var(--color-vermillion)";
  if (label === "medium") return "var(--color-ochre)";
  return "var(--color-moss)";
}

export default function SentenceHighlighting({
  annotations,
  onSentenceClick,
}: Props) {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

  return (
    <div>
      <div className="flex flex-wrap items-center gap-x-5 gap-y-1 mb-5 py-2 small-caps text-[color:var(--color-ink-fade)] border-y border-[color:var(--color-rule)]">
        <span>marginalia</span>
        <span className="inline-flex items-center gap-1.5">
          <span
            className="inline-block w-6"
            style={{ background: "var(--color-moss)", height: "3px" }}
          />
          distinct (1–2)
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span
            className="inline-block w-6"
            style={{ background: "var(--color-ochre)", height: "3px" }}
          />
          moderate (3)
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span
            className="inline-block w-6"
            style={{ background: "var(--color-vermillion)", height: "3px" }}
          />
          high (4–5)
        </span>
      </div>

      <ol className="divide-y divide-[color:var(--color-rule)]">
        {annotations.map((ann) => {
          const accent = rowAccent(ann.label);
          const hasMatches = ann.linked_sections.length > 0;
          const clickable = ann.label !== "high" || hasMatches;
          const isHovered = hoveredIndex === ann.index;
          const topMatch = ann.linked_sections[0];

          return (
            <li
              key={ann.index}
              className={`relative grid grid-cols-[3px_1fr_auto] gap-4 py-4 transition-colors ${
                clickable
                  ? "cursor-pointer hover:bg-[color:var(--color-paper-shade)]/60"
                  : ""
              }`}
              style={{ borderLeft: `0` }}
              onClick={() => clickable && onSentenceClick(ann)}
              onMouseEnter={() => setHoveredIndex(ann.index)}
              onMouseLeave={() => setHoveredIndex(null)}
            >
              <span
                aria-hidden
                style={{ background: accent }}
                className="block w-[3px] h-full"
              />
              <p className="font-body text-[15px] leading-relaxed text-[color:var(--color-ink)]">
                {ann.sentence}
              </p>
              <div className="flex flex-wrap items-start justify-end gap-2 max-w-[12rem] small-caps">
                {Object.entries(ann.criteria_labels ?? {}).map(
                  ([crit, lbl]) => (
                    <span
                      key={crit}
                      className="whitespace-nowrap"
                      style={{
                        color: critAccent(lbl),
                        borderBottom: `1px solid ${critAccent(lbl)}`,
                      }}
                    >
                      {CRITERION_DISPLAY[crit] ?? crit}
                    </span>
                  )
                )}
                {clickable && (
                  <span className="font-mono text-[10px] text-[color:var(--color-ink-fade)]">
                    ↗
                  </span>
                )}
              </div>

              {isHovered && topMatch && hasMatches && (
                <div className="absolute left-8 right-4 top-full mt-1 z-20 pointer-events-none">
                  <div className="bg-[color:var(--color-paper)] border border-[color:var(--color-rule-strong)] shadow-md px-4 py-3">
                    <p className="font-display italic text-[color:var(--color-ink)] text-sm mb-1">
                      {topMatch.paper_title}
                    </p>
                    {topMatch.text_snippet ? (
                      <p className="font-body text-xs text-[color:var(--color-ink-soft)] leading-relaxed line-clamp-3">
                        “{topMatch.text_snippet.slice(0, 200)}
                        {topMatch.text_snippet.length > 200 ? "…" : ""}”
                      </p>
                    ) : topMatch.reason ? (
                      <p className="font-body text-xs text-[color:var(--color-ink-soft)] italic">
                        {topMatch.reason}
                      </p>
                    ) : (
                      <p className="font-body text-xs text-[color:var(--color-ink-fade)] italic">
                        semantic similarity detected
                      </p>
                    )}
                  </div>
                </div>
              )}
            </li>
          );
        })}
      </ol>
    </div>
  );
}
