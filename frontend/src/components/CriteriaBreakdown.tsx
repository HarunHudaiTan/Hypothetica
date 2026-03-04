import type { CriteriaScores } from "../types/api";

interface Props {
  criteria: CriteriaScores;
}

const CRITERIA = [
  {
    key: "problem_similarity" as const,
    label: "Problem",
    icon: "🎯",
    description:
      "How closely does your research question match existing work? Lower similarity means your problem framing is more unique.",
  },
  {
    key: "method_similarity" as const,
    label: "Method",
    icon: "⚙️",
    description:
      "Overlap in proposed techniques, algorithms, or approaches. A low score suggests a novel methodological contribution.",
  },
  {
    key: "domain_overlap" as const,
    label: "Domain",
    icon: "🌐",
    description:
      "Overlap in the application field or area of study. Cross-domain ideas tend to score lower here.",
  },
  {
    key: "contribution_similarity" as const,
    label: "Contribution",
    icon: "💡",
    description:
      "How similar are the claimed results and findings? A low score indicates novel expected outcomes.",
  },
];

function barColor(originalityPct: number): string {
  if (originalityPct >= 70) return "bg-green-500";
  if (originalityPct >= 40) return "bg-amber-500";
  return "bg-red-500";
}

function labelColor(originalityPct: number): string {
  if (originalityPct >= 70) return "text-green-600";
  if (originalityPct >= 40) return "text-amber-600";
  return "text-red-600";
}

function levelText(originalityPct: number): string {
  if (originalityPct >= 70) return "High originality";
  if (originalityPct >= 40) return "Moderate";
  return "Low originality";
}

export default function CriteriaBreakdown({ criteria }: Props) {
  return (
    <div>
      <h3 className="text-sm font-semibold text-slate-700 uppercase tracking-wider mb-1">
        Criteria Breakdown
      </h3>
      <p className="text-xs text-slate-400 mb-4">
        Aggregated across all analyzed papers. Higher percentage = more original.
      </p>
      <div className="space-y-4">
        {CRITERIA.map(({ key, label, icon, description }) => {
          const similarity = criteria[key];
          const originalityPct = Math.round((1 - similarity) * 100);

          return (
            <div key={key}>
              <div className="flex items-center justify-between mb-1">
                <span className="text-sm font-medium text-slate-700">
                  {icon} {label}
                </span>
                <div className="flex items-center gap-2">
                  <span
                    className={`text-xs font-semibold ${labelColor(originalityPct)}`}
                  >
                    {levelText(originalityPct)}
                  </span>
                  <span className="text-sm font-bold text-slate-800">
                    {originalityPct}%
                  </span>
                </div>
              </div>
              <div className="w-full h-2 bg-slate-100 rounded-full overflow-hidden mb-1.5">
                <div
                  className={`h-full rounded-full transition-all duration-500 ${barColor(originalityPct)}`}
                  style={{ width: `${originalityPct}%` }}
                />
              </div>
              <p className="text-xs text-slate-400 leading-snug">
                {description}
              </p>
            </div>
          );
        })}
      </div>
    </div>
  );
}
