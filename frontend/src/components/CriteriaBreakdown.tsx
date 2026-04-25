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
    key: "domain_similarity" as const,
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

/** Convert 0-1 similarity float → Likert 1-5 */
function toLikert(similarity: number): number {
  if (similarity >= 1.0) return 5;
  if (similarity >= 0.75) return 4;
  if (similarity >= 0.5) return 3;
  if (similarity >= 0.25) return 2;
  return 1;
}

function likertColor(likert: number): string {
  if (likert >= 4) return "text-red-600";
  if (likert >= 3) return "text-amber-600";
  return "text-green-600";
}

function likertDotColor(likert: number): string {
  if (likert >= 4) return "bg-red-500";
  if (likert >= 3) return "bg-amber-500";
  return "bg-green-500";
}

export default function CriteriaBreakdown({ criteria }: Props) {
  return (
    <div>
      <h3 className="text-sm font-semibold text-slate-700 uppercase tracking-wider mb-1">
        Criteria Breakdown
      </h3>
      <p className="text-xs text-slate-400 mb-4">
        Aggregated across all analyzed papers. Score = similarity (1 = no overlap, 5 = identical).
      </p>
      <div className="space-y-4">
        {CRITERIA.map(({ key, label, icon, description }) => {
          const likert = toLikert(criteria[key]);

          return (
            <div key={key}>
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-sm font-medium text-slate-700">
                  {icon} {label}
                </span>
                <div className="flex items-center gap-2">
                  <div className="flex items-center gap-0.5">
                    {[1, 2, 3, 4, 5].map((d) => (
                      <div
                        key={d}
                        className={`w-3 h-3 rounded-full transition-colors ${
                          d <= likert ? likertDotColor(likert) : "bg-slate-200"
                        }`}
                      />
                    ))}
                  </div>
                  <span className={`text-sm font-bold ${likertColor(likert)}`}>
                    {likert}/5
                  </span>
                </div>
              </div>
              <p className="text-xs text-slate-400 leading-snug">{description}</p>
            </div>
          );
        })}
      </div>
    </div>
  );
}
