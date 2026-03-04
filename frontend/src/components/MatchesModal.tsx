import { useState, useEffect, useMemo } from "react";
import type { SentenceAnnotation, MatchedSection } from "../types/api";
import { getSentenceMatches } from "../lib/api";

interface Props {
  annotation: SentenceAnnotation;
  jobId: string;
  onClose: () => void;
}

interface PaperGroup {
  paperTitle: string;
  paperId: string;
  matches: MatchedSection[];
}

function groupByPaper(matches: MatchedSection[]): PaperGroup[] {
  const map = new Map<string, PaperGroup>();
  for (const m of matches) {
    const key = m.paper_id || m.paper_title;
    if (!map.has(key)) {
      map.set(key, {
        paperTitle: m.paper_title,
        paperId: m.paper_id,
        matches: [],
      });
    }
    map.get(key)!.matches.push(m);
  }
  return Array.from(map.values());
}

function similarityColor(sim: number): string {
  if (sim >= 0.7) return "bg-red-100 text-red-700 border-red-200";
  if (sim >= 0.4) return "bg-amber-100 text-amber-700 border-amber-200";
  return "bg-green-100 text-green-700 border-green-200";
}

function similarityBadge(sim: number): string {
  if (sim >= 0.7) return "bg-red-500";
  if (sim >= 0.4) return "bg-amber-500";
  return "bg-green-500";
}

export default function MatchesModal({ annotation, jobId, onClose }: Props) {
  const [ragMatches, setRagMatches] = useState<MatchedSection[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;

    async function fetchMatches() {
      try {
        const result = await getSentenceMatches(jobId, annotation.sentence);
        if (!cancelled) setRagMatches(result);
      } catch {
        // RAG fetch failed, will use linked_sections only
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

  const paperGroups = useMemo(() => groupByPaper(allMatches), [allMatches]);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm"
      onClick={(e) => e.target === e.currentTarget && onClose()}
    >
      <div className="bg-white rounded-2xl shadow-2xl max-w-5xl w-full max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="px-6 py-4 border-b border-slate-200 bg-slate-50 flex-shrink-0">
          <div className="flex items-start justify-between">
            <div className="flex-1 mr-4">
              <h3 className="text-lg font-semibold text-slate-800">
                Similarity Analysis
              </h3>
              <p className="text-xs text-slate-500 mt-1">
                Showing which papers have similar content and what specifically
                overlaps
              </p>
            </div>
            <button
              onClick={onClose}
              className="text-slate-400 hover:text-slate-600 transition-colors p-1 rounded-lg hover:bg-slate-200"
            >
              <svg
                className="w-5 h-5"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={2}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M6 18L18 6M6 6l12 12"
                />
              </svg>
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto">
          {/* User sentence */}
          <div className="px-6 pt-5 pb-4">
            <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
              Your Sentence
            </div>
            <div className="bg-indigo-50 border border-indigo-200 rounded-xl px-5 py-4">
              <p className="text-sm text-indigo-900 leading-relaxed font-medium">
                {annotation.sentence}
              </p>
              <div className="flex items-center gap-3 mt-3">
                <span
                  className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold text-white ${similarityBadge(annotation.overlap_score)}`}
                >
                  {Math.round(annotation.overlap_score * 100)}% overlap
                </span>
                <span className="text-xs text-slate-400">
                  Found in {paperGroups.length} paper
                  {paperGroups.length !== 1 ? "s" : ""}
                </span>
              </div>
            </div>
          </div>

          {/* Matches grouped by paper */}
          <div className="px-6 pb-6">
            {loading ? (
              <div className="flex items-center justify-center py-12">
                <div className="w-8 h-8 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin" />
                <span className="ml-3 text-slate-500">
                  Loading matching sources...
                </span>
              </div>
            ) : paperGroups.length === 0 ? (
              <p className="text-center text-slate-400 py-12">
                No detailed matches found for this sentence.
              </p>
            ) : (
              <div className="space-y-5">
                {paperGroups.map((group) => (
                  <PaperMatchGroup
                    key={group.paperId || group.paperTitle}
                    group={group}
                    userSentence={annotation.sentence}
                  />
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function PaperMatchGroup({
  group,
  userSentence: _userSentence,
}: {
  group: PaperGroup;
  userSentence: string;
}) {
  const [expanded, setExpanded] = useState(true);

  return (
    <div className="border border-slate-200 rounded-xl overflow-hidden">
      {/* Paper header */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full bg-slate-50 px-5 py-3 flex items-center justify-between hover:bg-slate-100 transition-colors"
      >
        <div className="flex items-center gap-3 min-w-0">
          <span className="text-base">📄</span>
          <h4 className="text-sm font-semibold text-slate-800 truncate text-left">
            {group.paperTitle}
          </h4>
        </div>
        <div className="flex items-center gap-2 flex-shrink-0 ml-3">
          <span className="text-xs text-slate-500">
            {group.matches.length} match
            {group.matches.length !== 1 ? "es" : ""}
          </span>
          <svg
            className={`w-4 h-4 text-slate-400 transition-transform ${expanded ? "rotate-180" : ""}`}
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={2}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M19 9l-7 7-7-7"
            />
          </svg>
        </div>
      </button>

      {/* Matched passages */}
      {expanded && (
        <div className="divide-y divide-slate-100">
          {group.matches.map((match, i) => (
            <div key={`${match.chunk_id}-${i}`} className="px-5 py-4">
              {/* Section heading + similarity badge */}
              <div className="flex items-center justify-between mb-3">
                <span className="text-xs font-medium text-slate-500">
                  Section: {match.heading || "Unknown"}
                </span>
                <span
                  className={`px-2 py-0.5 rounded-full text-xs font-semibold border ${similarityColor(match.similarity)}`}
                >
                  {Math.round(match.similarity * 100)}% similar
                </span>
              </div>

              {/* The similar paper passage */}
              {match.text_snippet && (
                <div className="bg-slate-50 border border-slate-200 rounded-lg px-4 py-3 mb-3">
                  <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1.5">
                    Similar passage from paper
                  </div>
                  <p className="text-sm text-slate-700 leading-relaxed italic">
                    "{match.text_snippet}"
                  </p>
                </div>
              )}

              {/* Explanation of what's similar */}
              {match.reason && (
                <div className="bg-amber-50 border border-amber-100 rounded-lg px-4 py-3">
                  <div className="text-xs font-semibold text-amber-500 uppercase tracking-wider mb-1.5">
                    What is similar
                  </div>
                  <p className="text-sm text-amber-900 leading-relaxed">
                    {match.reason}
                  </p>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
