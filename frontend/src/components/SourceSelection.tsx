import { useState, useEffect, useCallback } from "react";
import { motion } from "motion/react";
import type { EvidenceSelection } from "../types/api";

interface Source {
  name: string;
  description: string;
  evidence_noun_plural?: string;
  evidence_noun_singular?: string;
}

interface Props {
  onSourcesChange: (selection: EvidenceSelection | null) => void;
  disabled: boolean;
}

const ROMAN = ["i", "ii", "iii", "iv", "v", "vi"];

export default function SourceSelection({ onSourcesChange, disabled }: Props) {
  const [sources, setSources] = useState<Record<string, Source>>({});
  const [sourcesAvailability, setSourcesAvailability] = useState<
    Record<string, boolean>
  >({});
  const [selectedSources, setSelectedSources] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchAvailableSources();
  }, []);

  const notifySelection = useCallback(() => {
    const id = selectedSources[0] ?? null;
    if (!id) {
      onSourcesChange(null);
      return;
    }
    const meta = sources[id];
    onSourcesChange({
      id,
      displayName: meta?.name ?? id,
      nounPlural: meta?.evidence_noun_plural ?? "papers",
      nounSingular: meta?.evidence_noun_singular ?? "paper",
    });
  }, [selectedSources, sources, onSourcesChange]);

  useEffect(() => {
    notifySelection();
  }, [notifySelection]);

  const fetchAvailableSources = async () => {
    try {
      setLoading(true);
      const response = await fetch("/api/sources");
      if (!response.ok) throw new Error("Failed to fetch sources");
      const data = await response.json();
      setSources(data.all_sources);
      setSourcesAvailability(data.sources);

      if (selectedSources.length === 0) {
        const available = Object.keys(data.sources).filter(
          (key) => data.sources[key]
        );
        if (available.length > 0) {
          setSelectedSources([available[0]]);
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  const handleSourceToggle = (sourceName: string) => {
    setSelectedSources([sourceName]);
  };

  if (loading) {
    return (
      <div className="space-y-3">
        {[0, 1, 2].map((i) => (
          <div
            key={i}
            className="h-16 bg-[color:var(--color-paper-shade)] animate-pulse"
            style={{ animationDelay: `${i * 120}ms` }}
          />
        ))}
      </div>
    );
  }

  if (error) {
    return (
      <div className="border-l-2 border-[color:var(--color-vermillion)] pl-4 py-3 bg-[color:var(--color-paper-shade)]">
        <p className="small-caps text-[color:var(--color-vermillion)]">
          Erratum · sources unavailable
        </p>
        <p className="text-sm font-body text-[color:var(--color-ink-soft)] mt-1">
          {error}
        </p>
      </div>
    );
  }

  const entries = Object.entries(sources);

  return (
    <div>
      <div className="space-y-0">
        {entries.map(([key, source], idx) => {
          const isSelected = selectedSources.includes(key);
          const isAvailable = sourcesAvailability[key] ?? false;

          return (
            <motion.button
              key={key}
              type="button"
              onClick={() => isAvailable && handleSourceToggle(key)}
              disabled={disabled || !isAvailable}
              initial={{ opacity: 0, x: -8 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5, delay: 0.1 + idx * 0.08 }}
              className={`relative w-full text-left flex items-baseline gap-6 py-5 border-b border-[color:var(--color-rule)] transition-all group ${
                isAvailable
                  ? "cursor-pointer hover:bg-[color:var(--color-paper-shade)]/60"
                  : "opacity-40 cursor-not-allowed"
              }`}
            >
              {/* Selection ribbon */}
              <span
                aria-hidden
                className={`absolute left-0 top-0 bottom-0 w-1 transition-all ${
                  isSelected
                    ? "bg-[color:var(--color-vermillion)]"
                    : "bg-transparent group-hover:bg-[color:var(--color-rule)]"
                }`}
              />

              <span className="small-caps text-[color:var(--color-ink-fade)] w-8 flex-shrink-0 pl-3">
                {ROMAN[idx] ?? idx + 1}
              </span>

              <span className="flex-1 min-w-0">
                <span className="flex items-baseline gap-3 flex-wrap">
                  <span
                    className={`font-display text-xl md:text-2xl tracking-tight ${
                      isSelected
                        ? "text-[color:var(--color-vermillion)]"
                        : "text-[color:var(--color-ink)]"
                    }`}
                  >
                    {source.name}
                  </span>
                  {!isAvailable && (
                    <span className="small-caps text-[10px] text-[color:var(--color-ink-fade)] border border-[color:var(--color-rule)] px-2 py-0.5">
                      unavailable
                    </span>
                  )}
                </span>
                <span className="block mt-1 font-body italic text-sm text-[color:var(--color-ink-fade)]">
                  {source.description}
                </span>
              </span>

              <span className="flex-shrink-0 pr-3">
                {isSelected ? (
                  <span className="font-mono text-xs small-caps text-[color:var(--color-vermillion)]">
                    selected ●
                  </span>
                ) : isAvailable ? (
                  <span className="font-mono text-[color:var(--color-ink-fade)] opacity-0 group-hover:opacity-100 transition-opacity">
                    →
                  </span>
                ) : null}
              </span>
            </motion.button>
          );
        })}
      </div>

      <p className="mt-6 font-display italic text-sm text-[color:var(--color-ink-fade)]">
        ❡ One archive may be tried at a time. Match the source to the work ·
        peer literature, prior art, or implementation.
      </p>
    </div>
  );
}
