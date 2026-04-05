import { useState, useEffect } from "react";

interface Source {
  name: string;
  description: string;
  available: boolean;
}

interface Props {
  onSourcesChange: (sources: string[]) => void;
  disabled: boolean;
}

export default function SourceSelection({ onSourcesChange, disabled }: Props) {
  const [sources, setSources] = useState<Record<string, Source>>({});
  const [sourcesAvailability, setSourcesAvailability] = useState<Record<string, boolean>>({});
  const [selectedSources, setSelectedSources] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchAvailableSources();
  }, []);

  useEffect(() => {
    onSourcesChange(selectedSources);
  }, [selectedSources, onSourcesChange]);

  const fetchAvailableSources = async () => {
    try {
      setLoading(true);
      const response = await fetch("/api/sources");
      if (!response.ok) {
        throw new Error("Failed to fetch sources");
      }
      const data = await response.json();
      setSources(data.all_sources);
      setSourcesAvailability(data.sources);
      
      // Only set default if no sources are currently selected
      if (selectedSources.length === 0) {
        const available = Object.keys(data.sources).filter(key => data.sources[key]);
        if (available.length > 0) {
          // Default to all available sources instead of just arxiv
          setSelectedSources(available);
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
      console.error("Failed to fetch sources:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleSourceToggle = (sourceName: string) => {
    setSelectedSources(prev => {
      if (prev.includes(sourceName)) {
        // Don't allow deselecting if it's the only source
        if (prev.length === 1) {
          return prev;
        }
        return prev.filter(s => s !== sourceName);
      } else {
        return [...prev, sourceName];
      }
    });
  };

  if (loading) {
    return (
      <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
        <div className="animate-pulse">
          <div className="h-6 bg-slate-200 rounded mb-4"></div>
          <div className="space-y-3">
            <div className="h-4 bg-slate-200 rounded"></div>
            <div className="h-4 bg-slate-200 rounded w-3/4"></div>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-white rounded-2xl shadow-sm border border-red-200 p-6">
        <div className="text-red-600">
          <p className="font-medium">Error loading sources</p>
          <p className="text-sm mt-1">{error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
      <h2 className="text-xl font-semibold text-slate-800 mb-2">
        Select Paper Sources
      </h2>
      <p className="text-slate-500 mb-4 text-sm">
        Choose which sources to search for related papers and patents. At least one source must be selected.
      </p>

      <div className="space-y-3">
        {Object.entries(sources).map(([key, source]) => {
          const isSelected = selectedSources.includes(key);
          const isAvailable = sourcesAvailability[key] ?? false;
          
          return (
            <div
              key={key}
              className={`
                flex items-start space-x-3 p-3 rounded-lg border-2 transition-all
                ${!isAvailable 
                  ? 'border-slate-100 bg-slate-50 opacity-60' 
                  : isSelected 
                    ? 'border-indigo-500 bg-indigo-50' 
                    : 'border-slate-200 hover:border-slate-300'
                }
              `}
            >
              <input
                type="checkbox"
                id={key}
                checked={isSelected}
                disabled={disabled || !isAvailable}
                onChange={() => handleSourceToggle(key)}
                className={`
                  mt-1 w-4 h-4 rounded border-2 text-indigo-600 focus:ring-indigo-500 focus:ring-offset-0
                  ${!isAvailable ? 'cursor-not-allowed opacity-50' : 'cursor-pointer'}
                `}
              />
              <div className="flex-1">
                <label 
                  htmlFor={key}
                  className={`
                    font-medium text-sm cursor-pointer
                    ${!isAvailable ? 'text-slate-400' : 'text-slate-700'}
                  `}
                >
                  {source.name}
                  {!isAvailable && (
                    <span className="ml-2 text-xs bg-slate-200 text-slate-600 px-2 py-1 rounded">
                      Not Available
                    </span>
                  )}
                </label>
                <p className="text-xs text-slate-500 mt-1">
                  {source.description}
                </p>
                {!isAvailable && (
                  <p className="text-xs text-amber-600 mt-1">
                    {key === 'google_patents' 
                      ? 'Requires SERPAPI_KEY environment variable'
                      : 'Source configuration needed'
                    }
                  </p>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {selectedSources.length === 0 && (
        <div className="mt-4 p-3 bg-amber-50 border border-amber-200 rounded-lg">
          <p className="text-sm text-amber-800">
            ⚠️ At least one source must be selected to continue
          </p>
        </div>
      )}

      <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
        <p className="text-xs text-blue-700">
          💡 <strong>Tip:</strong> Using multiple sources (arXiv + Google Patents) provides comprehensive coverage of both academic research and intellectual property.
        </p>
      </div>
    </div>
  );
}
