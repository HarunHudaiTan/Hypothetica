import type { PipelineSettings } from "../types/api";

interface Props {
  settings: PipelineSettings;
  onChange: (s: PipelineSettings) => void;
  disabled: boolean;
}

function Slider({
  label,
  value,
  min,
  max,
  step,
  help,
  onChange,
  disabled,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  help: string;
  onChange: (v: number) => void;
  disabled: boolean;
}) {
  return (
    <div className="mb-4">
      <div className="flex justify-between items-center mb-1">
        <label className="text-sm font-medium text-slate-300">{label}</label>
        <span className="text-sm font-semibold text-indigo-400">{value}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        disabled={disabled}
        className="w-full h-2 bg-slate-600 rounded-lg appearance-none cursor-pointer accent-indigo-500 disabled:opacity-50"
      />
      <p className="text-xs text-slate-400 mt-1">{help}</p>
    </div>
  );
}

export default function SettingsPanel({ settings, onChange, disabled }: Props) {
  const update = (key: keyof PipelineSettings, value: number | boolean) =>
    onChange({ ...settings, [key]: value });

  return (
    <div className="bg-slate-800 border border-slate-700 rounded-xl p-5">
      <h3 className="text-sm font-semibold text-slate-200 mb-4 uppercase tracking-wider">
        Pipeline Settings
      </h3>

      <Slider
        label="Papers per query"
        value={settings.papers_per_query}
        min={50}
        max={300}
        step={50}
        help="Raw results to retrieve per query variant from the selected evidence source"
        onChange={(v) => update("papers_per_query", v)}
        disabled={disabled}
      />
      <Slider
        label="Embedding top-k"
        value={settings.embedding_topk}
        min={50}
        max={200}
        step={25}
        help="Candidates from embedding similarity"
        onChange={(v) => update("embedding_topk", v)}
        disabled={disabled}
      />
      <Slider
        label="Rerank top-k"
        value={settings.rerank_topk}
        min={10}
        max={50}
        step={5}
        help="Papers after cross-encoder reranking"
        onChange={(v) => update("rerank_topk", v)}
        disabled={disabled}
      />
      <Slider
        label="Final papers"
        value={settings.final_papers}
        min={3}
        max={10}
        step={1}
        help="Papers selected for detailed analysis"
        onChange={(v) => update("final_papers", v)}
        disabled={disabled}
      />

      <label className="flex items-center gap-2 mt-2 cursor-pointer">
        <input
          type="checkbox"
          checked={settings.use_reranker}
          onChange={(e) => update("use_reranker", e.target.checked)}
          disabled={disabled}
          className="rounded border-slate-600 bg-slate-700 text-indigo-500 focus:ring-indigo-500"
        />
        <span className="text-sm text-slate-300">
          Use cross-encoder reranking
        </span>
      </label>
    </div>
  );
}
