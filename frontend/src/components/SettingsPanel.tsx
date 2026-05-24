import type { PipelineSettings } from "../types/api";

interface Props {
  settings: PipelineSettings;
  onChange: (s: PipelineSettings) => void;
  disabled: boolean;
}

function Knob({
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
    <div className="mb-5">
      <div className="flex items-baseline justify-between mb-1">
        <label className="small-caps text-[color:var(--color-paper)]/70">
          {label}
        </label>
        <span className="font-display text-2xl text-[color:var(--color-vermillion-soft)] numeric-tabular">
          {value}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        disabled={disabled}
        className="ink-slider w-full disabled:opacity-40"
      />
      <p className="font-body italic text-[11px] text-[color:var(--color-paper)]/45 leading-snug mt-1">
        {help}
      </p>
    </div>
  );
}

export default function SettingsPanel({
  settings,
  onChange,
  disabled,
}: Props) {
  const update = (key: keyof PipelineSettings, value: number | boolean) =>
    onChange({ ...settings, [key]: value });

  return (
    <div className="text-[color:var(--color-paper)]">
      <p className="font-display italic text-[color:var(--color-paper)]/60 mb-5 text-sm">
        ↓ retrieval and assessment parameters for the next submission
      </p>

      <Knob
        label="Papers per query"
        value={settings.papers_per_query}
        min={50}
        max={300}
        step={50}
        help="minimum fetch budget across variant splits"
        onChange={(v) => update("papers_per_query", v)}
        disabled={disabled}
      />
      <Knob
        label="Papers per variant"
        value={settings.papers_per_variant_conversion}
        min={10}
        max={200}
        step={5}
        help="kept from each variant before deduplication"
        onChange={(v) => update("papers_per_variant_conversion", v)}
        disabled={disabled}
      />
      <Knob
        label="Embedding top-k"
        value={settings.embedding_topk}
        min={50}
        max={200}
        step={25}
        help="candidates retrieved by embedding similarity"
        onChange={(v) => update("embedding_topk", v)}
        disabled={disabled}
      />
      <Knob
        label="Rerank top-k"
        value={settings.rerank_topk}
        min={10}
        max={50}
        step={5}
        help="kept after cross-encoder reranking"
        onChange={(v) => update("rerank_topk", v)}
        disabled={disabled}
      />
      <Knob
        label="Final papers"
        value={settings.final_papers}
        min={3}
        max={10}
        step={1}
        help="selected for full criterion-by-criterion assessment"
        onChange={(v) => update("final_papers", v)}
        disabled={disabled}
      />

      <label className="flex items-center gap-3 cursor-pointer pt-3 border-t border-white/10">
        <input
          type="checkbox"
          checked={settings.use_reranker}
          onChange={(e) => update("use_reranker", e.target.checked)}
          disabled={disabled}
          className="appearance-none w-4 h-4 border border-[color:var(--color-paper)]/40 checked:bg-[color:var(--color-vermillion)] checked:border-[color:var(--color-vermillion)] transition-colors"
        />
        <span className="small-caps text-[color:var(--color-paper)]/75">
          engage cross-encoder reranking
        </span>
      </label>
    </div>
  );
}
