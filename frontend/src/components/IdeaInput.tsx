import { useState, useCallback } from "react";
import { motion } from "motion/react";
import SourceSelection from "./SourceSelection";
import type { EvidenceSelection } from "../types/api";

interface Props {
  onSubmit: (idea: string, selection: EvidenceSelection) => void;
  disabled: boolean;
}

export default function IdeaInput({ onSubmit, disabled }: Props) {
  const [idea, setIdea] = useState("");
  const [evidenceSelection, setEvidenceSelection] =
    useState<EvidenceSelection | null>(null);
  const [focused, setFocused] = useState(false);
  const charCount = idea.length;
  const isValid = charCount >= 50 && evidenceSelection !== null;

  const handleSourcesChange = useCallback((sel: EvidenceSelection | null) => {
    setEvidenceSelection(sel);
  }, []);

  return (
    <div className="grid grid-cols-12 gap-x-6 gap-y-12">
      {/* Marginalia */}
      <aside className="hidden lg:block col-span-2 pt-2">
        <p className="small-caps text-[color:var(--color-ink-fade)] mb-2">
          § i
        </p>
        <p className="font-display italic text-[color:var(--color-ink-soft)] leading-snug">
          The submission.
        </p>
        <div className="hairline mt-4 mb-4" />
        <p className="font-mono text-[10px] leading-relaxed text-[color:var(--color-ink-fade)]">
          State the hypothesis<br />
          in plain prose.<br />
          Specificity is novelty's<br />
          first witness.
        </p>
      </aside>

      {/* Manuscript page */}
      <motion.article
        initial={{ opacity: 0, y: 18 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7, delay: 0.15 }}
        className="col-span-12 lg:col-span-10 relative"
      >
        <div className="flex items-baseline justify-between mb-3">
          <h2 className="font-display text-3xl md:text-4xl tracking-tight text-[color:var(--color-ink)]">
            On the Matter of Your Hypothesis
          </h2>
          <span className="small-caps text-[color:var(--color-ink-fade)] hidden md:inline">
            chapter one
          </span>
        </div>
        <p className="font-display italic text-base text-[color:var(--color-ink-fade)] mb-6 max-w-2xl">
          Describe the work as you would in a grant abstract · the problem, the
          method, the contribution. Hypothetica reads carefully.
        </p>

        <div
          className={`relative bg-[color:var(--color-paper-shade)] border border-[color:var(--color-rule)] transition-shadow ${
            focused ? "shadow-[0_2px_0_0_var(--color-vermillion)]" : ""
          }`}
        >
          {/* corner ornaments */}
          <span className="absolute top-2 left-3 font-mono text-[10px] text-[color:var(--color-ink-fade)]">
            ms. i
          </span>
          <span className="absolute top-2 right-3 font-mono text-[10px] text-[color:var(--color-ink-fade)]">
            ✦
          </span>

          <textarea
            value={idea}
            onChange={(e) => setIdea(e.target.value)}
            onFocus={() => setFocused(true)}
            onBlur={() => setFocused(false)}
            placeholder="e.g. I propose a multimodal retrieval-augmented generation system that reasons jointly over text documents and accompanying figures, using a shared embedding space trained with contrastive alignment…"
            rows={8}
            disabled={disabled}
            spellCheck
            className="block w-full bg-transparent px-8 pt-9 pb-10 font-body text-[15px] leading-relaxed text-[color:var(--color-ink)] placeholder:text-[color:var(--color-ink-mute)] placeholder:italic resize-none focus:outline-none disabled:opacity-50"
          />
        </div>

        {/* Page meta + submit */}
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mt-4">
          <div className="flex items-center gap-4 font-mono text-[11px] text-[color:var(--color-ink-fade)]">
            <span
              className={
                charCount >= 50
                  ? "text-[color:var(--color-moss)]"
                  : ""
              }
            >
              [{charCount.toString().padStart(4, "0")} / 0050 char min]
            </span>
            <span className="hidden sm:inline opacity-70">·</span>
            <span className="hidden sm:inline opacity-70">
              {idea.trim().split(/\s+/).filter(Boolean).length} words
            </span>
          </div>

          <button
            onClick={() =>
              evidenceSelection && onSubmit(idea, evidenceSelection)
            }
            disabled={disabled || !isValid}
            className="group relative inline-flex items-center gap-3 px-7 py-3.5 bg-[color:var(--color-ink)] text-[color:var(--color-paper)] font-display text-lg tracking-tight shadow-[0_5px_0_0_var(--color-vermillion)] hover:shadow-[0_2px_0_0_var(--color-vermillion)] hover:translate-y-[3px] disabled:opacity-30 disabled:pointer-events-none disabled:shadow-none disabled:translate-y-0 transition-all"
          >
            <span className="font-mono text-sm opacity-70 group-hover:translate-x-0.5 transition-transform">
              →
            </span>
            <span>Submit</span>
          </button>
        </div>
      </motion.article>

      {/* Source ribbon section */}
      <aside className="hidden lg:block col-span-2 pt-2">
        <p className="small-caps text-[color:var(--color-ink-fade)] mb-2">
          § ii
        </p>
        <p className="font-display italic text-[color:var(--color-ink-soft)] leading-snug">
          The corpus.
        </p>
        <div className="hairline mt-4 mb-4" />
        <p className="font-mono text-[10px] leading-relaxed text-[color:var(--color-ink-fade)]">
          Choose the archive<br />
          against which your idea<br />
          shall be cross-examined.
        </p>
      </aside>

      <motion.div
        initial={{ opacity: 0, y: 18 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7, delay: 0.35 }}
        className="col-span-12 lg:col-span-10"
      >
        <div className="flex items-baseline justify-between mb-3">
          <h2 className="font-display text-3xl md:text-4xl tracking-tight text-[color:var(--color-ink)]">
            Choose Your Tribunal
          </h2>
          <span className="small-caps text-[color:var(--color-ink-fade)] hidden md:inline">
            chapter two
          </span>
        </div>
        <p className="font-display italic text-base text-[color:var(--color-ink-fade)] mb-6 max-w-2xl">
          Each archive surfaces different evidence · peer-reviewed papers, prior
          art in patents, or open-source artefacts.
        </p>
        <SourceSelection
          onSourcesChange={handleSourcesChange}
          disabled={disabled}
        />
      </motion.div>
    </div>
  );
}
