import { useState, useCallback } from "react";
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
  const charCount = idea.length;
  const isValid = charCount >= 50 && evidenceSelection !== null;

  const handleSourcesChange = useCallback((sel: EvidenceSelection | null) => {
    setEvidenceSelection(sel);
  }, []);

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-8">
        <h2 className="text-xl font-semibold text-slate-800 mb-2">
          Enter Your Research Idea
        </h2>
        <p className="text-slate-500 mb-4 text-sm">
          Describe your research idea in detail. The more specific, the better the
          analysis.
        </p>

        <textarea
          value={idea}
          onChange={(e) => setIdea(e.target.value)}
          placeholder="Example: I want to develop a multimodal retrieval-augmented generation (RAG) system that can process and reason over both text documents and images simultaneously..."
          rows={7}
          disabled={disabled}
          className="w-full rounded-xl border-2 border-slate-200 bg-slate-50 px-4 py-3 text-slate-800 text-base placeholder:text-slate-400 focus:border-indigo-500 focus:ring-2 focus:ring-indigo-500/20 focus:outline-none resize-none transition-colors disabled:opacity-50"
        />

        <div className="flex items-center justify-between mt-3">
          <span
            className={`text-xs ${charCount >= 50 ? "text-green-600" : "text-slate-400"}`}
          >
            {charCount}/50 characters minimum
          </span>
          <button
            onClick={() =>
              evidenceSelection && onSubmit(idea, evidenceSelection)
            }
            disabled={disabled || !isValid}
            className="px-6 py-2.5 bg-gradient-to-r from-indigo-600 to-purple-600 text-white font-semibold rounded-lg shadow-md hover:shadow-lg hover:-translate-y-0.5 transition-all disabled:opacity-50 disabled:pointer-events-none"
          >
            Analyze Originality
          </button>
        </div>
      </div>

      <SourceSelection 
        onSourcesChange={handleSourcesChange}
        disabled={disabled}
      />
    </div>
  );
}
