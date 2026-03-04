import { useState } from "react";
import type { FollowUpQuestion } from "../types/api";

interface Props {
  questions: FollowUpQuestion[];
  onSubmit: (answers: string[]) => void;
  onSkip: () => void;
}

const categoryEmoji: Record<string, string> = {
  problem: "🎯",
  method: "⚙️",
  novelty: "✨",
  application: "🌍",
};

export default function FollowUpQuestions({
  questions,
  onSubmit,
  onSkip,
}: Props) {
  const [answers, setAnswers] = useState<string[]>(
    questions.map(() => "")
  );

  const updateAnswer = (idx: number, value: string) => {
    setAnswers((prev) => {
      const next = [...prev];
      next[idx] = value;
      return next;
    });
  };

  return (
    <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-8">
      <h2 className="text-xl font-semibold text-slate-800 mb-2">
        Clarifying Questions
      </h2>
      <p className="text-slate-500 mb-6 text-sm">
        Please answer these questions to help us better assess your idea's
        originality.
      </p>

      <div className="space-y-5">
        {questions.map((q, i) => (
          <div key={q.id ?? i}>
            <label className="block text-sm font-medium text-slate-700 mb-1.5">
              {categoryEmoji[q.category] ?? "❓"} {q.question}
            </label>
            <textarea
              value={answers[i]}
              onChange={(e) => updateAnswer(i, e.target.value)}
              rows={3}
              className="w-full rounded-lg border-2 border-slate-200 bg-slate-50 px-3 py-2 text-slate-800 text-sm placeholder:text-slate-400 focus:border-indigo-500 focus:ring-2 focus:ring-indigo-500/20 focus:outline-none resize-none transition-colors"
            />
          </div>
        ))}
      </div>

      <div className="flex items-center gap-3 mt-6">
        <button
          onClick={() => onSubmit(answers)}
          className="px-6 py-2.5 bg-gradient-to-r from-indigo-600 to-purple-600 text-white font-semibold rounded-lg shadow-md hover:shadow-lg hover:-translate-y-0.5 transition-all"
        >
          Continue →
        </button>
        <button
          onClick={() => onSkip()}
          className="px-4 py-2.5 text-slate-500 hover:text-slate-700 text-sm font-medium transition-colors"
        >
          Skip questions
        </button>
      </div>
    </div>
  );
}
