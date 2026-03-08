import { useState, useRef, useEffect } from "react";
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

const categoryLabel: Record<string, string> = {
  problem: "Problem & Research Gap",
  method: "Methodology & Approach",
  novelty: "Innovation & Novelty",
  application: "Application Domain",
};

export default function FollowUpQuestions({
  questions,
  onSubmit,
  onSkip,
}: Props) {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [answers, setAnswers] = useState<string[]>(questions.map(() => ""));
  const [direction, setDirection] = useState<"next" | "prev">("next");
  const [isAnimating, setIsAnimating] = useState(false);
  const [showError, setShowError] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const currentQuestion = questions[currentIndex];
  const isLastQuestion = currentIndex === questions.length - 1;
  const isFirstQuestion = currentIndex === 0;
  const hasAnswer = answers[currentIndex].trim().length > 0;

  useEffect(() => {
    textareaRef.current?.focus();
  }, [currentIndex]);

  const animateTransition = (newIndex: number, dir: "next" | "prev") => {
    setDirection(dir);
    setIsAnimating(true);
    setTimeout(() => {
      setCurrentIndex(newIndex);
      setIsAnimating(false);
    }, 150);
  };

  const goNext = () => {
    if (!hasAnswer) {
      setShowError(true);
      textareaRef.current?.focus();
      return;
    }
    setShowError(false);
    if (isLastQuestion) {
      onSubmit(answers);
    } else {
      animateTransition(currentIndex + 1, "next");
    }
  };

  const goPrev = () => {
    if (!isFirstQuestion) {
      setShowError(false);
      animateTransition(currentIndex - 1, "prev");
    }
  };

  const skipCurrent = () => {
    setShowError(false);
    if (isLastQuestion) {
      onSubmit(answers);
    } else {
      animateTransition(currentIndex + 1, "next");
    }
  };

  const updateAnswer = (value: string) => {
    if (showError && value.trim().length > 0) {
      setShowError(false);
    }
    setAnswers((prev) => {
      const next = [...prev];
      next[currentIndex] = value;
      return next;
    });
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && e.metaKey) {
      e.preventDefault();
      goNext();
    }
  };

  return (
    <div className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden">
      {/* Progress bar */}
      <div className="h-1.5 bg-slate-100">
        <div
          className="h-full bg-gradient-to-r from-indigo-500 to-purple-500 transition-all duration-300 ease-out"
          style={{ width: `${((currentIndex + 1) / questions.length) * 100}%` }}
        />
      </div>

      <div className="p-8">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-xl font-semibold text-slate-800">
              Clarifying Questions
            </h2>
            <p className="text-slate-500 text-sm mt-1">
              Help us better understand your research idea
            </p>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-sm text-slate-400">
              Question {currentIndex + 1} of {questions.length}
            </span>
          </div>
        </div>

        {/* Question dots indicator */}
        <div className="flex items-center justify-center gap-2 mb-6">
          {questions.map((_, idx) => (
            <button
              key={idx}
              onClick={() => {
                if (idx < currentIndex) {
                  animateTransition(idx, "prev");
                } else if (idx > currentIndex) {
                  animateTransition(idx, "next");
                }
              }}
              className={`w-2.5 h-2.5 rounded-full transition-all duration-200 ${
                idx === currentIndex
                  ? "bg-indigo-500 scale-125"
                  : idx < currentIndex && answers[idx].trim()
                  ? "bg-green-400"
                  : "bg-slate-300 hover:bg-slate-400"
              }`}
              title={`Question ${idx + 1}`}
            />
          ))}
        </div>

        {/* Question card with animation */}
        <div className="relative min-h-[280px]">
          <div
            className={`transition-all duration-150 ease-out ${
              isAnimating
                ? direction === "next"
                  ? "opacity-0 -translate-x-8"
                  : "opacity-0 translate-x-8"
                : "opacity-100 translate-x-0"
            }`}
          >
            {/* Category badge */}
            <div className="flex items-center gap-2 mb-4">
              <span className="text-2xl">{categoryEmoji[currentQuestion.category] ?? "❓"}</span>
              <span className="text-xs font-semibold text-indigo-600 uppercase tracking-wider">
                {categoryLabel[currentQuestion.category] ?? currentQuestion.category}
              </span>
            </div>

            {/* Question text */}
            <label className="block text-lg font-medium text-slate-800 mb-4 leading-relaxed">
              {currentQuestion.question}
            </label>

            {/* Answer textarea */}
            <textarea
              ref={textareaRef}
              value={answers[currentIndex]}
              onChange={(e) => updateAnswer(e.target.value)}
              onKeyDown={handleKeyDown}
              rows={4}
              placeholder="Type your answer here..."
              className={`w-full rounded-xl border-2 px-4 py-3 text-slate-800 text-sm placeholder:text-slate-400 focus:outline-none resize-none transition-all ${
                showError
                  ? "border-red-400 bg-red-50 focus:border-red-500 focus:ring-2 focus:ring-red-500/20"
                  : "border-slate-200 bg-slate-50 focus:border-indigo-500 focus:ring-2 focus:ring-indigo-500/20"
              }`}
            />

            {/* Error message */}
            {showError ? (
              <p className="text-xs text-red-500 mt-2 flex items-center gap-1">
                <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
                Please enter your answer or use "Skip" to continue
              </p>
            ) : (
              <p className="text-xs text-slate-400 mt-2">
                Press ⌘+Enter to continue
              </p>
            )}
          </div>
        </div>

        {/* Navigation buttons */}
        <div className="flex items-center justify-between mt-6 pt-6 border-t border-slate-100">
          {/* Left side */}
          <div className="flex items-center gap-2">
            {!isFirstQuestion && (
              <button
                onClick={goPrev}
                className="flex items-center gap-1.5 px-4 py-2 text-slate-500 hover:text-slate-700 hover:bg-slate-100 rounded-lg text-sm font-medium transition-all"
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M15 19l-7-7 7-7" />
                </svg>
                Back
              </button>
            )}
          </div>

          {/* Right side */}
          <div className="flex items-center gap-3">
            {/* Skip this question */}
            <button
              onClick={skipCurrent}
              className="px-4 py-2 text-slate-500 hover:text-slate-700 text-sm font-medium transition-colors"
            >
              Skip
            </button>

            {/* Skip all */}
            <button
              onClick={onSkip}
              className="px-4 py-2 text-slate-400 hover:text-slate-600 text-sm transition-colors"
            >
              Skip all →
            </button>

            {/* Next / Submit */}
            <button
              onClick={goNext}
              className={`flex items-center gap-2 px-6 py-2.5 rounded-lg font-semibold shadow-md transition-all ${
                hasAnswer
                  ? "bg-gradient-to-r from-indigo-600 to-purple-600 text-white hover:shadow-lg hover:-translate-y-0.5"
                  : "bg-slate-200 text-slate-500"
              }`}
            >
              {isLastQuestion ? "Submit Answers" : "Next"}
              {!isLastQuestion && (
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
                </svg>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
