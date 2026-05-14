import { useState, useEffect } from "react";
import { AnimatePresence, motion } from "motion/react";
import { useAnalysis } from "./hooks/useAnalysis";
import Header from "./components/Header";
import SettingsPanel from "./components/SettingsPanel";
import IdeaInput from "./components/IdeaInput";
import ChatInterview from "./components/ChatInterview";
import PipelineProgress from "./components/PipelineProgress";
import ResultsView from "./components/ResultsView";

function CursorRing() {
  const [pos, setPos] = useState({ x: -100, y: -100 });
  const [shown, setShown] = useState(false);

  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    if (window.matchMedia("(pointer: coarse)").matches) return;
    const move = (e: MouseEvent) => {
      setPos({ x: e.clientX, y: e.clientY });
      setShown(true);
    };
    const leave = () => setShown(false);
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseleave", leave);
    return () => {
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseleave", leave);
    };
  }, []);

  return (
    <div
      aria-hidden
      className="pointer-events-none fixed z-[60] w-6 h-6 rounded-full border border-[color:var(--color-vermillion)] mix-blend-multiply"
      style={{
        left: pos.x - 12,
        top: pos.y - 12,
        opacity: shown ? 0.55 : 0,
        transition: "transform 90ms ease-out, opacity 200ms",
      }}
    />
  );
}

export default function App() {
  const analysis = useAnalysis();
  const [settingsOpen, setSettingsOpen] = useState(false);
  const isProcessing =
    analysis.step === "processing" || analysis.step === "interview";

  return (
    <div className="relative min-h-screen overflow-x-hidden">
      <CursorRing />

      {/* Settings drawer */}
      <AnimatePresence>
        {settingsOpen && (
          <>
            <motion.div
              key="settings-overlay"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.25 }}
              className="fixed inset-0 z-30 bg-[color:var(--color-night)]/45"
              onClick={() => setSettingsOpen(false)}
            />
            <motion.aside
              key="settings-panel"
              initial={{ x: "-100%" }}
              animate={{ x: 0 }}
              exit={{ x: "-100%" }}
              transition={{ type: "spring", stiffness: 280, damping: 32 }}
              className="fixed inset-y-0 left-0 z-40 w-[22rem] bg-[color:var(--color-night)] text-[color:var(--color-paper)] shadow-[12px_0_60px_-20px_rgba(0,0,0,0.5)] flex flex-col"
            >
              <div className="px-6 pt-6 pb-4 flex items-start justify-between border-b border-white/10">
                <div>
                  <p className="small-caps text-[color:var(--color-paper)]/55">
                    Apparatus
                  </p>
                  <h2 className="font-display text-2xl mt-1">
                    Pipeline Controls
                  </h2>
                </div>
                <button
                  onClick={() => setSettingsOpen(false)}
                  className="text-[color:var(--color-paper)]/60 hover:text-[color:var(--color-vermillion-soft)] transition-colors text-sm font-mono"
                  aria-label="Close"
                >
                  ✕
                </button>
              </div>
              <div className="flex-1 overflow-y-auto px-6 py-5">
                <SettingsPanel
                  settings={analysis.settings}
                  onChange={analysis.setSettings}
                  disabled={isProcessing}
                />
              </div>
              <div className="px-6 py-3 border-t border-white/10 small-caps text-[color:var(--color-paper)]/40">
                changes apply to next submission
              </div>
            </motion.aside>
          </>
        )}
      </AnimatePresence>

      {/* Settings float button */}
      <button
        onClick={() => setSettingsOpen(true)}
        className="group fixed top-5 right-5 z-20 w-10 h-10 flex items-center justify-center border border-[color:var(--color-rule)] bg-[color:var(--color-paper)] text-[color:var(--color-ink-fade)] hover:text-[color:var(--color-vermillion)] hover:border-[color:var(--color-vermillion)] transition-colors"
        aria-label="Pipeline settings"
      >
        <svg
          className="w-4 h-4 group-hover:rotate-45 transition-transform duration-500"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          strokeWidth={1.5}
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.066 2.573c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.573 1.066c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.066-2.573c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"
          />
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
          />
        </svg>
      </button>

      {/* Main column */}
      <div className="relative z-10 max-w-6xl mx-auto px-5 md:px-10 pt-6 pb-24">
        <Header />

        <AnimatePresence mode="wait">
          <motion.section
            key={analysis.step}
            initial={{ opacity: 0, y: 18, filter: "blur(6px)" }}
            animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
            exit={{ opacity: 0, y: -10, filter: "blur(6px)" }}
            transition={{ duration: 0.55, ease: [0.2, 0.7, 0.25, 1] }}
            className="relative"
          >
            {analysis.step === "input" && (
              <IdeaInput
                onSubmit={analysis.startNewAnalysis}
                disabled={false}
              />
            )}

            {analysis.step === "interview" && (
              <ChatInterview
                messages={analysis.chatMessages}
                isAiTyping={analysis.isAiTyping}
                currentRound={analysis.currentRound}
                maxRounds={analysis.maxRounds}
                onSendMessage={analysis.sendMessage}
                onSkip={analysis.skipInterview}
                onEndEarly={analysis.endInterviewEarly}
              />
            )}

            {analysis.step === "processing" && (
              <PipelineProgress
                progress={analysis.progress}
                message={analysis.progressMessage}
                error={analysis.error}
                realityCheck={analysis.realityCheck}
                onRetry={analysis.reset}
                evidenceDisplayName={
                  analysis.evidenceSelection?.displayName ?? "arXiv"
                }
                evidenceNounSingular={
                  analysis.evidenceSelection?.nounSingular ?? "paper"
                }
                userIdea={analysis.userIdea}
              />
            )}

            {analysis.step === "results" &&
              analysis.results &&
              analysis.jobId && (
                <ResultsView
                  results={analysis.results}
                  jobId={analysis.jobId}
                  realityCheck={analysis.realityCheck}
                  onNewAnalysis={analysis.reset}
                />
              )}
          </motion.section>
        </AnimatePresence>

        {analysis.error && analysis.step !== "processing" && (
          <div className="mt-8 border-l-2 border-[color:var(--color-vermillion)] pl-4 py-3 bg-[color:var(--color-paper-shade)]">
            <p className="small-caps text-[color:var(--color-vermillion)] mb-1">
              Erratum
            </p>
            <p className="text-sm text-[color:var(--color-ink-soft)] font-body">
              {analysis.error}
            </p>
            <button
              onClick={analysis.reset}
              className="mt-2 text-sm font-display italic text-[color:var(--color-vermillion)] underline decoration-1 underline-offset-4 hover:decoration-2"
            >
              ↺ begin afresh
            </button>
          </div>
        )}

      </div>
    </div>
  );
}
