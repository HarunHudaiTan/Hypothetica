import { useState } from "react";
import { useAnalysis } from "./hooks/useAnalysis";
import Header from "./components/Header";
import SettingsPanel from "./components/SettingsPanel";
import IdeaInput from "./components/IdeaInput";
import FollowUpQuestions from "./components/FollowUpQuestions";
import PipelineProgress from "./components/PipelineProgress";
import ResultsView from "./components/ResultsView";

export default function App() {
  const analysis = useAnalysis();
  const [settingsOpen, setSettingsOpen] = useState(false);
  const isProcessing = analysis.step === "processing" || analysis.step === "questions";

  return (
    <div className="min-h-screen bg-slate-50">
      {/* Settings sidebar */}
      <div
        className={`fixed inset-y-0 left-0 z-40 w-80 bg-slate-900 shadow-2xl transform transition-transform duration-300 ${
          settingsOpen ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <div className="p-5 h-full overflow-y-auto">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-lg font-semibold text-white">Settings</h2>
            <button
              onClick={() => setSettingsOpen(false)}
              className="text-slate-400 hover:text-white transition-colors"
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
          <SettingsPanel
            settings={analysis.settings}
            onChange={analysis.setSettings}
            disabled={isProcessing}
          />
        </div>
      </div>

      {/* Overlay */}
      {settingsOpen && (
        <div
          className="fixed inset-0 z-30 bg-black/30"
          onClick={() => setSettingsOpen(false)}
        />
      )}

      {/* Main content */}
      <div className="max-w-6xl mx-auto px-4 py-8">
        {/* Top bar */}
        <div className="flex justify-end mb-4">
          <button
            onClick={() => setSettingsOpen(true)}
            className="flex items-center gap-2 px-3 py-1.5 text-sm text-slate-600 hover:text-slate-800 bg-white border border-slate-200 rounded-lg shadow-sm hover:shadow transition-all"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.066 2.573c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.573 1.066c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.066-2.573c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
              <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
            Pipeline Settings
          </button>
        </div>

        <Header />

        {/* Step: Input */}
        {analysis.step === "input" && (
          <IdeaInput
            onSubmit={analysis.startNewAnalysis}
            disabled={false}
          />
        )}

        {/* Step: Follow-up Questions */}
        {analysis.step === "questions" && analysis.questions.length > 0 && (
          <FollowUpQuestions
            questions={analysis.questions}
            onSubmit={analysis.submitFollowUpAnswers}
            onSkip={() =>
              analysis.submitFollowUpAnswers(
                analysis.questions.map(() => "")
              )
            }
          />
        )}

        {/* Step: Processing */}
        {analysis.step === "processing" && (
          <PipelineProgress
            progress={analysis.progress}
            message={analysis.progressMessage}
            error={analysis.error}
            realityCheck={analysis.realityCheck}
            githubStatus={analysis.githubStatus}
            onRetry={analysis.reset}
          />
        )}

        {/* Step: Results */}
        {analysis.step === "results" && analysis.results && analysis.jobId && (
          <ResultsView
            results={analysis.results}
            jobId={analysis.jobId}
            userIdea={analysis.userIdea}
            realityCheck={analysis.realityCheck}
            onNewAnalysis={analysis.reset}
          />
        )}

        {/* Global error (not in processing) */}
        {analysis.error && analysis.step !== "processing" && (
          <div className="mt-6 bg-red-50 border border-red-200 rounded-xl p-4">
            <p className="text-red-700 text-sm">{analysis.error}</p>
            <button
              onClick={analysis.reset}
              className="mt-2 text-red-600 text-sm underline"
            >
              Start over
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
