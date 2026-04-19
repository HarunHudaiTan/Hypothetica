import { useState, useCallback, useRef, useEffect } from "react";
import type {
  AnalysisResults,
  EvidenceSelection,
  FollowUpQuestion,
  JobStatus,
  PipelineSettings,
} from "../types/api";
import {
  startAnalysis,
  submitAnswers,
  subscribeToEvents,
  getJobStatus,
} from "../lib/api";

export type AppStep = "input" | "questions" | "processing" | "results";

interface AnalysisState {
  step: AppStep;
  jobId: string | null;
  jobStatus: JobStatus | null;
  progress: number;
  progressMessage: string;
  questions: FollowUpQuestion[];
  results: AnalysisResults | null;
  realityCheck: { warning: string; result: Record<string, unknown> } | null;
  error: string | null;
  userIdea: string;
  githubStatus: string;
  evidenceSelection: EvidenceSelection | null;
}

const DEFAULT_SETTINGS: PipelineSettings = {
  papers_per_query: 150,
  embedding_topk: 100,
  rerank_topk: 20,
  final_papers: 5,
  use_reranker: true,
};

export function useAnalysis() {
  const [state, setState] = useState<AnalysisState>({
    step: "input",
    jobId: null,
    jobStatus: null,
    progress: 0,
    progressMessage: "",
    questions: [],
    results: null,
    realityCheck: null,
    error: null,
    userIdea: "",
    githubStatus: "",
    evidenceSelection: null,
  });

  const [settings, setSettings] = useState<PipelineSettings>(DEFAULT_SETTINGS);
  const unsubRef = useRef<(() => void) | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const cleanup = useCallback(() => {
    unsubRef.current?.();
    unsubRef.current = null;
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  useEffect(() => cleanup, [cleanup]);

  const listenToJob = useCallback(
    (jobId: string) => {
      cleanup();

      const unsub = subscribeToEvents(
        jobId,
        (event) => {
          switch (event.type) {
            case "progress": {
              const msg = (event.data.message as string) ?? "";
              const prog = (event.data.progress as number) ?? 0;
              const isGitHub = msg.includes("GitHub") || msg.includes("repositories");
              setState((s) => ({
                ...s,
                progress: prog > s.progress ? prog : s.progress,
                progressMessage: isGitHub ? s.progressMessage : (msg || s.progressMessage),
                githubStatus: isGitHub ? msg : s.githubStatus,
              }));
              break;
            }
            case "questions":
              setState((s) => ({
                ...s,
                step: "questions",
                questions:
                  (event.data.questions as FollowUpQuestion[]) ?? s.questions,
              }));
              break;
            case "reality_check":
              setState((s) => ({
                ...s,
                realityCheck: {
                  warning: (event.data.warning as string) ?? null,
                  result: (event.data.reality_check as Record<string, unknown>) ?? {},
                },
              }));
              break;
            case "completed":
              setState((s) => ({
                ...s,
                step: "results",
                results: (event.data.results as AnalysisResults) ?? null,
                progress: 1,
                progressMessage: "Analysis complete!",
              }));
              break;
            case "error":
              setState((s) => ({
                ...s,
                error: (event.data.error as string) ?? "Unknown error",
              }));
              break;
          }
        },
        () => {}
      );
      unsubRef.current = unsub;

      pollRef.current = setInterval(async () => {
        try {
          const status = await getJobStatus(jobId);
          setState((s) => {
            const next = { ...s, jobStatus: status.status };
            if (status.reality_check) next.realityCheck = status.reality_check;
            if (
              status.status === "waiting_for_answers" &&
              status.questions &&
              s.step !== "questions"
            ) {
              next.step = "questions";
              next.questions = status.questions;
            }
            if (status.status === "completed" && status.results) {
              next.step = "results";
              next.results = status.results;
              next.progress = 1;
            }
            if (status.status === "error") {
              next.error = status.error ?? "Unknown error";
            }
            return next;
          });
          if (
            status.status === "completed" ||
            status.status === "error"
          ) {
            cleanup();
          }
        } catch {
          // ignore poll errors
        }
      }, 2000);
    },
    [cleanup]
  );

  const startNewAnalysis = useCallback(
    async (userIdea: string, selection: EvidenceSelection) => {
      cleanup();
      setState((s) => ({
        ...s,
        step: "processing",
        jobId: null,
        progress: 0,
        progressMessage: "Starting analysis...",
        githubStatus: "",
        questions: [],
        results: null,
        error: null,
        userIdea: userIdea,
        evidenceSelection: selection,
      }));

      try {
        const { job_id } = await startAnalysis({
          user_idea: userIdea,
          selected_adapter: selection.id,
          ...settings,
        });
        setState((s) => ({ ...s, jobId: job_id }));
        listenToJob(job_id);
      } catch (e: unknown) {
        setState((s) => ({
          ...s,
          error: e instanceof Error ? e.message : "Failed to start analysis",
          step: "input",
        }));
      }
    },
    [settings, listenToJob, cleanup]
  );

  const submitFollowUpAnswers = useCallback(
    async (answers: string[]) => {
      if (!state.jobId) return;
      setState((s) => ({
        ...s,
        step: "processing",
        progress: 0.1,
        progressMessage: "Processing your answers...",
      }));
      try {
        await submitAnswers(state.jobId, answers);
        listenToJob(state.jobId);
      } catch (e: unknown) {
        setState((s) => ({
          ...s,
          error:
            e instanceof Error ? e.message : "Failed to submit answers",
        }));
      }
    },
    [state.jobId, listenToJob]
  );

  const reset = useCallback(() => {
    cleanup();
    setState({
      step: "input",
      jobId: null,
      jobStatus: null,
      progress: 0,
      progressMessage: "",
      questions: [],
      results: null,
      realityCheck: null,
      error: null,
      userIdea: "",
      githubStatus: "",
      evidenceSelection: null,
    });
  }, [cleanup]);

  return {
    ...state,
    settings,
    setSettings,
    startNewAnalysis,
    submitFollowUpAnswers,
    reset,
  };
}
