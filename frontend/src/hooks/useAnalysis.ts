import { useState, useCallback, useRef, useEffect } from "react";
import type {
  AnalysisResults,
  ChatMessage,
  FollowUpQuestion,
  JobStatus,
  PipelineSettings,
} from "../types/api";
import {
  startAnalysis,
  submitAnswers,
  sendChatMessage as apiSendChat,
  finalizeInterview as apiFinalizeInterview,
  subscribeToEvents,
  getJobStatus,
} from "../lib/api";

export type AppStep = "input" | "interview" | "processing" | "results";

interface AnalysisState {
  step: AppStep;
  jobId: string | null;
  jobStatus: JobStatus | null;
  progress: number;
  progressMessage: string;
  questions: FollowUpQuestion[];
  chatMessages: ChatMessage[];
  isAiTyping: boolean;
  currentRound: number;
  maxRounds: number;
  results: AnalysisResults | null;
  realityCheck: { warning: string; result: Record<string, unknown> } | null;
  error: string | null;
  userIdea: string;
  githubStatus: string;
}

const DEFAULT_SETTINGS: PipelineSettings = {
  papers_per_query: 150,
  embedding_topk: 100,
  rerank_topk: 20,
  final_papers: 5,
  use_reranker: true,
};

let msgIdCounter = 0;
function nextMsgId() {
  return `msg-${++msgIdCounter}-${Date.now()}`;
}

export function useAnalysis() {
  const [state, setState] = useState<AnalysisState>({
    step: "input",
    jobId: null,
    jobStatus: null,
    progress: 0,
    progressMessage: "",
    questions: [],
    chatMessages: [],
    isAiTyping: true,
    currentRound: 0,
    maxRounds: 3,
    results: null,
    realityCheck: null,
    error: null,
    userIdea: "",
    githubStatus: "",
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
            case "chat_message": {
              const role = event.data.role as "ai" | "user";
              if (role === "ai") {
                const round = (event.data.round as number) ?? 1;
                const newMsg: ChatMessage = {
                  id: nextMsgId(),
                  role: "ai",
                  content: (event.data.content as string) ?? "",
                  category: (event.data.category as string) ?? undefined,
                  round,
                  timestamp: Date.now(),
                };
                setState((s) => ({
                  ...s,
                  step: "interview",
                  chatMessages: [...s.chatMessages, newMsg],
                  isAiTyping: false,
                  currentRound: round,
                }));
              }
              break;
            }
            case "interview_complete":
              setState((s) => ({
                ...s,
                step: "processing",
                isAiTyping: false,
                progress: 0.1,
                progressMessage: "Processing your answers...",
              }));
              break;
            case "questions":
              // Legacy fallback — should not fire in interview mode
              setState((s) => ({
                ...s,
                step: "interview",
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
              status.status === "interviewing" &&
              s.step !== "interview" &&
              s.step !== "processing"
            ) {
              next.step = "interview";
              // Reconstruct chat from conversation_history if available
              const history = (status.stats as Record<string, unknown>)?.conversation_history as
                | Array<Record<string, unknown>>
                | undefined;
              if (history && history.length > 0 && s.chatMessages.length === 0) {
                next.chatMessages = history.map((entry) => ({
                  id: nextMsgId(),
                  role: entry.role as "ai" | "user",
                  content: entry.content as string,
                  category: (entry.category as string) ?? undefined,
                  round: (entry.round as number) ?? 1,
                  timestamp: Date.now(),
                }));
                next.isAiTyping = false;
              }
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
    async (userIdea: string, selectedSources: string[]) => {
      cleanup();
      msgIdCounter = 0;
      setState((s) => ({
        ...s,
        step: "processing",
        jobId: null,
        progress: 0,
        progressMessage: "Starting analysis...",
        githubStatus: "",
        questions: [],
        chatMessages: [],
        isAiTyping: true,
        currentRound: 0,
        results: null,
        error: null,
        userIdea: userIdea,
      }));

      try {
        const { job_id } = await startAnalysis({
          user_idea: userIdea,
          selected_sources: selectedSources,
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

  const sendMessage = useCallback(
    async (text: string) => {
      if (!state.jobId) return;

      const userMsg: ChatMessage = {
        id: nextMsgId(),
        role: "user",
        content: text,
        timestamp: Date.now(),
      };

      setState((s) => ({
        ...s,
        chatMessages: [...s.chatMessages, userMsg],
        isAiTyping: true,
      }));

      try {
        await apiSendChat(state.jobId, text);
      } catch (e: unknown) {
        setState((s) => ({
          ...s,
          isAiTyping: false,
          error: e instanceof Error ? e.message : "Failed to send message",
        }));
      }
    },
    [state.jobId]
  );

  const skipInterview = useCallback(async () => {
    if (!state.jobId) return;
    setState((s) => ({
      ...s,
      step: "processing",
      progress: 0.1,
      progressMessage: "Skipping interview, starting analysis...",
    }));
    try {
      await submitAnswers(state.jobId, []);
      listenToJob(state.jobId);
    } catch (e: unknown) {
      setState((s) => ({
        ...s,
        error: e instanceof Error ? e.message : "Failed to skip interview",
      }));
    }
  }, [state.jobId, listenToJob]);

  const endInterviewEarly = useCallback(async () => {
    if (!state.jobId) return;
    setState((s) => ({
      ...s,
      step: "processing",
      progress: 0.1,
      progressMessage: "Finishing interview, starting analysis...",
    }));
    try {
      await apiFinalizeInterview(state.jobId);
    } catch (e: unknown) {
      setState((s) => ({
        ...s,
        error: e instanceof Error ? e.message : "Failed to end interview",
      }));
    }
  }, [state.jobId]);

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
    msgIdCounter = 0;
    setState({
      step: "input",
      jobId: null,
      jobStatus: null,
      progress: 0,
      progressMessage: "",
      questions: [],
      chatMessages: [],
      isAiTyping: true,
      currentRound: 0,
      maxRounds: 3,
      results: null,
      realityCheck: null,
      error: null,
      userIdea: "",
      githubStatus: "",
    });
  }, [cleanup]);

  return {
    ...state,
    settings,
    setSettings,
    startNewAnalysis,
    sendMessage,
    skipInterview,
    endInterviewEarly,
    submitFollowUpAnswers,
    reset,
  };
}
