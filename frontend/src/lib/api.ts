import type {
  AnalyzeRequest,
  JobStatusResponse,
  MatchedSection,
} from "../types/api";

const BASE = "/api";

export async function startAnalysis(
  req: AnalyzeRequest
): Promise<{ job_id: string }> {
  const res = await fetch(`${BASE}/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail ?? `Request failed (${res.status})`);
  }
  return res.json();
}

export async function getJobStatus(
  jobId: string
): Promise<JobStatusResponse> {
  const res = await fetch(`${BASE}/analyze/${jobId}/status`);
  if (!res.ok) throw new Error(`Failed to fetch status (${res.status})`);
  return res.json();
}

export async function submitAnswers(
  jobId: string,
  answers: string[]
): Promise<void> {
  const res = await fetch(`${BASE}/analyze/${jobId}/answers`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ answers }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail ?? `Failed to submit answers (${res.status})`);
  }
}

export async function sendChatMessage(
  jobId: string,
  message: string
): Promise<void> {
  const res = await fetch(`${BASE}/analyze/${jobId}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail ?? `Failed to send message (${res.status})`);
  }
}

export async function finalizeInterview(jobId: string): Promise<void> {
  const res = await fetch(`${BASE}/analyze/${jobId}/finalize`, {
    method: "POST",
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail ?? `Failed to finalize interview (${res.status})`);
  }
}

export async function getSentenceMatches(
  jobId: string,
  sentence: string,
  topK = 5
): Promise<MatchedSection[]> {
  const res = await fetch(`${BASE}/analyze/${jobId}/matches`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sentence, top_k: topK }),
  });
  if (!res.ok) throw new Error(`Failed to fetch matches (${res.status})`);
  const data = await res.json();
  return data.matches;
}

export function subscribeToEvents(
  jobId: string,
  onEvent: (event: { type: string; data: Record<string, unknown> }) => void,
  onDone: () => void
): () => void {
  const es = new EventSource(`${BASE}/analyze/${jobId}/stream`);

  es.addEventListener("progress", (e) => {
    onEvent({ type: "progress", data: JSON.parse(e.data) });
  });

  es.addEventListener("questions", (e) => {
    onEvent({ type: "questions", data: JSON.parse(e.data) });
  });

  es.addEventListener("chat_message", (e) => {
    onEvent({ type: "chat_message", data: JSON.parse(e.data) });
  });

  es.addEventListener("interview_complete", (e) => {
    onEvent({ type: "interview_complete", data: JSON.parse(e.data) });
  });

  es.addEventListener("reality_check", (e) => {
    onEvent({ type: "reality_check", data: JSON.parse(e.data) });
  });

  es.addEventListener("completed", (e) => {
    onEvent({ type: "completed", data: JSON.parse(e.data) });
    es.close();
    onDone();
  });

  es.addEventListener("error", (e) => {
    if (e instanceof MessageEvent && e.data) {
      onEvent({ type: "error", data: JSON.parse(e.data) });
    }
    es.close();
    onDone();
  });

  es.addEventListener("done", () => {
    es.close();
    onDone();
  });

  return () => es.close();
}
