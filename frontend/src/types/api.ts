export interface AnalyzeRequest {
  user_idea: string;
  papers_per_query: number;
  embedding_topk: number;
  rerank_topk: number;
  final_papers: number;
  use_reranker: boolean;
}

export interface FollowUpQuestion {
  id: number;
  question: string;
  category: string;
}

export type JobStatus =
  | "generating_questions"
  | "waiting_for_answers"
  | "processing"
  | "completed"
  | "error";

export interface MatchedSection {
  chunk_id: string;
  paper_id: string;
  paper_title: string;
  heading: string;
  text_snippet: string;
  similarity: number;
  reason: string;
}

export interface SentenceAnnotation {
  index: number;
  sentence: string;
  originality_score: number;
  overlap_score: number;
  label: "high" | "medium" | "low";
  linked_sections: MatchedSection[];
}

export interface CriteriaScores {
  problem_similarity: number;
  method_similarity: number;
  domain_overlap: number;
  contribution_similarity: number;
}

export interface PaperDetail {
  paper_id: string;
  arxiv_id: string;
  title: string;
  abstract: string;
  url: string;
  pdf_url: string;
  authors: string[];
  categories: string[];
  is_processed: boolean;
  overall_overlap_score?: number;
  criteria_scores?: CriteriaScores;
}

export interface CostBreakdown {
  estimated_cost_usd: number;
  breakdown: {
    retrieval: number;
    layer1: number;
    layer2: number;
    followup: number;
    keywords: number;
  };
}

export interface AnalysisResults {
  global_originality_score: number;
  global_overlap_score: number;
  label: "high" | "medium" | "low";
  sentence_annotations: SentenceAnnotation[];
  summary: string;
  comprehensive_report: string;
  aggregated_criteria: CriteriaScores | null;
  papers_analyzed: number;
  papers?: PaperDetail[];
  cost: CostBreakdown;
  total_processing_time: number;
  reality_check?: {
    already_exists: boolean;
    confidence: number;
    existing_examples: Array<{
      name: string;
      description: string;
      similarity: number;
    }>;
  };
  reality_check_warning?: string;
  stats?: {
    query_variants: number;
    total_fetched: number;
    unique_after_dedup: number;
    after_rerank: number;
    papers_found: number;
    papers_analyzed: number;
    papers_processed: number;
    total_chunks: number;
    keywords: string[];
  };
}

export interface JobStatusResponse {
  job_id: string;
  status: JobStatus;
  progress: number;
  progress_message: string;
  questions: FollowUpQuestion[] | null;
  reality_check: {
    warning: string;
    result: Record<string, unknown>;
  } | null;
  results: AnalysisResults | null;
  error: string | null;
  stats: Record<string, unknown> | null;
}

export interface PipelineSettings {
  papers_per_query: number;
  embedding_topk: number;
  rerank_topk: number;
  final_papers: number;
  use_reranker: boolean;
}
