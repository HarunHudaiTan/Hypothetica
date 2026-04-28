-- New evaluation table (API + runner). Keeps your existing public.benchmark untouched.
-- Inserts use public.benchmark2.

create table if not exists public.benchmark2 (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  benchmark_run_id text,
  case_id text not null,
  source text not null,
  domain text,
  idea text,
  true_label text,
  predicted_label text,
  api_label text,
  originality_score int,
  global_similarity_score double precision,
  likert_problem_similarity int,
  likert_method_similarity int,
  likert_domain_overlap int,
  likert_contribution_similarity int,
  criteria_problem_similarity double precision,
  criteria_method_similarity double precision,
  criteria_domain_overlap double precision,
  criteria_contribution_similarity double precision,
  layer1_results jsonb,
  layer2_full jsonb,
  papers jsonb,
  selected_sources jsonb,
  source_results jsonb,
  search_funnel jsonb,
  stats jsonb,
  sentence_annotations jsonb,
  papers_analyzed int,
  total_processing_time double precision,
  cost_breakdown jsonb,
  summary text,
  comprehensive_report text,
  job_id text
);

alter table public.benchmark2 add column if not exists benchmark_run_id text;

create index if not exists benchmark2_run_id_idx on public.benchmark2 (benchmark_run_id);
create index if not exists benchmark2_case_id_idx on public.benchmark2 (case_id);
create index if not exists benchmark2_source_idx on public.benchmark2 (source);
