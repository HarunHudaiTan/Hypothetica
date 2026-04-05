-- Benchmark / evaluation: full Layer 1 & Layer 2 payloads and source attribution
-- Run in Supabase SQL Editor after 002_add_github_analysis.sql

-- Align with API naming (001 used global_overlap_score; backend sends global_similarity_score)
alter table public.queries
  add column if not exists global_similarity_score double precision;

update public.queries
set global_similarity_score = coalesce(global_similarity_score, global_overlap_score)
where global_similarity_score is null;

alter table public.queries
  add column if not exists layer1_results jsonb;

alter table public.queries
  add column if not exists layer2_full jsonb;

alter table public.queries
  add column if not exists search_funnel jsonb;

alter table public.queries
  add column if not exists selected_sources jsonb;

alter table public.queries
  add column if not exists source_results jsonb;

alter table public.queries
  add column if not exists patent_warnings jsonb;

comment on column public.queries.layer1_results is 'Per-paper Layer 1 outputs (criteria, sentences, matches)';
comment on column public.queries.layer2_full is 'Full Layer 2 payload: scores, summary, report, per-paper threats, cost snapshot';
comment on column public.queries.selected_sources is 'Literature sources requested for retrieval, e.g. ["arxiv","google_patents"]';
comment on column public.queries.source_results is 'Counts of unique PDF-backed hits per source before semantic funnel';
