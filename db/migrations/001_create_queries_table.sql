-- Hypothetica: Analysis queries table
-- Run this in Supabase SQL Editor: https://supabase.com/dashboard/project/jqdjzvnqvkwyaiqednyf/sql

create table if not exists public.queries (
  id uuid primary key default gen_random_uuid(),
  job_id text unique not null,
  user_idea text not null,
  enriched_idea text,
  followup_questions jsonb,
  followup_answers jsonb,
  global_originality_score int,
  global_overlap_score float,
  label text,
  summary text,
  aggregated_criteria jsonb,
  cost_breakdown jsonb,
  estimated_cost_usd float,
  papers_analyzed int default 0,
  total_processing_time float,
  reality_check jsonb,
  reality_check_warning text,
  stats jsonb,
  sentence_annotations jsonb,
  papers jsonb,
  created_at timestamptz default now()
);

-- Optional: enable RLS (Row Level Security)
-- alter table public.queries enable row level security;

-- Optional: allow service role full access (default when RLS is off)
-- create policy "Service role full access" on public.queries
--   for all using (true) with check (true);

create index if not exists idx_queries_job_id on public.queries (job_id);
create index if not exists idx_queries_created_at on public.queries (created_at desc);
create index if not exists idx_queries_global_score on public.queries (global_originality_score);
