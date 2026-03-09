-- Add GitHub analysis column to queries table
-- Run this in Supabase SQL Editor after 001_create_queries_table.sql

alter table public.queries
  add column if not exists github_analysis jsonb;
