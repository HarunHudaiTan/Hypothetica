# Hypothetica Database (Supabase)

Analysis queries are persisted to Supabase when using the FastAPI backend (React frontend).

## Setup

1. **Run migrations** in Supabase SQL Editor ([SQL](https://supabase.com/dashboard/project/jqdjzvnqvkwyaiqednyf/sql)), in order:
   - `db/migrations/001_create_queries_table.sql`
   - `db/migrations/002_add_github_analysis.sql`
   - `db/migrations/003_benchmark_layer_outputs.sql` — full Layer 1 / Layer 2 JSON, funnel, and per-source counts

2. **Environment variables** (in `.env`):
   - `SUPABASE_URL` – Project URL (default: https://jqdjzvnqvkwyaiqednyf.supabase.co)
   - `SUPABASE_SERVICE_ROLE_KEY` – Service role or secret key

If Supabase is not configured, the API still works; queries are simply not persisted.
