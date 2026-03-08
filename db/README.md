# Hypothetica Database (Supabase)

Analysis queries are persisted to Supabase when using the FastAPI backend (React frontend).

## Setup

1. **Run the migration** in Supabase SQL Editor:
   - Go to [Supabase Dashboard](https://supabase.com/dashboard/project/jqdjzvnqvkwyaiqednyf/sql)
   - Copy the contents of `db/migrations/001_create_queries_table.sql`
   - Paste and run

2. **Environment variables** (in `.env`):
   - `SUPABASE_URL` – Project URL (default: https://jqdjzvnqvkwyaiqednyf.supabase.co)
   - `SUPABASE_SERVICE_ROLE_KEY` – Service role or secret key

If Supabase is not configured, the API still works; queries are simply not persisted.
