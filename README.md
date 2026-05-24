# Hypothetica

**Research Originality Assessment Tool**

Hypothetica is an AI-powered system that evaluates the originality of research ideas by analyzing them against existing academic literature, open-source repositories, and patents. It helps researchers, students, and innovators assess whether their concepts are novel or if similar work already exists.

## Features

- **Multi-Source Evidence**: Search arXiv, OpenAlex, GitHub repositories, and Google Patents in one pipeline
- **Chat Interview**: AI-driven clarification phase to refine the research idea before analysis
- **Reality Check**: Preliminary LLM screening to flag obviously existing work
- **Two-Layer Analysis**: Per-paper criterion scoring (Layer 1) followed by deterministic aggregation (Layer 2)
- **Criteria-Based Scoring**: Problem, method, domain, and contribution evaluated independently per paper
- **Sentence Annotations**: UI-level sentence highlighting showing which parts of the idea overlap with evidence
- **Real-Time Progress**: Live SSE stream with detailed pipeline stage updates
- **Cost Tracking**: Per-job token usage and API cost monitoring

## Architecture

### Backend (Python / FastAPI)

**Services**
- `AnalysisService` — orchestrates the full pipeline: interview → search → process → score → report
- `PaperSearchService` — multi-adapter search dispatch and result deduplication
- `PaperProcessingService` — PDF download, text extraction, chunking, and embedding
- `OriginalityService` — runs Layer 1 and Layer 2 analysis, returns scored results
- `AdapterService` — unified interface over all source adapters
- `GitHubService` — GitHub-specific search, filtering, and README retrieval
- `BenchmarkRunService` — headless analysis runs for benchmark evaluation

**Source Adapters** (`app/adapters/`)
| Adapter | Source | Notes |
|---|---|---|
| `arxiv` | arXiv preprint server | Default source |
| `openalex` | OpenAlex scholarly metadata | Requires `OPENALEX_MAILTO` for better rate limits |
| `github` | GitHub repositories | Requires `GITHUB_TOKEN` |
| `patents` | Google Patents via SerpApi | Requires `SERPAPI_API_KEY` |

**Agents** (`app/agents/`)
| Agent | Role |
|---|---|
| `RealityCheckAgent` | Preliminary LLM screening before full pipeline |
| `FollowUpAgent` | Conversational interview to clarify the research idea |
| `QueryVariantAgent` | Generates diverse search query variants from the refined idea |
| `RelevantPaperSelectorAgent` | Filters retrieved papers for relevance |
| `HeadingSelectorAgent` | Identifies relevant sections within papers |
| `Layer1Agent` | 5 stateless LLM calls per paper (4 criteria + sentence-level) |
| `Layer2Agent` | Summary and narrative generation from scored results |
| `ReportGeneratorAgent` | Compiles the final originality report |
| `GitHubQueryAgent` | Generates GitHub-specific search queries |
| `GitHubSynthesisAgent` | Synthesizes GitHub evidence into structured findings |
| `RepoRelevanceAgent` | Scores individual GitHub repos for relevance |

**Retrieval Pipeline** (`app/retrieval/`)
- Embedding model: `intfloat/e5-base-v2` (auto-detects CUDA / MPS / CPU)
- Vector store: ChromaDB (in-memory)
- Reranking: cross-encoder reranker, top-20 → top-5 final papers

### Scoring Pipeline

Layer 1 runs 5 independent LLM calls per paper:
- **4 criterion calls**: problem, method, domain, contribution — each scored on a 1–5 Likert scale
- **1 sentence call**: per-sentence overlap anchored to the criterion scores

Layer 2 aggregation (deterministic):
```
paper_threat     = 0.5 * max(criteria) + 0.5 * weighted_mean(criteria)
global_overlap   = 0.7 * max(paper_threats) + 0.3 * mean(paper_threats)
originality      = (1 - global_overlap ^ 1.5) * 100
```

Criteria weights: contribution=0.45, method=0.30, problem=0.15, domain=0.10

Guardrails:
- Any criterion = Likert 5 → overlap floor 0.65
- 2+ criteria ≥ Likert 4 → overlap floor 0.50

Score bands: 0–40 = low originality, 40–70 = medium, 70–100 = high

### Frontend (React / TypeScript)

Built with React 19, TypeScript, Vite, and Tailwind CSS 4.

**Components** (`frontend/src/components/`)
| Component | Purpose |
|---|---|
| `IdeaInput` | Initial idea submission form |
| `SourceSelection` | Source adapter picker |
| `SettingsPanel` | Pipeline configuration |
| `ChatInterview` | Conversational follow-up interface |
| `PipelineProgress` | Real-time stage progress display |
| `ResultsView` | Top-level results layout |
| `OriginalityGauge` | Visual score dial |
| `PaperTable` | Per-paper breakdown table |
| `SimilarPapersSection` | Similar paper cards with evidence |
| `CriteriaBreakdown` | Per-criterion score visualization |
| `SentenceHighlighting` | Color-coded sentence overlap annotations |
| `HighlightedIdea` | Annotated idea text with overlap markers |
| `MatchesModal` | RAG chunk matches for a selected sentence |
| `GitHubEvidence` | GitHub-specific evidence display |
| `Header` | App header |

### Infrastructure

- **Supabase**: persists analysis queries and benchmark results
- **ChromaDB**: in-memory vector store for document chunks during analysis
- **FastAPI**: async API with SSE support
- **Nginx**: serves frontend and proxies API requests (via `frontend/Dockerfile`)

## Installation & Setup

### Prerequisites
- Python 3.11
- Google AI API key (Gemini 2.5 Flash)

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/HarunHudaiTan/Hypothetica.git
   cd Hypothetica
   ```

2. **Set up environment**:
   ```bash
   mkdir -p envfiles
   cat > envfiles/.env << EOF
   GOOGLE_API_KEY=your_google_api_key
   # Optional — enables additional sources:
   GITHUB_TOKEN=your_github_token
   SERPAPI_API_KEY=your_serpapi_key
   SUPABASE_URL=your_supabase_url
   SUPABASE_SERVICE_ROLE_KEY=your_supabase_key
   OPENALEX_MAILTO=your_email
   EOF
   ```

### Development Setup

**Backend**
```bash
cd backend
pip install -r requirements.txt
python main.py
```

**Frontend**
```bash
cd frontend
npm install
npm run dev
```

## Usage

1. **Submit idea** — describe your research concept
2. **Select source** — choose arXiv, OpenAlex, GitHub, or Patents
3. **Chat interview** — answer AI follow-up questions to refine the idea (or skip)
4. **Monitor progress** — watch real-time pipeline stages
5. **Review results** — examine originality score, per-paper breakdowns, sentence-level overlaps, and matched evidence

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/analyze` | Start a new analysis job |
| `GET` | `/api/analyze/{job_id}/status` | Poll job status and results |
| `GET` | `/api/analyze/{job_id}/stream` | SSE stream for real-time updates |
| `POST` | `/api/analyze/{job_id}/chat` | Send a chat message during interview |
| `POST` | `/api/analyze/{job_id}/answers` | Submit answers and start analysis |
| `POST` | `/api/analyze/{job_id}/finalize` | End interview early and start analysis |
| `POST` | `/api/analyze/{job_id}/matches` | Get RAG chunk matches for a sentence |
| `GET` | `/api/adapters` | List available source adapters |
| `GET` | `/api/health` | Health check |

**Example**
```bash
# Start analysis
curl -X POST http://localhost:8005/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"user_idea": "A transformer-based model for protein folding prediction"}'

# Stream progress
curl http://localhost:8005/api/analyze/{job_id}/stream

# Get final results
curl http://localhost:8005/api/analyze/{job_id}/status
```

## Configuration

Key parameters in `backend/core/config.py`:

| Parameter | Default | Description |
|---|---|---|
| `PAPERS_PER_QUERY_VARIANT` | 150 | Papers fetched per search query variant |
| `RERANK_TOPK` | 20 | Papers after reranking |
| `MAX_PAPERS_TO_ANALYZE` | 5 | Papers sent to Layer 1 |
| `EMBEDDING_MODEL` | `intfloat/e5-base-v2` | Sentence embedding model |
| `LLM_MODEL` | `gemini-2.5-flash` | LLM for all agents |
| `HIGH_OVERLAP_THRESHOLD` | 0.7 | Overlap score = low originality |
| `OVERLAP_CURVE_POWER` | 1.5 | Power curve exponent for final score |

## Benchmarks

Benchmark data and results live in `benchmarks/`. Sources: arXiv, OpenAlex, GitHub.

```bash
cd benchmarks
python run_benchmark.py          # run evaluation
python sweep_scoring_params.py   # parameter sweep
```

Results stored in `benchmarks/results_v3/`.

## Database Migrations

SQL migrations in `db/migrations/` — apply against Supabase:
- `001_create_queries_table.sql`
- `002_add_github_analysis.sql`
- `003_benchmark_layer_outputs.sql`
- `004_benchmark_table.sql`

## License

[Add license information here]
