# Hypothetica Project - Object and Class Model

## Project Overview

**Hypothetica** is an AI-powered research originality assessment system that analyzes research ideas against existing academic literature to determine their novelty and originality. The system uses a multi-layered analysis approach combining ArXiv paper retrieval, PDF processing, vector embeddings, and LLM-based evaluation.

## Architecture Overview

The system follows a **pipeline architecture** with the following main components:

1. **Data Models** - Core data structures for papers, analysis results, and chunks
2. **Processing Pipeline** - ArXiv search, PDF processing, and content extraction
3. **RAG System** - Vector storage and semantic retrieval using ChromaDB
4. **Agent System** - LLM-powered analysis agents for different evaluation tasks
5. **Web Interface** - Streamlit-based UI for user interaction

---

## Core Data Models

### 1. Paper Domain Models (`models/paper.py`)

#### `Chunk`
Represents atomic units of text extracted from paper sections for embedding and retrieval.

```python
@dataclass
class Chunk:
    chunk_id: str                    # Unique ID: paper_id_heading_idx_chunk_idx
    paper_id: str                    # Parent paper ID
    heading: str                     # Parent heading text
    heading_index: int               # Index of heading in paper
    chunk_index: int                 # Index of chunk within heading
    text: str                        # The actual chunk text
    char_start: int                  # Start position in original section text
    char_end: int                    # End position in original section text
    is_valid: bool = True            # Whether chunk meets quality thresholds
    quality_reason: Optional[str] = None
```

**Key Methods:**
- `__post_init__()` - Auto-generates chunk_id if not provided

#### `Heading`
Represents a section/heading extracted from a paper with all its content.

```python
@dataclass
class Heading:
    heading_id: str                  # Unique ID: paper_id_heading_idx
    paper_id: str                    # Parent paper ID
    index: int                       # Position in paper (0-indexed)
    level: int                       # Heading level (1-6)
    text: str                        # Heading text (cleaned)
    raw_text: str                    # Original heading text
    section_text: str                # Full text under this heading
    chunks: List[Chunk] = field(default_factory=list)
    is_valid: bool = True
    quality_score: float = 1.0       # 0-1 score based on content quality
    abstract_similarity: Optional[float] = None
```

**Key Methods:**
- `__post_init__()` - Auto-generates heading_id if not provided

#### `Paper`
Main entity representing a research paper with all extracted content.

```python
@dataclass
class Paper:
    paper_id: str                    # Internal ID (e.g., "paper_01")
    arxiv_id: str                    # ArXiv ID (e.g., "2401.12345")
    title: str
    abstract: str
    url: str                         # ArXiv URL
    pdf_url: str                     # PDF URL
    authors: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    published_date: Optional[str] = None
    headings: List[Heading] = field(default_factory=list)
    markdown_content: Optional[str] = None
    is_processed: bool = False
    processing_error: Optional[str] = None
    processed_at: Optional[datetime] = None
```

**Key Methods:**
- `total_chunks` (property) - Total number of chunks across all headings
- `valid_chunks` (property) - All valid chunks from all headings
- `chunk_ids` (property) - List of all chunk IDs for ChromaDB reference
- `get_chunk_metadata()` - Get metadata dict for all chunks (for ChromaDB storage)
- `to_dict()` - Convert to dictionary for serialization

### 2. Analysis Domain Models (`models/analysis.py`)

#### `OriginalityLabel`
Enum for originality classification.

```python
class OriginalityLabel(str, Enum):
    LOW = "low"       # Red - high overlap
    MEDIUM = "medium"  # Yellow - moderate overlap
    HIGH = "high"      # Green - low overlap/novel
```

#### `MatchedSection`
Represents a section/chunk that matches a user's sentence.

```python
@dataclass
class MatchedSection:
    chunk_id: str
    paper_id: str
    paper_title: str
    heading: str
    text_snippet: str              # Relevant excerpt from chunk
    similarity: float              # Cosine similarity score
    reason: str                    # Why this matches
```

#### `CriteriaScores`
TÜBİTAK-style originality criteria scores (0-1, higher = more similar/less original).

```python
@dataclass
class CriteriaScores:
    problem_similarity: float      # How similar is the problem definition
    method_similarity: float       # How similar is the proposed method
    domain_overlap: float          # How much domain/application overlap
    contribution_similarity: float # How similar are the claimed contributions
```

**Key Methods:**
- `to_dict()` - Convert to dictionary
- `average` (property) - Average of all criteria scores

#### `Layer1Result`
Complete Layer 1 analysis result for a single paper.

```python
@dataclass
class Layer1Result:
    paper_id: str
    paper_title: str
    arxiv_id: str
    overall_overlap_score: float   # 0-1, higher = more similar
    criteria_scores: CriteriaScores
    sentence_analyses: List[SentenceAnalysis] = field(default_factory=list)
    tokens_used: int = 0
    processing_time: float = 0.0
```

#### `Layer2Result`
Complete Layer 2 result - global originality assessment.

```python
@dataclass
class Layer2Result:
    global_originality_score: int  # 0-100, higher = more original
    global_overlap_score: float    # 0-1, average overlap
    label: OriginalityLabel        # Overall label
    sentence_annotations: List[SentenceAnnotation] = field(default_factory=list)
    summary: str = ""              # 1-2 sentence explanation
    aggregated_criteria: Optional[CriteriaScores] = None
    papers_analyzed: int = 0
    cost: CostBreakdown = field(default_factory=CostBreakdown)
    total_processing_time: float = 0.0
```

**Key Methods:**
- `to_dict()` - Convert to dictionary for serialization
- `get_sentences_by_label()` - Get sentences with specific label
- `red_sentences`, `yellow_sentences`, `green_sentences` (properties)

---

## Agent System

### Base Agent Class (`Agents/Agent.py`)

#### `Agent`
Base class for all LLM-powered agents using Google Gemini API.

```python
class Agent:
    def __init__(self, system_prompt, top_p, top_k, temperature, 
                 response_mime_type, max_output_tokens=65535,
                 model="gemini-2.5-flash", timebuffer=3, create_chat=True)
```

**Key Methods:**
- `generate_chat_response(prompt)` - Generate response using chat interface
- `generate_text_generation_response(prompt, max_retries=3)` - Generate response with retry logic
- `get_chat_history()` - Retrieve conversation history
- `count_token_price(response)` - Calculate API cost

**Key Features:**
- Automatic rate limiting and retry logic
- Error handling for API failures
- Token usage tracking and cost calculation

### Specialized Agent Classes

#### `FollowUpAgent` (`Agents/followup_agent.py`)
Generates targeted follow-up questions to clarify research ideas.

**Key Methods:**
- `generate_questions(user_idea: str) -> List[Dict]` - Generate 3 focused questions
- `enrich_idea_with_answers(original_idea, questions, answers) -> str` - Combine idea with Q&A
- `get_cost() -> float` - Calculate generation cost

#### `Layer1Agent` (`Agents/layer1_agent.py`)
Per-paper originality analysis comparing user's idea against a single paper.

**Key Methods:**
- `analyze_paper(user_idea, user_sentences, paper, paper_context) -> Layer1Result`
- `_build_analysis_prompt()` - Build comprehensive analysis prompt
- `_parse_result()` - Parse JSON response into Layer1Result
- `get_cost() -> float` - Calculate analysis cost

#### `Layer2Aggregator` (`Agents/layer2_agent.py`)
Aggregates Layer 1 results into final originality assessment using hard-coded logic.

**Key Methods:**
- `aggregate(layer1_results, user_sentences, cost_breakdown) -> Layer2Result`
- `_aggregate_criteria()` - Average criteria scores across papers
- `_compute_sentence_annotations()` - Compute sentence-level originality
- `_generate_summary()` - Generate natural language summary using LLM

#### `RealityCheckAgent` (`Agents/reality_check_agent.py`)
Checks if research ideas already exist using LLM's general knowledge.

**Key Methods:**
- `check_idea(user_idea: str) -> Dict` - Check for existing similar products/research
- `get_warning_message(result: Dict) -> str` - Generate warning message
- `adjust_originality_score(original_score, reality_check_result) -> int` - Adjust score based on findings

---

## Processing Pipeline

### `OriginalityPipeline` (`pipeline/originality_pipeline.py`)
Main orchestrator coordinating all components with real-time progress updates.

#### `PipelineState`
Holds the current state of the pipeline.

```python
@dataclass
class PipelineState:
    user_idea: str = ""
    enriched_idea: str = ""
    user_sentences: List[str] = None
    followup_questions: List[Dict] = None
    followup_answers: List[str] = None
    keywords: List[str] = None
    all_papers: List[Dict] = None
    selected_papers: List[Paper] = None
    layer1_results: List[Layer1Result] = None
    layer2_result: Layer2Result = None
    cost: CostBreakdown = None
    reality_check_result: Dict = None
    reality_check_warning: str = None
```

#### Pipeline Flow:
1. **Reality Check** - Check if idea already exists
2. **Follow-up Questions** - Generate and process clarifying questions
3. **Keyword Generation & Search** - Generate keywords and search ArXiv
4. **PDF Processing** - Extract content, chunk, and index in ChromaDB
5. **Layer 1 Analysis** - Analyze each paper individually
6. **Layer 2 Aggregation** - Combine results into final assessment

**Key Methods:**
- `run_full_analysis(user_idea, followup_answers) -> Layer2Result` - Complete pipeline
- `generate_followup_questions(user_idea) -> List[Dict]`
- `search_papers() -> List[Paper]`
- `process_papers() -> int` - Returns total chunks indexed
- `run_layer1_analysis() -> List[Layer1Result]`
- `run_layer2_analysis() -> Layer2Result`
- `get_matches_for_sentence(sentence, top_k) -> List[Dict]` - RAG query for UI
- `get_stats() -> Dict[str, Any]` - Pipeline statistics

### Processing Components

#### `ArxivClient` (`processing/arxiv_client.py`)
Client for interacting with the arXiv API.

**Key Methods:**
- `search(query, max_results, search_field, sort_by) -> List[Dict]`
- `search_multiple_keywords(keywords, results_per_keyword) -> List[Dict]`
- `papers_to_models(paper_dicts, limit) -> List[Paper]` - Convert to Paper objects
- `get_paper_by_id(arxiv_id) -> Optional[Paper]`

#### `PDFProcessor` (`processing/pdf_processor.py`)
Processes PDF papers into structured content using Docling.

**Key Methods:**
- `process_paper(paper: Paper) -> Paper` - Main processing method
- `_convert_to_markdown(source: str) -> Optional[str]` - PDF to Markdown conversion
- `_extract_headings_with_content(markdown, paper_id) -> List[Heading]`
- `_calculate_section_quality(section_text) -> float` - Quality scoring

#### `ChunkProcessor` (`processing/chunk_processor.py`)
Splits paper sections into chunks for embedding.

**Key Methods:**
- `process_paper(paper: Paper) -> Paper` - Chunk all headings in paper
- `chunk_section(heading: Heading) -> List[Chunk]` - Split section into chunks
- `_split_text_into_chunks(text, max_length) -> List[str]` - Text splitting logic

---

## RAG (Retrieval-Augmented Generation) System

### `ChromaStore` (`rag/chroma_store.py`)
ChromaDB-based vector store for paper chunks.

**Key Methods:**
- `add_paper(paper: Paper) -> int` - Add all chunks from paper, returns count
- `add_papers(papers: List[Paper]) -> int` - Batch add papers
- `search(query, n_results, filter_paper_id) -> List[Dict]` - Semantic search
- `search_by_sentence(sentence, n_results) -> List[Dict]` - Sentence-level search
- `get_chunk_by_id(chunk_id) -> Optional[Dict]` - Retrieve specific chunk
- `get_chunks_by_paper(paper_id) -> List[Dict]` - Get all chunks for paper
- `count() -> int` - Total chunks in store
- `clear()` - Clear all data

**Key Features:**
- Uses SentenceTransformer for embeddings (E5 model support)
- Cosine similarity search
- In-memory or persistent storage
- Automatic E5 prefix handling ("query:" for queries, "passage:" for documents)

### `Retriever` (`rag/retriever.py`)
High-level interface for finding relevant content.

**Key Methods:**
- `find_matches_for_sentence(sentence, top_k, similarity_threshold) -> List[MatchedSection]`
- `find_matches_for_idea(idea, top_k, similarity_threshold) -> List[MatchedSection]`
- `get_context_for_paper(paper_id, query) -> List[Dict]` - Get relevant chunks from specific paper
- `get_evidence_for_match(chunk_id, expand_context) -> Dict` - Detailed evidence with context
- `batch_search_sentences(sentences, top_k_per_sentence) -> Dict[int, List[MatchedSection]]`
- `compute_idea_paper_similarity(idea, paper_id) -> float` - Overall similarity score

---

## Web Interface

### `Streamlit App` (`app.py`)
Main web interface providing interactive originality assessment.

#### Session State Management:
```python
def init_session_state():
    defaults = {
        'step': 'input',  # input, questions, processing, results
        'pipeline': None,
        'user_idea': '',
        'followup_questions': [],
        'followup_answers': [],
        'result': None,
        'selected_sentence_idx': None,
        'progress_message': '',
        'progress_pct': 0,
    }
```

#### UI Flow:
1. **Input Step** - User enters research idea
2. **Questions Step** - Answer follow-up questions (optional)
3. **Processing Step** - Real-time pipeline execution with progress updates
4. **Results Step** - Interactive results with sentence highlighting and source exploration

**Key UI Components:**
- `render_gauge(score)` - Originality score visualization
- `render_sentence_with_highlighting(annotations)` - Color-coded sentence display
- `render_matches_panel(pipeline, sentence_idx, annotations)` - Source exploration
- `render_cost_breakdown(cost)` - API cost display

---

## Configuration System

### `config.py`
Central configuration for all system parameters.

#### Key Configuration Categories:

**API Settings:**
- `GOOGLE_API_KEY` - Gemini API key
- `INPUT_TOKEN_PRICE`, `OUTPUT_TOKEN_PRICE` - Cost calculation

**Agent Parameters:**
- Temperature, top_p, top_k settings for each agent type
- Model selection (gemini-2.5-flash)

**Processing Limits:**
- `MAX_PAPERS_TO_ANALYZE` - Maximum papers per analysis
- `PAPERS_PER_KEYWORD` - Results per search keyword
- `MIN_SECTION_LENGTH` - Minimum section length for processing

**Chunking Parameters:**
- `CHUNK_SIZE` - Target chunk size in characters
- `CHUNK_OVERLAP` - Overlap between chunks

**RAG Settings:**
- `EMBEDDING_MODEL` - SentenceTransformer model name
- `CHROMA_COLLECTION_NAME` - ChromaDB collection
- `RAG_TOP_K` - Default number of retrieval results

**Scoring Thresholds:**
- `HIGH_OVERLAP_THRESHOLD` - Threshold for red (low originality) classification
- `MEDIUM_OVERLAP_THRESHOLD` - Threshold for yellow classification
- `SCORE_RED_MAX`, `SCORE_YELLOW_MAX` - Score boundaries

---

## Key Design Patterns

### 1. **Pipeline Pattern**
The main `OriginalityPipeline` orchestrates all components in a sequential flow with state management and progress tracking.

### 2. **Strategy Pattern**
Different agent classes implement specific analysis strategies while inheriting from the base `Agent` class.

### 3. **Repository Pattern**
`ChromaStore` abstracts vector storage operations, while `Retriever` provides domain-specific query methods.

### 4. **Data Transfer Objects (DTOs)**
Extensive use of dataclasses for structured data transfer between components.

### 5. **Observer Pattern**
Progress callback mechanism allows real-time UI updates during pipeline execution.

### 6. **Factory Pattern**
`ArxivClient.papers_to_models()` converts raw API responses to domain objects.

---

## Data Flow Summary

1. **User Input** → `FollowUpAgent` → **Enriched Idea**
2. **Enriched Idea** → `KeywordAgent` → **Search Keywords**
3. **Keywords** → `ArxivClient` → **Paper Metadata**
4. **Paper Metadata** → `PDFProcessor` → **Structured Content**
5. **Structured Content** → `ChunkProcessor` → **Text Chunks**
6. **Text Chunks** → `ChromaStore` → **Vector Embeddings**
7. **Enriched Idea + Papers** → `Layer1Agent` → **Per-Paper Analysis**
8. **Layer1 Results** → `Layer2Aggregator` → **Final Assessment**
9. **Final Assessment** → **Streamlit UI** → **Interactive Results**

---

## Error Handling and Resilience

### API Resilience:
- Automatic retry logic with exponential backoff
- Rate limiting compliance (3-second delays for ArXiv)
- Graceful degradation when services fail

### Data Validation:
- Quality scoring for extracted content
- Chunk validation before indexing
- Fallback responses when LLM calls fail

### User Experience:
- Real-time progress updates
- Detailed error messages
- Cost transparency

This architecture provides a robust, scalable system for research originality assessment with clear separation of concerns and comprehensive error handling.
