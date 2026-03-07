# Hypothetica

**Research Originality Assessment Tool**

Hypothetica is an AI-powered system that evaluates the originality of research ideas by systematically analyzing them against existing academic literature. It helps researchers, students, and innovators assess whether their concepts are novel or if similar work already exists.

## Features

- **AI-Powered Analysis**: Uses advanced language models (Gemini 2.5 Flash) and retrieval-augmented generation (RAG) for comprehensive originality assessment
- **Interactive Workflow**: Step-by-step process with follow-up questions for idea clarification
- **Reality Check**: Preliminary screening using LLM's general knowledge to identify obvious existing solutions
- **Academic Paper Search**: Automated retrieval and analysis of relevant research papers from arXiv
- **Detailed Reports**: Layered analysis with overlap scoring, criteria-based evaluation, and evidence-backed results
- **Real-time Progress**: Live updates during analysis with detailed progress tracking
- **Cost Tracking**: Built-in token usage and cost monitoring for API calls

## Architecture

### Backend (Python/FastAPI)
- **Core Components**:
  - `AnalysisService`: Orchestrates the entire originality assessment pipeline
  - `PaperSearchService`: Handles academic paper discovery and retrieval
  - `PaperProcessingService`: Processes PDFs, extracts text, and creates vector embeddings
  - `OriginalityService`: Performs multi-layer originality analysis
  - `ChromaStore`: Vector database for efficient similarity search
- **AI Agents**:
  - `FollowUpAgent`: Generates targeted questions to clarify research ideas
  - `RealityCheckAgent`: Performs preliminary screening for existing solutions
  - `HeadingSelectorAgent`: Identifies relevant sections in academic papers

### Frontend (React/TypeScript)
- Modern single-page application built with React and TypeScript
- Real-time progress updates via Server-Sent Events (SSE)
- Responsive design with Tailwind CSS
- Component-based architecture for maintainability

### Infrastructure
- **Docker**: Containerized deployment with docker-compose
- **Nginx**: Reverse proxy for API routing and static file serving
- **ChromaDB**: Vector database for document chunks and embeddings
- **FastAPI**: High-performance async API framework

## Installation & Setup

### Prerequisites
- Docker and Docker Compose
- Google AI API key (for Gemini models)

### Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/HarunHudaiTan/Hypothetica.git
   cd Hypothetica
   ```

2. **Set up environment**:
   ```bash
   mkdir -p envfiles
   echo "GOOGLE_API_KEY=your_api_key_here" > envfiles/.env
   ```

3. **Start the application**:
   ```bash
   docker compose up --build
   ```

4. **Access the application**:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8005/api/health

### Development Setup

#### Backend Development
```bash
cd backend
pip install -r requirements.txt
python main.py
```

#### Frontend Development
```bash
cd frontend
npm install
npm run dev
```

## Usage

1. **Input Your Idea**: Describe your research concept or hypothesis in the text area
2. **Answer Clarifying Questions**: Respond to AI-generated follow-up questions to refine your idea
3. **Monitor Progress**: Watch real-time updates as the system searches papers and analyzes originality
4. **Review Results**: Examine detailed originality scores, matched papers, and specific overlaps

### API Usage

The backend provides a REST API for programmatic access:

```bash
# Start analysis
curl -X POST http://localhost:8005/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"user_idea": "Your research idea here"}'

# Check status
curl http://localhost:8005/api/analyze/{job_id}/status

# Get streaming updates
curl http://localhost:8005/api/analyze/{job_id}/stream
```

## Configuration

### Pipeline Parameters (in `backend/core/config.py`)
- `PAPERS_PER_QUERY_VARIANT`: Papers to fetch per search query (default: 150)
- `MAX_PAPERS_TO_ANALYZE`: Final papers for detailed analysis (default: 5)
- `HIGH_OVERLAP_THRESHOLD`: Threshold for low originality detection (default: 0.7)
- `EMBEDDING_MODEL`: Sentence transformer model (default: intfloat/e5-base-v2)

### Environment Variables
- `GOOGLE_API_KEY`: Required for Gemini API access

## How It Works

1. **Idea Submission**: User submits research idea
2. **Reality Check**: LLM evaluates if similar concepts already exist
3. **Clarification**: AI generates targeted follow-up questions
4. **Paper Search**: System searches arXiv for relevant academic papers
5. **Content Processing**: PDFs are downloaded, processed, and chunked into vector embeddings
6. **Layer 1 Analysis**: Sentence-level comparison between user idea and paper content
7. **Layer 2 Analysis**: Deeper semantic analysis with evidence extraction
8. **Results Compilation**: Generates comprehensive originality report

## Key Technologies

- **Language Models**: Google Gemini 2.5 Flash for reasoning and generation
- **Embeddings**: Sentence Transformers for semantic similarity
- **Vector Search**: ChromaDB for efficient document retrieval
- **PDF Processing**: Custom extraction and chunking pipeline
- **Frontend**: React with Vite for modern web development
- **Backend**: FastAPI with async processing and SSE support

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes and add tests
4. Run the application locally to verify
5. Submit a pull request

## License

[Add license information here]

## Contact

For questions or support, please open an issue on GitHub.
