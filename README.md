# Market Sentiment Analyzer 

**Market Sentinel** is a production-grade financial sentiment analysis system featuring a real-time dashboard, temporal-aware caching, cross-source context fusion, and exponential decay analytics.

## Features

- **Real-Time Sentiment Analysis**: Analyzes market alerts (e.g., from RSS feeds) and classifies them as BULLISH, BEARISH, or NEUTRAL using advanced LLMs.
- **Cross-Source Context Fusion**: Ingests PDF documents (like market reports or earnings transcripts), chunks them, and stores embeddings in a vector database to provide rich context for sentiment analysis.
- **Semantic Caching**: Utilizes Redis to cache similar semantic queries, drastically reducing LLM API calls and latency.
- **Exponential Decay Analytics**: Applies temporal decay to sentiment scores based on configurable half-lives (e.g., real-time news decays faster than structural PDF reports).
- **Interactive Dashboard**: A sleek, real-time UI showing dominant market sentiment, distribution charts, cache hit rates, average exposure/confidence, and a live activity feed.

## Technology Stack

- **Backend**: Python 3.10+, FastAPI, Uvicorn, Pydantic
- **AI/LLM**: Google Gemini (default: `gemini-2.5-flash`), extensible to OpenAI
- **Vector Database**: Qdrant (supports in-memory mode for local development)
- **Cache**: Redis
- **Frontend**: HTML5, CSS3 (Glassmorphism design), Vanilla JavaScript

## Project Structure

```text
market-sentiment-analyzer/
├── api/
│   └── main.py              # FastAPI application and endpoints
├── dashboard/
│   ├── index.html           # Real-time dashboard UI
│   └── static/              # CSS and JS assets
├── src/
│   ├── cache/               # Redis semantic cache implementation
│   ├── database/            # Data models (Pydantic)
│   ├── ingestion/           # RSS and PDF ingestion workers
│   ├── orchestrator/        # Pipeline orchestration (cache → fusion → LLM → decay)
│   ├── pipeline/            # Core processing logic
│   ├── services/            # External service integrations
│   ├── storage/             # Vector storage (Qdrant)
│   └── config.py            # Centralized configuration (environment variables)
├── .env                     # Environment variables (create this)
├── requirements.txt         # Python dependencies
└── run.py                   # Application entry point
```

## Getting Started

### Prerequisites

- Python 3.10 or higher
- [Redis](https://redis.io/) (Optional, but recommended for caching)
- [Qdrant](https://qdrant.tech/) (Defaults to in-memory if not running a dedicated instance)

### Installation

1. Navigate to the project directory:
   ```bash
   cd market-sentiment-analyzer
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have a `.env` file in the root directory configured with your API keys and service endpoints:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   # OPENAI_API_KEY=your_openai_api_key_here # If using OpenAI

   # Optional overrides (defaults to localhost/in-memory)
   # REDIS_HOST=localhost
   # REDIS_PORT=6379
   # QDRANT_IN_MEMORY=true
   ```

### Running the Application

Start the application using the provided run script:

```bash
python run.py
```

The application will start with Uvicorn, usually accessible at `http://localhost:8000`.

### Accessing the Dashboard

Once the server is running, simply navigate to:
**http://localhost:8000/**

## API Endpoints

- `GET /` — Serves the real-time dashboard.
- `GET /health` — System health check (components status).
- `POST /analyze` — Accept an alert JSON payload, run the full pipeline, and return a classification.
- `POST /ingest/rss` — Trigger a manual RSS polling cycle.
- `POST /ingest/pdf` — Upload a PDF for processing and vector storage context.
- `GET /api/dashboard/stats` — Aggregate pipeline statistics.
- `GET /api/dashboard/recent` — Recent analysis results.
