import json
from datetime import datetime, timezone, timedelta
from fastapi import FastAPI, Query, Depends
from fastapi.responses import HTMLResponse
from core.cache import TemporalCache
from core.reranker import TimeDecayReranker
from sentence_transformers import SentenceTransformer

app = FastAPI()

# Dependency Singletons
cache = TemporalCache()
reranker = TimeDecayReranker(decay_rate=0.35)
model = None

@app.on_event("startup")
async def startup():
    global model
    await cache.connect()
    model = SentenceTransformer('all-MiniLM-L6-v2')

@app.on_event("shutdown")
async def shutdown():
    await cache.close()

def mock_vector_space_data():
    now = datetime.now(timezone.utc)
    return [
        {"text": "FLASH: Flash volatility alert in bank equities.", "semantic_score": 0.82, "source": "Live RSS Feed", "timestamp": (now - timedelta(minutes=2)).isoformat()},
        {"text": "REPORT: Appendix B margin note details risk limits.", "semantic_score": 0.95, "source": "PDF Margin Statement", "timestamp": (now - timedelta(hours=4)).isoformat()}
    ]

@app.get("/api/query")
async def execute_query(q: str = Query(...)):
    start_time = datetime.now()
    
    # 1. Pipeline Layer 1: Cache Check
    cached_hit = await cache.get(q)
    if cached_hit:
        return {
            "status": "CACHE_HIT",
            "latency_ms": round((datetime.now() - start_time).total_seconds() * 1000, 2),
            "results": json.loads(cached_hit)
        }
        
    # 2. Pipeline Layer 2: Vectorization & Ingestion Fusion
    _ = model.encode(q).tolist()
    raw_retrieved_data = mock_vector_space_data()
    
    # 3. Pipeline Layer 3: Reranking Math
    optimized_context = reranker.rerank(raw_retrieved_data)
    
    # 4. Pipeline Layer 4: Cache Update
    await cache.set(q, json.dumps(optimized_context))
    
    return {
        "status": "CACHE_MISS",
        "latency_ms": round((datetime.now() - start_time).total_seconds() * 1000, 2),
        "results": optimized_context
    }

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    # Dashboard HTML code goes here (Kept clean and separate from core engine logic)
    return "<h1>Dashboard Active</h1><p>Query API directly at /api/query?q=test</p>"