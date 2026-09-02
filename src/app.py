import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from src.inference import HotelNERAndMapper

app = FastAPI(
    title="Hotel Operations NER & Semantic Mapping Engine",
    description="Enterprise NLP service for extracting operational tasks, items, and room locations from unstructured guest dispatches.",
    version="1.0.0"
)

# Global model engine instance
engine = None

@app.on_event("startup")
def load_engine():
    global engine
    engine = HotelNERAndMapper()

class DispatchRequest(BaseModel):
    text: str

class DispatchResponse(BaseModel):
    raw_input: str
    extracted_entities: Dict[str, List[str]]
    standardized_dispatch: List[Dict[str, Any]]

@app.get("/health")
def health():
    return {"status": "online", "service": "Hotel NER & Mapping Engine", "version": "1.0.0"}

@app.post("/dispatch/extract-and-map", response_model=DispatchResponse)
def extract_and_map(req: DispatchRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text input cannot be empty.")
    try:
        return engine.process_request(req.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
