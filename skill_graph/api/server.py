"""FastAPI server for SkillGraph."""
from fastapi import FastAPI
from pydantic import BaseModel

from skill_graph import SkillGraph, RetrieveRequest, RetrieveResponse

app = FastAPI(title="SkillGraph API", version="0.1.0")

# Global singleton (lazy init on first request to avoid import-time overhead)
_skill_graph: SkillGraph | None = None


def get_skill_graph() -> SkillGraph:
    global _skill_graph
    if _skill_graph is None:
        _skill_graph = SkillGraph()
    return _skill_graph


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(request: RetrieveRequest) -> RetrieveResponse:
    sg = get_skill_graph()
    return sg.retrieve_api(request)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
