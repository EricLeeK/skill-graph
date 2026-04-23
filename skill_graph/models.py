"""Pydantic data models for SkillGraph."""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class Skill(BaseModel):
    """A single skill node."""
    id: str
    name: str
    description: str
    category_tags: List[str] = Field(default_factory=list)
    embedding: Optional[List[float]] = None
    related_skill_ids: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrieveRequest(BaseModel):
    """API request body for skill retrieval."""
    query: str
    top_k: int = 10
    use_keywords: bool = True
    use_sre: bool = True


class RetrieveResponse(BaseModel):
    """API response for skill retrieval."""
    query: str
    skills: List[Skill]
    refined_skills: List[Skill] = Field(default_factory=list)
    latency_ms: float = 0.0
