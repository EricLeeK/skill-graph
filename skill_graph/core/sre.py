"""Skill Refinement Engine (SRE)."""
from typing import List

import numpy as np

from skill_graph.models import Skill


class SkillRefinementEngine:
    """Lightweight rule-based skill refinement using embedding projection."""

    def __init__(self, alpha: float = 0.15):
        self.alpha = alpha

    def refine(self, query_embedding: np.ndarray, skills: List[Skill]) -> List[Skill]:
        """Generate query-specific optimized skill descriptions.

        Strategy: project skill embedding toward query embedding and use the
        projection strength to prepend a relevance hint to the description.
        """
        q = query_embedding.astype(np.float32)
        q_norm = q / (np.linalg.norm(q) + 1e-10)
        refined = []
        for skill in skills:
            if skill.embedding is None:
                refined.append(skill)
                continue
            s = np.array(skill.embedding, dtype=np.float32)
            s_norm = s / (np.linalg.norm(s) + 1e-10)
            sim = float(np.dot(q_norm, s_norm))
            hint = self._hint(sim)
            new_desc = f"[{hint}] {skill.description}"
            # Create a new Skill with refined description
            data = skill.model_dump()
            data["description"] = new_desc
            refined.append(Skill(**data))
        return refined

    def _hint(self, similarity: float) -> str:
        if similarity >= 0.85:
            return "HIGHLY RELEVANT"
        elif similarity >= 0.70:
            return "RELEVANT"
        elif similarity >= 0.55:
            return "SOMEWHAT RELEVANT"
        else:
            return "CONTEXTUALLY RELATED"
