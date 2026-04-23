"""Lightweight graph manager for Skill Knowledge Graph (SKG)."""
import json
from pathlib import Path
from typing import Dict, List, Optional

import networkx as nx
import numpy as np

from skill_graph.models import Skill


class SkillGraphManager:
    """In-memory NetworkX graph manager for skills and relationships."""

    def __init__(self) -> None:
        self.graph: nx.DiGraph = nx.DiGraph()
        self._skills: Dict[str, Skill] = {}

    @classmethod
    def from_jsonl(cls, path: str) -> "SkillGraphManager":
        """Load skills from JSONL and build graph."""
        manager = cls()
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Skills file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                raw = json.loads(line.strip())
                skill = Skill(**raw)
                manager.add_skill(skill)

        # Add edges after all nodes exist
        for skill in manager._skills.values():
            for related_id in skill.related_skill_ids:
                if related_id in manager._skills:
                    manager.graph.add_edge(skill.id, related_id, relation="related_to")
        return manager

    def add_skill(self, skill: Skill) -> None:
        """Add a skill node to the graph."""
        self._skills[skill.id] = skill
        self.graph.add_node(
            skill.id,
            name=skill.name,
            description=skill.description,
            category_tags=skill.category_tags,
            embedding=skill.embedding,
        )

    def get_skill(self, skill_id: str) -> Optional[Skill]:
        """Retrieve a skill by ID."""
        return self._skills.get(skill_id)

    def get_related(self, skill_id: str) -> List[Skill]:
        """Get skills related to the given skill ID."""
        if skill_id not in self.graph:
            return []
        return [self._skills[nid] for nid in self.graph.successors(skill_id) if nid in self._skills]

    def all_skills(self) -> List[Skill]:
        """Return all skills."""
        return list(self._skills.values())

    def embedding_matrix(self) -> np.ndarray:
        """Return (N, D) numpy array of skill embeddings in ID-sorted order."""
        skills = self.all_skills()
        if not skills or skills[0].embedding is None:
            return np.array([])
        return np.array([s.embedding for s in skills], dtype=np.float32)

    def id_list(self) -> List[str]:
        """Return list of skill IDs corresponding to embedding_matrix rows."""
        return [s.id for s in self.all_skills()]
