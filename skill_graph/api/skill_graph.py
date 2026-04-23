"""Universal SkillGraph API.

Core design: LLM-for-Index, Zero-Token Runtime.
- Preprocessing (one-time, token-acceptable):
  LLM generates diverse synthetic user queries per skill.
  All queries are embedded and stored in a multi-vector index.
- Runtime (per-query, zero-token):
  Query embedding is compared against skill embeddings AND
  synthetic query embeddings. No LLM calls at runtime.
"""
import pickle
import time
from pathlib import Path
from typing import List, Optional, Set

import numpy as np
from sentence_transformers import SentenceTransformer

from skill_graph.core.graph import SkillGraphManager
from skill_graph.core.sre import SkillRefinementEngine
from skill_graph.matching import HybridRanker, KeywordMatcher
from skill_graph.models import RetrieveRequest, RetrieveResponse, Skill


class SkillGraph:
    """Zero-token skill retrieval via multi-vector dense search + optional keyword boost."""

    def __init__(
        self,
        skills_path: str = "data_real/skills_ucsb_34k.jsonl",
        index_path: str = "data/index/multi_vector_index.pkl",
        embedder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_auto_keywords: bool = False,
    ):
        self.skills_path = skills_path
        self.index_path = index_path
        self.embedder = SentenceTransformer(embedder_name)

        # Load skill metadata
        self.manager = SkillGraphManager.from_jsonl(skills_path)
        self.skill_ids = self.manager.id_list()
        self._id_to_idx = {sid: i for i, sid in enumerate(self.skill_ids)}

        # Load multi-vector index
        if not Path(index_path).exists():
            raise FileNotFoundError(
                f"Multi-vector index not found at {index_path}; "
                "run scripts/build_multi_vector_index.py first."
            )
        with open(index_path, "rb") as f:
            index = pickle.load(f)

        self._skill_embeddings = index["skill_embeddings"]      # (N, D)
        self._syn_embeddings = index["synthetic_embeddings"]    # (N, 10, D)
        self._index_skill_ids = index["skill_ids"]

        # Build cross-reference: index position -> manager position
        self._index_to_mgr = {}
        for i, sid in enumerate(self._index_skill_ids):
            if sid in self._id_to_idx:
                self._index_to_mgr[i] = self._id_to_idx[sid]

        # Keyword matching (optional high-precision signal)
        self.keyword_matcher = KeywordMatcher(self.manager, use_auto=use_auto_keywords)
        self.ranker = HybridRanker(self.manager)
        self.sre = SkillRefinementEngine()

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        use_keywords: bool = True,
        use_sre: bool = True,
    ) -> RetrieveResponse:
        q_emb = self.embedder.encode(query, convert_to_numpy=True)
        return self.retrieve_from_embedding(
            q_emb,
            query=query,
            top_k=top_k,
            use_keywords=use_keywords,
            use_sre=use_sre,
        )

    def retrieve_from_embedding(
        self,
        query_embedding: np.ndarray,
        query: str = "",
        top_k: int = 10,
        use_keywords: bool = True,
        use_sre: bool = True,
    ) -> RetrieveResponse:
        t0 = time.perf_counter()
        q = query_embedding.astype(np.float32)
        q_norm = q / (np.linalg.norm(q) + 1e-10)

        # --- Multi-vector dense retrieval (ZERO TOKEN) ---
        # Skill embedding similarity: (N,)
        skill_sims = self._skill_embeddings @ q_norm

        # Best synthetic query similarity per skill: (N,)
        syn_sims = self._syn_embeddings @ q_norm          # (N, 10)
        best_syn_sims = np.max(syn_sims, axis=1)          # (N,)

        # Combined: max(skill_sim, best_syn_sim)
        combined_scores = np.maximum(skill_sims, best_syn_sims)

        # --- Keyword matching (optional boost) ---
        keyword_ids: Set[str] = set()
        if use_keywords and query:
            keyword_ids = self.keyword_matcher.match(query)
            for sid in keyword_ids:
                idx = self._id_to_idx.get(sid)
                if idx is not None and idx < len(combined_scores):
                    combined_scores[idx] += 0.5

        # --- Select top-k ---
        if len(combined_scores) <= top_k:
            top_indices = np.argsort(-combined_scores)
        else:
            top_indices = np.argpartition(combined_scores, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(-combined_scores[top_indices])]

        selected_ids = [self.skill_ids[i] for i in top_indices[:top_k]]
        skills = [self.manager.get_skill(sid) for sid in selected_ids
                  if self.manager.get_skill(sid) is not None]

        # --- Optional SRE ---
        refined: List[Skill] = []
        if use_sre:
            refined = self.sre.refine(q, skills)

        latency_ms = (time.perf_counter() - t0) * 1000
        return RetrieveResponse(
            query=query,
            skills=skills,
            refined_skills=refined if use_sre else [],
            latency_ms=latency_ms,
        )

    def retrieve_api(self, request: RetrieveRequest) -> RetrieveResponse:
        return self.retrieve(
            query=request.query,
            top_k=request.top_k,
            use_keywords=request.use_keywords,
            use_sre=request.use_sre,
        )
