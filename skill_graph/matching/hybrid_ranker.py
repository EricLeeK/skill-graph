"""Hybrid ranking: semantic similarity + keyword boost."""
from typing import List, Set

import numpy as np

from skill_graph.core.graph import SkillGraphManager
from skill_graph.models import Skill


class HybridRanker:
    """Rank candidate skills with a hybrid score combining cosine similarity
    and a boost for keyword-matched skills.
    """

    KEYWORD_BOOST = 0.5

    def __init__(self, manager: SkillGraphManager):
        self.manager = manager
        self._skill_ids = manager.id_list()
        self._id_to_idx = {sid: i for i, sid in enumerate(self._skill_ids)}

        embs = manager.embedding_matrix()
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        self._skill_embeddings_normed = embs / (norms + 1e-10)

    def rank(
        self,
        query_embedding: np.ndarray,
        candidate_ids: List[str],
        keyword_ids: Set[str],
        top_k: int,
    ) -> List[Skill]:
        """Rank candidates and return top-k skills.

        Args:
            query_embedding: Query embedding vector.
            candidate_ids: Candidate skill IDs to rank.
            keyword_ids: IDs matched by keyword; receive a boost.
            top_k: Number of results to return.

        Returns:
            List of top-k Skill objects.
        """
        q = query_embedding.astype(np.float32)
        q_norm = q / (np.linalg.norm(q) + 1e-10)

        indices = [self._id_to_idx[sid] for sid in candidate_ids if sid in self._id_to_idx]
        if not indices:
            indices = list(range(len(self._skill_ids)))

        cand_embs = self._skill_embeddings_normed[indices]
        sims = cand_embs @ q_norm

        for j, idx in enumerate(indices):
            sid = self._skill_ids[idx]
            if sid in keyword_ids:
                sims[j] += self.KEYWORD_BOOST

        if len(sims) <= top_k:
            top_idx = np.argsort(-sims)
        else:
            top_idx = np.argpartition(sims, -top_k)[-top_k:]
            top_idx = top_idx[np.argsort(-sims[top_idx])]

        selected = [self._skill_ids[indices[i]] for i in top_idx[:top_k]]
        return [self.manager.get_skill(sid) for sid in selected if self.manager.get_skill(sid) is not None]
