"""Keyword-based skill matching with regex whole-word matching."""
import re
from typing import Dict, List, Set

from skill_graph.config.keywords import load_keyword_map, load_auto_keyword_map
from skill_graph.core.graph import SkillGraphManager


class KeywordMatcher:
    """Match query keywords to skills via exact name mapping.

    Uses a static keyword->skill_name dictionary loaded from config.
    At init time, resolves skill names to actual skill IDs via the manager.
    Matching uses regex ``\\bkeyword\\b`` for whole-word detection.
    """

    def __init__(
        self,
        manager: SkillGraphManager,
        keyword_map: Dict[str, List[str]] | None = None,
        use_auto: bool = False,
    ):
        self.manager = manager

        # Build name -> list of IDs index (handles duplicate names)
        self._name_to_ids: Dict[str, List[str]] = {}
        for sid in manager.id_list():
            sk = manager.get_skill(sid)
            if sk:
                self._name_to_ids.setdefault(sk.name.lower(), []).append(sid)

        # Resolve keywords to skill IDs
        if keyword_map is not None:
            mappings = keyword_map
        elif use_auto:
            mappings = load_auto_keyword_map()
        else:
            mappings = load_keyword_map()

        self._keyword_to_ids: Dict[str, Set[str]] = {}
        for keyword, skill_names in mappings.items():
            ids: Set[str] = set()
            for sn in skill_names:
                ids.update(self._name_to_ids.get(sn.lower(), []))
            if ids:
                self._keyword_to_ids[keyword] = ids

    def match(self, query: str) -> Set[str]:
        """Return skill IDs whose keywords appear in *query*.

        Args:
            query: Raw user query string.

        Returns:
            Set of matched skill IDs.
        """
        query_lower = query.lower()
        matched: Set[str] = set()
        for keyword, skill_ids in self._keyword_to_ids.items():
            pattern = r"\b" + re.escape(keyword) + r"\b"
            if re.search(pattern, query_lower):
                matched.update(skill_ids)
        return matched
