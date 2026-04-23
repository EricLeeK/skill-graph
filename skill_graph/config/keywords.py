"""Keyword configuration loader."""
import json
import logging
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

DEFAULT_KEYWORDS_PATH = Path(__file__).parents[2] / "config" / "keywords.json"
AUTO_KEYWORDS_PATH = Path(__file__).parents[2] / "config" / "keywords_auto.json"


def load_keyword_map(path: str | Path | None = None) -> Dict[str, List[str]]:
    """Load keyword-to-skill-name mappings from JSON config.

    Args:
        path: Path to keywords.json. Defaults to project config/keywords.json.

    Returns:
        Dict mapping lowercase keyword -> list of exact skill names.
    """
    path = Path(path) if path else DEFAULT_KEYWORDS_PATH
    if not path.exists():
        logger.warning("Keywords config not found at %s, returning empty map.", path)
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("mappings", {})


def load_auto_keyword_map(path: str | Path | None = None) -> Dict[str, List[str]]:
    """Load auto-generated keyword map (zero test-set leakage).

    This map is generated purely from skill library analysis without using
    any SkillsBench ground truth information.

    Args:
        path: Path to keywords_auto.json. Defaults to project config/keywords_auto.json.

    Returns:
        Dict mapping lowercase keyword -> list of exact skill names.
    """
    path = Path(path) if path else AUTO_KEYWORDS_PATH
    if not path.exists():
        logger.warning("Auto keywords config not found at %s, returning empty map.", path)
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("mappings", {})
