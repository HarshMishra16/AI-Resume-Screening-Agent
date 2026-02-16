from typing import Set, List, Optional
from pathlib import Path
import re
import logging

logger = logging.getLogger(__name__)

# A conservative default skills list. Projects should provide a more
# comprehensive database via a file in `data/skills.txt` if desired.
DEFAULT_SKILLS = [
    "python",
    "java",
    "c++",
    "c#",
    "javascript",
    "typescript",
    "react",
    "nodejs",
    "django",
    "flask",
    "sql",
    "postgresql",
    "mysql",
    "aws",
    "azure",
    "gcp",
    "docker",
    "kubernetes",
    "nlp",
    "pandas",
    "numpy",
    "tensorflow",
    "pytorch",
]


def load_skills(path: Optional[str] = None) -> Set[str]:
    """Load skills from a file (one per line) or fall back to defaults."""
    skills = set()
    if path:
        p = Path(path)
        if p.exists():
            try:
                for line in p.read_text(encoding="utf-8").splitlines():
                    s = line.strip().lower()
                    if s:
                        skills.add(s)
            except Exception:
                logger.exception("Failed to load skills file: %s", path)
    if not skills:
        skills = set(DEFAULT_SKILLS)
    return skills


def extract_skills(text: str, skills_db: Set[str]) -> List[str]:
    """Extract skills present in text by simple token/phrase matching.

    Args:
        text: Preprocessed text.
        skills_db: Set of skills (lowercased).

    Returns:
        List of matched skills (lowercased).
    """
    if not text:
        return []
    found = set()
    # Use word boundary matching for each skill to avoid substrings
    for skill in skills_db:
        pattern = rf"\b{re.escape(skill)}\b"
        if re.search(pattern, text, flags=re.IGNORECASE):
            found.add(skill)
    return sorted(found)
