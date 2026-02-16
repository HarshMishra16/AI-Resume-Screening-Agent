from typing import Optional
import re
import logging

logger = logging.getLogger(__name__)


def extract_years_experience(text: str) -> Optional[float]:
    """Extract years of experience from free text.

    Looks for patterns like '5 years', '3 yrs', '2+ years', '7-year', etc.

    Returns the maximum years found or None if nothing detected.
    """
    if not text:
        return None

    patterns = [
        r"(\d{1,2})\s*\+?\s*(?:years|year|yrs|yrs\.|yrs)",
        r"(\d{1,2})\s*-\s*year",
        r"(\d{1,2})\s*years?",
    ]
    found = []
    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            try:
                val = float(m.group(1))
                found.append(val)
            except Exception:
                continue

    if not found:
        return None
    years = max(found)
    # Cap to a reasonable maximum (we'll normalize later)
    return float(years)


def normalize_experience(years: Optional[float], cap: int = 10) -> float:
    """Normalize experience to [0,1] with a cap (default 10 years)."""
    if years is None:
        return 0.0
    try:
        y = float(years)
    except Exception:
        return 0.0
    if y <= 0:
        return 0.0
    y = min(y, float(cap))
    return round(y / float(cap), 4)
