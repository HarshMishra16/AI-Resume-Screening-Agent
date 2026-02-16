from typing import List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def semantic_similarity(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """Compute cosine similarity between two 1D embedding vectors."""
    if emb_a is None or emb_b is None:
        return 0.0
    if emb_a.size == 0 or emb_b.size == 0:
        return 0.0
    # Ensure 2D for sklearn
    a = emb_a.reshape(1, -1)
    b = emb_b.reshape(1, -1)
    sim = cosine_similarity(a, b)[0, 0]
    return float(sim)


def skill_score(jd_skills: List[str], resume_skills: List[str]) -> float:
    """Skill matching score: intersection / |jd_skills|.

    If JD has no skills listed, return 0.0
    """
    if not jd_skills:
        return 0.0
    jd_set = set([s.lower() for s in jd_skills])
    res_set = set([s.lower() for s in resume_skills])
    inter = jd_set.intersection(res_set)
    return float(len(inter) / max(1, len(jd_set)))


def final_score(semantic_sim: float, skill_sc: float, exp_sc: float) -> float:
    """Combine scores using the prescribed weighted formula.

    final_score = 0.6*semantic + 0.3*skill + 0.1*experience
    Rounded to 3 decimals.
    """
    sc = 0.6 * semantic_sim + 0.3 * skill_sc + 0.1 * exp_sc
    return round(float(sc), 3)
