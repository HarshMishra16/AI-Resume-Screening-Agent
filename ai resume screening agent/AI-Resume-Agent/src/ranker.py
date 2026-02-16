from typing import List, Dict, Any
from pathlib import Path
import logging
import pandas as pd

from .parser import extract_text
from .preprocess import preprocess_text, load_spacy_model
from .skill_extractor import load_skills, extract_skills
from .experience_extractor import extract_years_experience, normalize_experience
from .embedding import Embedder
from .scorer import semantic_similarity, skill_score, final_score

logger = logging.getLogger(__name__)


def rank_resumes(resume_paths: List[str], jd_text: str, skills_path: str = None,
                 model_name: str = "all-MiniLM-L6-v2", cache_dir: str = None) -> pd.DataFrame:
    """Process multiple resumes and return a ranked pandas DataFrame.

    Args:
        resume_paths: List of resume file paths.
        jd_text: Job description text.
        skills_path: Optional path to skills file.
        model_name: SentenceTransformer model name.
        cache_dir: Directory for caching embeddings.

    Returns:
        DataFrame sorted by final score descending.
    """
    nlp = load_spacy_model()
    skills_db = load_skills(skills_path)
    embedder = Embedder(model_name=model_name, cache_path=Path(cache_dir) if cache_dir else None)

    # Preprocess job description and extract skills
    jd_clean = preprocess_text(jd_text or "", nlp=nlp)
    jd_skills = extract_skills(jd_clean, skills_db)
    jd_emb = embedder.embed([jd_clean])[0] if jd_clean else None

    records: List[Dict[str, Any]] = []
    for path in resume_paths:
        try:
            raw = extract_text(path)
            clean = preprocess_text(raw or "", nlp=nlp)
            res_skills = extract_skills(clean, skills_db)
            years = extract_years_experience(raw or "")
            exp_norm = normalize_experience(years)

            res_emb = embedder.embed([clean])[0] if clean else None
            sem_sim = semantic_similarity(res_emb, jd_emb) if (res_emb is not None and jd_emb is not None) else 0.0
            sk_sc = skill_score(jd_skills, res_skills)
            final = final_score(sem_sim, sk_sc, exp_norm)

            records.append({
                "name": Path(path).name,
                "path": path,
                "similarity": round(sem_sim, 4),
                "skill_score": round(sk_sc, 4),
                "experience_years": years if years is not None else 0,
                "experience_score": round(exp_norm, 4),
                "final_score": final,
                "matched_skills": ", ".join(res_skills),
            })
        except Exception:
            logger.exception("Failed to process resume: %s", path)

    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df
    df = df.sort_values(by=["final_score", "similarity"], ascending=False).reset_index(drop=True)
    return df
