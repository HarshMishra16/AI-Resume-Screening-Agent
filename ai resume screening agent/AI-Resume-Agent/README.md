# AI Resume Screening Agent

Production-style, modular AI Resume Screening Agent using NLP and transformer embeddings.

## Overview

This project accepts multiple resumes (PDF/DOCX) and a Job Description (JD), extracts skills and years of experience, computes semantic similarity using SentenceTransformers (`all-MiniLM-L6-v2`), and ranks candidates using a weighted score.

## Project Structure

AI-Resume-Agent/
│
├── data/
│   ├── resumes/            # Uploaded resumes saved by app
│   └── job_description.txt
│
├── src/
│   ├── parser.py
│   ├── preprocess.py
│   ├── skill_extractor.py
│   ├── experience_extractor.py
│   ├── embedding.py
│   ├── scorer.py
│   └── ranker.py
│
├── models/                 # caches (embeddings) and model artifacts
├── app.py                  # Streamlit UI
├── requirements.txt
├── README.md
└── LICENSE

## Scoring Logic

- Semantic similarity (SentenceTransformer embeddings): weight 0.6
- Skill matching score = |intersection(jd_skills, resume_skills)| / |jd_skills| (weight 0.3)
- Experience score: normalized to [0,1] with cap at 10 years (weight 0.1)

Final score = 0.6 * semantic_similarity + 0.3 * skill_score + 0.1 * experience_score
Rounded to 3 decimals.

## How to run

1. Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

2. Run the Streamlit app:

```bash
streamlit run app.py
```

3. Upload resumes and paste the JD in the web UI. Download the ranked CSV.

## Files of interest

- `src/parser.py` — extracts text from PDF/DOCX
- `src/preprocess.py` — normalization and lemmatization using spaCy
- `src/skill_extractor.py` — skill matching using a predefined skills list
- `src/experience_extractor.py` — regex-based experience extraction and normalization
- `src/embedding.py` — wrapper around SentenceTransformer with optional caching
- `src/scorer.py` — computes similarity, skill score, and final score
- `src/ranker.py` — main orchestration to process multiple resumes
- `app.py` — Streamlit UI

## Production notes & scaling

- Use a persistent cache (S3 / Redis) for embeddings.
- Run the embedding service separately (microservice) to scale horizontally.
- Add authentication and persistence for audit logs.
- Use async workers (Celery / RQ) for large batches of resumes.

## Interview Prep (Concise)

- Why BERT over TF-IDF?
  - BERT embeddings capture contextual, semantic meaning; TF-IDF is lexical and sparse.

- How to reduce bias?
  - Audit training data, remove proxies for protected attributes, apply fairness-aware reweighting, and human-in-the-loop review.

- How to scale to enterprise?
  - Separate embedding service, use GPU-backed inference clusters, shard resume processing, and employ persistent caches for embeddings.

- How to fine-tune BERT?
  - Collect labeled pairs (resume, JD) with relevance labels, fine-tune with a contrastive or classification objective on top of SBERT.

- How to improve ATS integration?
  - Export standardized JSON, map resume fields to ATS fields, and provide webhooks or SFTP-based transfer.

## License

MIT
