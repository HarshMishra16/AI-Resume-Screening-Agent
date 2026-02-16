# AI-Resume-Screening-Agent

AI-powered resume screening and candidate ranking system using NLP and Machine Learning. Automates skill extraction, job matching, and resume scoring.


An AI-powered resume screening and ranking system that automates candidate shortlisting using NLP, Machine Learning, and BERT embeddings.

This system extracts skills and experience from resumes, compares them with job descriptions, and generates a weighted ranking score.

---

##  Problem Statement

Manual resume screening is:
- Time-consuming
- Biased
- Inconsistent
- Not scalable

This project solves it using NLP and semantic similarity scoring.

---

##  Core Features

- Resume Parsing (PDF / DOCX)
- Skill Extraction using NLP
- Experience Extraction (Regex-based)
- Semantic Matching using BERT
- TF-IDF Keyword Matching
- Weighted Ranking Algorithm
- Streamlit Professional UI
- CSV Export of Results

---

##  Architecture

        Resume → Parsing → Cleaning → Skill Extraction  
                                ↓  
            Job Description → BERT Embedding  
                                ↓  
                    Cosine Similarity  
                                ↓  
                    Experience Weightage  
                                ↓  
                     Final Ranking Score  

---

##  Tech Stack

- Python
- Scikit-learn
- spaCy
- Sentence-Transformers (BERT)
- Pandas
- NumPy
- Streamlit
- Regex

---

##  Scoring Logic

Final Score =  
0.6 * Semantic Similarity (BERT)  
+ 0.3 * Skill Match Score  
+ 0.1 * Experience Score  

---

##  Run the Project

```bash
pip install -r requirements.txt
streamlit run app.py


---






