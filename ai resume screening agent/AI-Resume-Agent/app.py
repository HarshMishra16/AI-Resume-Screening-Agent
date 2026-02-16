import sys
from pathlib import Path
import io
import logging

import streamlit as st
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from ranker import rank_resumes

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("ai-resume-agent")


def main():
    st.set_page_config(page_title="AI Resume Screening Agent", layout="wide")
    st.title("AI Resume Screening Agent")

    st.markdown("Upload multiple resumes (PDF/DOCX) and paste the Job Description.")

    uploaded_files = st.file_uploader("Upload resumes", type=["pdf", "docx"], accept_multiple_files=True)
    jd_text = st.text_area("Job Description", height=200)
    skills_file = st.file_uploader("(Optional) skills list (one per line)", type=["txt"])

    cache_dir = ROOT / "models"

    if st.button("Run Screening"):
        if not uploaded_files:
            st.warning("Please upload at least one resume.")
            return
        if not jd_text:
            st.warning("Please provide a Job Description.")
            return

        # Save uploaded resumes to a temp folder
        tmp_dir = ROOT / "data" / "resumes"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for uf in uploaded_files:
            dest = tmp_dir / uf.name
            with open(dest, "wb") as f:
                f.write(uf.getbuffer())
            paths.append(str(dest))

        skills_path = None
        if skills_file is not None:
            skills_path = str(ROOT / "data" / "custom_skills.txt")
            with open(skills_path, "wb") as f:
                f.write(skills_file.getbuffer())

        with st.spinner("Ranking candidates..."):
            df = rank_resumes(paths, jd_text, skills_path=skills_path, cache_dir=str(cache_dir))

        if df.empty:
            st.info("No candidates processed or no matches found.")
            return

        st.subheader("Ranking")
        st.dataframe(df[["name", "similarity", "skill_score", "experience_years", "experience_score", "final_score", "matched_skills"]])

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, "ranked_candidates.csv", "text/csv")

        st.subheader("Candidate Breakdown")
        for _, row in df.iterrows():
            with st.expander(f"{row['name']} — Final: {row['final_score']}"):
                st.write(f"Similarity: {row['similarity']}")
                st.write(f"Skill score: {row['skill_score']}")
                st.write(f"Experience (detected years): {row['experience_years']}")
                st.write(f"Experience score: {row['experience_score']}")
                st.write(f"Matched skills: {row['matched_skills']}")


if __name__ == "__main__":
    main()
