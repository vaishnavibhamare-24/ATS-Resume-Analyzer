import re
import unicodedata
from collections import defaultdict
from datetime import datetime
import pandas as pd
import streamlit as st

st.set_page_config(page_title= " ⚡ATS Resume Keyword Analyzer", page_icon="🧲", layout="wide")

# -------------------------------
# Curated DS/DA/DE keyword map
# term -> {synonyms: [...], category: "..."}
# Edit/extend this anytime to fit your target roles.
# -------------------------------
KEYWORDS = {
    # Core programming
    "python": {"synonyms": ["python3"], "category": "Programming"},
    "r": {"synonyms": [], "category": "Programming"},
    "java": {"synonyms": [], "category": "Programming"},
    "sql": {"synonyms": ["postgres", "mysql", "sqlite", "t-sql", "mssql"], "category": "Programming"},
    "git": {"synonyms": ["github", "gitlab"], "category": "Tooling"},
    "linux": {"synonyms": ["bash", "shell"], "category": "Tooling"},

    # Libraries / ML
    "pandas": {"synonyms": [], "category": "Python Lib"},
    "numpy": {"synonyms": [], "category": "Python Lib"},
    "matplotlib": {"synonyms": [], "category": "Python Lib"},
    "seaborn": {"synonyms": [], "category": "Python Lib"},
    "scikit-learn": {"synonyms": ["sklearn"], "category": "ML"},
    "tensorflow": {"synonyms": ["tf", "keras"], "category": "ML"},
    "pytorch": {"synonyms": [], "category": "ML"},
    # "xgboost": {"synonyms": ["xgb"], "category": "ML"},
    # "lightgbm": {"synonyms": ["lgbm"], "category": "ML"},
    # "nlp": {"synonyms": ["natural language processing", "text mining"], "category": "ML"},
    # "time series": {"synonyms": ["arima", "sarima", "prophet"], "category": "ML"},
    # "deep learning": {"synonyms": ["cnn", "rnn", "lstm", "transformer"], "category": "ML"},

    # Analytics / Stats
    "statistics": {"synonyms": ["statistical analysis"], "category": "Analytics"},
    "hypothesis testing": {"synonyms": ["ab testing", "a/b testing", "t-test", "chi-square"], "category": "Analytics"},
    "regression": {"synonyms": ["linear regression", "logistic regression"], "category": "Analytics"},
    "anova": {"synonyms": ["two-way anova"], "category": "Analytics"},
    "feature engineering": {"synonyms": [], "category": "ML"},
    "model evaluation": {"synonyms": ["roc auc", "precision", "recall", "f1"], "category": "ML"},

    # BI / Viz
    "tableau": {"synonyms": [], "category": "BI/Viz"},
    "power bi": {"synonyms": ["powerbi"], "category": "BI/Viz"},
    "excel": {"synonyms": ["vlookup", "pivot table", "pivot tables"], "category": "BI/Viz"},
    "looker": {"synonyms": ["looker studio", "data studio"], "category": "BI/Viz"},

    # Data Eng / Cloud
    "etl": {"synonyms": ["elt", "pipeline"], "category": "Data Eng"},
    "airflow": {"synonyms": [], "category": "Data Eng"},
    "dbt": {"synonyms": [], "category": "Data Eng"},
    "spark": {"synonyms": ["pyspark"], "category": "Data Eng"},
    "aws": {"synonyms": ["s3", "ec2", "lambda", "redshift", "glue"], "category": "Cloud"},
    "azure": {"synonyms": ["databricks", "synapse"], "category": "Cloud"},
    "gcp": {"synonyms": ["bigquery", "cloud run"], "category": "Cloud"},
    "docker": {"synonyms": [], "category": "DevOps"},
    "kubernetes": {"synonyms": ["k8s"], "category": "DevOps"},
}

STOPWORDS = {
    "a","an","the","and","or","of","to","for","in","on","by","with","from","at","as","is","are","was","were",
    "this","that","these","those","it","its","be","being","been","you","your","we","our","they","their",
    "i","me","my","he","she","his","her","them","us","but"
}

# -------------------------------
# Helpers
# -------------------------------
def normalize(text: str) -> str:
    """Lowercase, strip accents, collapse spaces & punctuation spacing."""
    text = text.lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def word_boundary_regex(phrase: str) -> re.Pattern:
    """
    Build a regex that matches a whole-word phrase.
    e.g., 'power bi' -> r'\bpower\s+bi\b'
    """
    parts = [re.escape(p) for p in phrase.split()]
    pattern = r"\b" + r"\s+".join(parts) + r"\b"
    return re.compile(pattern, flags=re.IGNORECASE)

def find_hits(text: str, term: str, synonyms: list[str]) -> bool:
    """Return True if term or any synonym appears as a whole word/phrase."""
    rx = [word_boundary_regex(term)] + [word_boundary_regex(s) for s in synonyms]
    return any(r.search(text) for r in rx)

def collect_keywords_present(text: str, keyword_map: dict) -> dict:
    """
    Return {canonical_term: True} for all terms detected in text.
    """
    hits = {}
    for term, info in keyword_map.items():
        if find_hits(text, term, info["synonyms"]):
            hits[term] = True
    return hits

def compute_score(resume_terms: set, jd_terms: set) -> float:
    if not jd_terms:
        return 100.0
    return round(100.0 * len(resume_terms & jd_terms) / len(jd_terms), 1)

def csv_bytes(rows: list[dict]) -> bytes:
    if not rows:
        return b"keyword,category,suggestion\n"
    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode("utf-8")

def suggestion_for(term: str, category: str) -> str:
    # Simple, role-appropriate suggestion templates
    templates = {
        "Programming": f"Used {term} to automate data processing and build reusable analytics scripts.",
        "Python Lib": f"Leveraged {term} for efficient data wrangling and analysis.",
        "ML": f"Built and evaluated {term} models to improve predictive performance.",
        "Analytics": f"Applied {term} to derive insights and guide data-driven decisions.",
        "BI/Viz": f"Created dashboards in {term} to track KPIs and communicate trends.",
        "Data Eng": f"Developed {term}-based pipelines to ingest, transform, and validate data.",
        "Cloud": f"Deployed workloads using {term} services to improve scalability and cost-efficiency.",
        "DevOps": f"Containerized and orchestrated services using {term} for reliable deployments.",
        "Tooling": f"Version-controlled projects and collaborated effectively using {term}.",
    }
    return templates.get(category, f"Added practical experience with {term} in relevant projects.")

# -------------------------------
# UI
# -------------------------------
st.markdown(
    """
    <style>
    /* Keep background white */
    .stApp {
        background-color: #ffffff;
        color: #000000;
        font-family: 'Trebuchet MS', sans-serif;
    }

    /* Title styling */
    h1 {
        color: #2E86C1;  /* nice blue */
        text-align: center;
        font-size: 42px !important;
    }

    /* Subheaders */
    h2, h3 {
        color: #117A65;  /* dark green */
    }

    /* Info / alert box */
    .stAlert {
        border-radius: 10px;
        font-size: 18px;
    }

    /* Keyword chips */
    .chip {
        display: inline-block;
        padding: 6px 12px;
        margin: 4px;
        border-radius: 15px;
        background-color: #f5f7fa;  /* light gray */
        color: #2E4053; /* dark gray text */
        font-size: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.title("ATS Resume Keyword Analyzer")
st.caption("Paste your resume and a job description. Get a match score, missing keywords, and concrete suggestions to tailor your resume.")

left, right = st.columns(2, gap="large")
with left:
    resume_text = st.text_area(
        "📄 Your Resume (paste plain text)",
        height=300,
        placeholder="Paste the text from your resume here...",
    )
    custom_keys = st.text_input(
        "➕ Optional: Add custom keywords (comma-separated)",
        placeholder="e.g., snowflake, superset, graph databases",
    )
with right:
    jd_text = st.text_area(
        "🧾 Job Description",
        height=300,
        placeholder="Paste the target job description here...",
    )
    min_match = st.slider("Minimum match score to aim for (%)", 0, 100, 70, 5)

analyze = st.button("Analyze", use_container_width=True)

# -------------------------------
# Analysis
# -------------------------------
if analyze:
    if not resume_text.strip() or not jd_text.strip():
        st.warning("Please paste both your resume text and the job description.")
        st.stop()

    # Start with curated keywords; add any custom ones as a new category
    keyword_map = {k: v.copy() for k, v in KEYWORDS.items()}
    if custom_keys.strip():
        for raw in custom_keys.split(","):
            t = raw.strip().lower()
            if not t:
                continue
            if t not in keyword_map:
                keyword_map[t] = {"synonyms": [], "category": "Custom"}

    norm_resume = normalize(resume_text)
    norm_jd = normalize(jd_text)

    resume_hits = collect_keywords_present(norm_resume, keyword_map)
    jd_hits = collect_keywords_present(norm_jd, keyword_map)

    resume_terms = set(resume_hits.keys())
    jd_terms = set(jd_hits.keys())

    missing = sorted(jd_terms - resume_terms)
    matched = sorted(resume_terms & jd_terms)
    extra = sorted(resume_terms - jd_terms)

    score = compute_score(resume_terms, jd_terms)

    # --- Score block
    st.subheader("Match Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Match Score", f"{score}%")
    c2.metric("JD Keywords Found", f"{len(matched)}/{len(jd_terms)}")
    c3.metric("Missing Keywords", str(len(missing)))

    st.progress(min(int(score), 100))

    # --- Keyword chips
    def chips(items, label, empty_msg, color="#eef6ff"):
        st.markdown(f"**{label} ({len(items)})**")
        if not items:
            st.write(empty_msg)
            return
        html = " ".join(
            [f"<span style='background:{color};padding:6px 10px;border-radius:12px;margin:4px;display:inline-block'>{x}</span>"
             for x in items]
        )
        st.markdown(html, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        chips(matched, "Present in Resume & JD", "No overlaps detected.", "#e9fbe7")
        chips(extra, "Present in Resume but not in JD", "None.", "#f7f7f7")
    with col_b:
        chips(missing, "Missing (Important to Add)", "Great—nothing missing!", "#ffe9e9")

    # --- Suggestions table & downloads
    st.subheader("🛠 Suggestions to Add Missing Keywords")
    rows = []
    for term in missing:
        cat = keyword_map.get(term, {}).get("category", "Other")
        rows.append({"keyword": term, "category": cat, "suggestion": suggestion_for(term, cat)})

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "⬇️ Download suggestions (CSV)",
            data=csv_bytes(rows),
            file_name=f"suggestions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.info("Tip: Don’t stuff keywords. Integrate them into authentic bullet points with impact + metrics.")
    else:
        st.success("You're already covering the JD keywords from our dictionary—nice!")

#     # --- Guidance
#     st.subheader("💡 How to Use This Output")
#     st.markdown(
#         """
# - Turn suggestions into measurable bullets, e.g.,  
#   *“Built **XGBoost** model with **scikit-learn** to predict churn (AUC **0.91**), deployed on **AWS Lambda**, automated daily **ETL** with **Airflow**.”*  
# - Mirror language in the JD (e.g., “**A/B testing**” vs “AB testing”).  
# - Keep it honest—only include skills you truly have; be ready to discuss them.
# """
    # )

else:
    st.info("Paste your resume and a JD, then click **Analyze**.")
