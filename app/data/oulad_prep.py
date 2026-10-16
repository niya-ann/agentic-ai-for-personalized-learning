"""
oulad_prep.py — turns the raw OULAD CSVs into a clean, partitioned feature table
for the risk-prediction classifier (Section 7/17) and the Federated Learning demo
(Section 11).

Why OULAD fits the FL "simulated institutions" story so well: `code_module`
(course code: AAA, BBB, CCC, ...) is a real, natural partition of the data with
no overlapping students across modules used here -- so treating each module as
one "institution" is a legitimate partitioning, not an artificial one.

Label: at_risk = 1 if final_result in {Withdrawn, Fail} else 0
       (proxy for "this learner is likely to hit a learning gap")

Run:  python -m app.data.oulad_prep
Output: db_files/oulad_features.csv  (institution_id, features..., label)
"""
import os
import pandas as pd
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(HERE)), "data", "oulad")
OUT_PATH = os.path.join(os.path.dirname(os.path.dirname(HERE)), "db_files", "oulad_features.csv")

# The four largest modules become the four "simulated institutions"
INSTITUTIONS = ["BBB", "FFF", "DDD", "CCC"]

IMD_ORDER = {
    "0-10%": 0, "10-20": 1, "10-20%": 1, "20-30%": 2, "30-40%": 3, "40-50%": 4,
    "50-60%": 5, "60-70%": 6, "70-80%": 7, "80-90%": 8, "90-100%": 9,
}
AGE_ORDER = {"0-35": 0, "35-55": 1, "55<=": 2}
EDU_ORDER = {
    "No Formal quals": 0, "Lower Than A Level": 1, "A Level or Equivalent": 2,
    "HE Qualification": 3, "Post Graduate Qualification": 4,
}


def build_features():
    student_info = pd.read_csv(os.path.join(RAW_DIR, "studentInfo.csv"))
    student_vle = pd.read_csv(os.path.join(RAW_DIR, "studentVle.csv"))
    student_assess = pd.read_csv(os.path.join(RAW_DIR, "studentAssessment.csv"))
    assessments = pd.read_csv(os.path.join(RAW_DIR, "assessments.csv"))

    student_info = student_info[student_info["code_module"].isin(INSTITUTIONS)].copy()

    # engagement: total VLE clicks per (module, presentation, student)
    engagement = (
        student_vle.groupby(["code_module", "code_presentation", "id_student"])["sum_click"]
        .sum()
        .reset_index()
        .rename(columns={"sum_click": "total_clicks"})
    )

    # average assessment score per student (join studentAssessment -> assessments for module/presentation)
    student_assess = student_assess.copy()
    student_assess["score"] = pd.to_numeric(student_assess["score"], errors="coerce")
    sa = student_assess.merge(assessments, on="id_assessment", how="left")
    avg_score = (
        sa.groupby(["code_module", "code_presentation", "id_student"])["score"]
        .mean()
        .reset_index()
        .rename(columns={"score": "avg_assessment_score"})
    )

    df = student_info.merge(
        engagement, on=["code_module", "code_presentation", "id_student"], how="left"
    ).merge(
        avg_score, on=["code_module", "code_presentation", "id_student"], how="left"
    )

    df["total_clicks"] = df["total_clicks"].fillna(0)
    df["avg_assessment_score"] = df["avg_assessment_score"].fillna(df["avg_assessment_score"].median())

    df["imd_rank"] = df["imd_band"].map(IMD_ORDER).fillna(-1)
    df["age_rank"] = df["age_band"].map(AGE_ORDER).fillna(0)
    df["edu_rank"] = df["highest_education"].map(EDU_ORDER).fillna(0)
    df["gender_m"] = (df["gender"] == "M").astype(int)
    df["disability_flag"] = (df["disability"] == "Y").astype(int)

    df["label"] = df["final_result"].isin(["Withdrawn", "Fail"]).astype(int)

    feature_cols = [
        "num_of_prev_attempts", "studied_credits", "imd_rank", "age_rank",
        "edu_rank", "gender_m", "disability_flag", "total_clicks", "avg_assessment_score",
    ]

    out = df[["code_module"] + feature_cols + ["label"]].rename(columns={"code_module": "institution_id"})
    out = out.dropna()

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(out)} rows to {OUT_PATH}")
    print(out["institution_id"].value_counts())
    print("Overall at-risk rate:", out["label"].mean().round(3))
    return out


FEATURE_COLS = [
    "num_of_prev_attempts", "studied_credits", "imd_rank", "age_rank",
    "edu_rank", "gender_m", "disability_flag", "total_clicks", "avg_assessment_score",
]


def load_features():
    if not os.path.exists(OUT_PATH):
        return build_features()
    return pd.read_csv(OUT_PATH)


if __name__ == "__main__":
    build_features()
