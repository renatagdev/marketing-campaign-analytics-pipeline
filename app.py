
import streamlit as st
import pandas as pd
import sqlite3
import numpy as np
import os

# Path to local SQLite DB (in the same folder as app.py)
DB_PATH = "marketing_pipeline.db"


# Helper: safe division (avoid divide-by-zero)
def safe_div(numer, denom):
    return np.where(denom == 0, np.nan, numer / denom)


def init_db():
    """
    Make sure the database file and the raw table exist.
    This lets the app run even before first upload.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Create empty staging table if it doesn't exist yet
    cur.execute("""
    CREATE TABLE IF NOT EXISTS stg_campaigns_raw (
        id INTEGER,
        c_date TEXT,
        campaign_name TEXT,
        category TEXT,
        campaign_id TEXT,
        impressions INTEGER,
        mark_spent REAL,
        clicks INTEGER,
        leads INTEGER,
        orders INTEGER,
        revenue REAL
    );
    """)

    conn.commit()
    conn.close()


def save_uploaded_csv_to_db(uploaded_df: pd.DataFrame):
    """
    Replace stg_campaigns_raw with the uploaded CSV data.
    """
    conn = sqlite3.connect(DB_PATH)
    # overwrite the table each time -> always latest snapshot
    uploaded_df.to_sql("stg_campaigns_raw", conn, if_exists="replace", index=False)
    conn.close()


def run_pipeline():
    """
    1. Read stg_campaigns_raw from DB
    2. Clean data
    3. Feature engineering (KPIs)
    4. Write fact_campaigns_clean back to DB
    5. Return both DataFrames for display
    """
    conn = sqlite3.connect(DB_PATH)

    # 1. Read raw data
    df_raw = pd.read_sql("SELECT * FROM stg_campaigns_raw;", conn)

    # 2. CLEANING
    df_clean = df_raw.copy()

    # Drop full duplicates
    df_clean = df_clean.drop_duplicates()

    # Drop rows missing critical values
    required_cols = ["c_date", "campaign_name", "impressions", "clicks", "mark_spent", "revenue"]
    df_clean = df_clean.dropna(subset=required_cols)

    # Remove impossible values
    df_clean = df_clean[df_clean["impressions"] > 0]
    numeric_cols = ["impressions", "clicks", "leads", "orders", "mark_spent", "revenue"]
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean = df_clean[df_clean[col] >= 0]

    # Normalize / validate date
    df_clean["c_date"] = pd.to_datetime(df_clean["c_date"], errors="coerce")
    df_clean = df_clean.dropna(subset=["c_date"])
    df_clean = df_clean.sort_values("c_date")
    df_clean["c_date"] = df_clean["c_date"].dt.strftime("%Y-%m-%d")

    # Drop duplicate IDs (keep last = latest info)
    if "id" in df_clean.columns:
        df_clean = df_clean.drop_duplicates(subset=["id"], keep="last")

    # 3. FEATURE ENGINEERING (KPIs)
    df_feat = df_clean.copy()

    dt_tmp = pd.to_datetime(df_feat["c_date"], errors="coerce")

    # Core marketing KPIs
    df_feat["CTR_pct"] = safe_div(df_feat["clicks"], df_feat["impressions"]) * 100
    df_feat["CPC"] = safe_div(df_feat["mark_spent"], df_feat["clicks"])
    df_feat["CPA"] = safe_div(df_feat["mark_spent"], df_feat["orders"])
    df_feat["ConversionRate_pct"] = safe_div(df_feat["orders"], df_feat["clicks"]) * 100
    df_feat["ROAS"] = safe_div(df_feat["revenue"], df_feat["mark_spent"])
    df_feat["Profit"] = df_feat["revenue"] - df_feat["mark_spent"]
    df_feat["LeadRate_pct"] = safe_div(df_feat["leads"], df_feat["clicks"]) * 100

    # Time intelligence
    df_feat["Year"] = dt_tmp.dt.year
    df_feat["Month"] = dt_tmp.dt.month
    df_feat["Weekday"] = dt_tmp.dt.day_name()
    df_feat["Is_Weekend"] = dt_tmp.dt.weekday.isin([5, 6]).astype(int)

    # Round for nicer display
    round_cols = [
        "CTR_pct", "ConversionRate_pct", "LeadRate_pct",
        "CPC", "CPA", "ROAS", "Profit"
    ]
    for c in round_cols:
        if c in df_feat.columns:
            df_feat[c] = df_feat[c].round(2)

    # 4. Save cleaned / feature engineered data into fact_campaigns_clean
    df_feat.to_sql("fact_campaigns_clean", conn, if_exists="replace", index=False)

    conn.close()

    # 5. Return both tables for UI
    return df_raw, df_feat


# ---------------- STREAMLIT UI ----------------

# Make sure DB + table exist before we do anything else
init_db()

st.title("Marketing Campaign ETL Pipeline Demo")

st.write("1. Upload raw campaign CSV")
st.write("2. We store it in SQLite as stg_campaigns_raw (replacing old data)")
st.write("3. We run cleaning + KPI feature engineering")
st.write("4. We show RAW vs CLEAN tables below")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    # Read CSV the user uploaded
    new_df = pd.read_csv(uploaded_file)

    st.subheader("Preview of uploaded file")
    st.dataframe(new_df.head())

    # Save raw (replace mode)
    save_uploaded_csv_to_db(new_df)

    # Run pipeline to build the clean + KPI table
    try:
        df_raw_latest, df_clean_latest = run_pipeline()

        st.subheader("RAW data in DB (stg_campaigns_raw)")
        st.dataframe(df_raw_latest.head(20))

        st.subheader("CLEAN data with KPIs (fact_campaigns_clean)")
        st.dataframe(df_clean_latest.head(20))

    except Exception as e:
        st.error(f"Pipeline error: {e}")

else:
    st.info("Upload a CSV to start the pipeline.")
