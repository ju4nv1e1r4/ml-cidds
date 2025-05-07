import streamlit as st
import pandas as pd

from src.ml.data_drift import DataDrift

TRAIN_PATH = "preprocessed/CIDDS-001.csv"
CURRENT_PATH = "src/data/new_data_sup.csv"
MIN_REGISTROS = 1000

dd = DataDrift(TRAIN_PATH, CURRENT_PATH)

st.markdown("## Data Monitoring & Observability 📊")

n_records = len(dd.df_current)
if n_records < MIN_REGISTROS:
    st.warning(
        f"Attention: there are only {n_records} current records. Interpret carefully!"
    )

options = dd.df_train.columns.tolist()
col = st.selectbox("Choose a column", options=options)

if col is not None and col in dd.df_train.columns:
    st.subheader(f"Distribution: {col}")
    fig = dd.generate_histogram(col)
    st.pyplot(fig)
else:
    st.warning("Choose a valid column.")

if pd.api.types.is_numeric_dtype(dd.df_train[col]):
    stat, p_value = dd.kolmogorov_smirnov(col)
    st.subheader("📏 Kolmogorov-Smirnov Test")
    st.write(f"Stat: `{stat:.4f}`")
    st.write(f"P-value: `{p_value:.4f}`")
    if p_value < 0.05:
        st.error("⚠️ Possible data drift detected (p < 0.05)")
    else:
        st.success("✅ No signs of drift")

st.subheader("📐 Jensen-Shannon Distance")
jsd = dd.jensen_shannon(col)
if jsd is not None:
    st.write(f"Jensen-Shannon Distance: `{jsd:.4f}`")
    if jsd > 0.1:
        st.warning("⚠️ Possible distribution shift detected")
    else:
        st.success("✅ Similar distributions")
else:
    st.info("Unable to calculate Jensen-Shannon for this feature.")
