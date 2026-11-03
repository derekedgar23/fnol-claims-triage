"""
FNOL Claims Triage Dashboard
=============================
Streamlit app that scores synthetic claims with the trained model and shows
triage tier distributions, feature importance, and per-claim drill-down.

Run locally (no Databricks required):
    pip install streamlit plotly scikit-learn shap
    streamlit run app/triage_dashboard.py

All data is synthetic and generated in code.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add src/ to path so we can import claims_triage
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from claims_triage.data import LARGE_LOSS_THRESHOLD, generate
from claims_triage.evaluation import classification_metrics
from claims_triage.features import ALL_FEATURES, CATEGORICAL, NUMERIC
from claims_triage.train import feature_frame, gbm_classifier
from claims_triage.triage import TIER_THRESHOLDS, assign_tier, reason_codes

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="FNOL Claims Triage", page_icon="📊", layout="wide")
st.title("FNOL Claims Triage Dashboard")
st.caption("Synthetic data -- all programs, MGAs, insureds and amounts are fictional.")

# ---------------------------------------------------------------------------
# Generate data and train model (cached so it only runs once)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_data_and_model():
    policies, claims = generate()
    merged = claims.merge(policies, on="policy_id", how="left")

    X = feature_frame(merged)
    y = merged["large_loss"]

    # Quick train/test split by index (deterministic)
    train_mask = merged.index % 5 != 0  # 80/20
    X_train, X_test = X[train_mask], X[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]

    model = gbm_classifier()
    model.fit(X_train, y_train)

    proba = model.predict_proba(X)[:, 1]
    merged["predicted_proba"] = proba
    merged["triage_tier"] = [assign_tier(p) for p in proba]
    merged["reason_codes"] = merged.apply(reason_codes, axis=1)

    test_proba = model.predict_proba(X_test)[:, 1]
    test_metrics = classification_metrics(y_test, test_proba)

    return merged, model, test_metrics


data, model, test_metrics = load_data_and_model()

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
st.sidebar.header("Filters")

lines = ["All"] + sorted(data["line_of_business"].dropna().unique().tolist())
selected_line = st.sidebar.selectbox("Line of Business", lines)

tiers = ["All", "HIGH", "ELEVATED", "STANDARD"]
selected_tier = st.sidebar.selectbox("Triage Tier", tiers)

filtered = data.copy()
if selected_line != "All":
    filtered = filtered[filtered["line_of_business"] == selected_line]
if selected_tier != "All":
    filtered = filtered[filtered["triage_tier"] == selected_tier]

# ---------------------------------------------------------------------------
# KPIs row
# ---------------------------------------------------------------------------
st.markdown("---")
col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Total Claims", f"{len(filtered):,}")
col2.metric("Large Losses", f"{filtered['large_loss'].sum():,.0f}")
col3.metric("Large Loss Rate", f"{filtered['large_loss'].mean():.1%}")
col4.metric("Test PR AUC", f"{test_metrics['pr_auc']:.3f}")
col5.metric("Test ROC AUC", f"{test_metrics['roc_auc']:.3f}")

# ---------------------------------------------------------------------------
# Row 1: Tier distribution + Tier performance
# ---------------------------------------------------------------------------
st.markdown("---")
r1c1, r1c2 = st.columns(2)

with r1c1:
    st.subheader("Triage Tier Distribution")
    tier_counts = filtered["triage_tier"].value_counts().reindex(["HIGH", "ELEVATED", "STANDARD"], fill_value=0)
    fig_tier = px.bar(
        x=tier_counts.index, y=tier_counts.values,
        color=tier_counts.index,
        color_discrete_map={"HIGH": "#F44336", "ELEVATED": "#FF9800", "STANDARD": "#4CAF50"},
        labels={"x": "Tier", "y": "Claims"},
    )
    fig_tier.update_layout(showlegend=False, height=350)
    st.plotly_chart(fig_tier, use_container_width=True)

with r1c2:
    st.subheader("Tier Performance")
    tier_stats = (filtered.groupby("triage_tier")
                  .agg(claims=("large_loss", "size"),
                       large_losses=("large_loss", "sum"),
                       mean_proba=("predicted_proba", "mean"))
                  .reindex(["HIGH", "ELEVATED", "STANDARD"]))
    tier_stats["share_of_claims"] = (tier_stats["claims"] / tier_stats["claims"].sum()).map("{:.1%}".format)
    tier_stats["share_of_large_losses"] = (tier_stats["large_losses"] / tier_stats["large_losses"].sum()).map("{:.1%}".format)
    tier_stats["large_loss_rate"] = (tier_stats["large_losses"] / tier_stats["claims"]).map("{:.1%}".format)
    tier_stats["mean_proba"] = tier_stats["mean_proba"].map("{:.3f}".format)
    st.dataframe(tier_stats[["claims", "large_losses", "share_of_claims",
                              "share_of_large_losses", "large_loss_rate", "mean_proba"]],
                 use_container_width=True)

# ---------------------------------------------------------------------------
# Row 2: Score distribution + by line of business
# ---------------------------------------------------------------------------
st.markdown("---")
r2c1, r2c2 = st.columns(2)

with r2c1:
    st.subheader("Score Distribution")
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=filtered[filtered["large_loss"] == 0]["predicted_proba"],
                                     name="Not Large Loss", opacity=0.6, nbinsx=50,
                                     marker_color="#4CAF50"))
    fig_hist.add_trace(go.Histogram(x=filtered[filtered["large_loss"] == 1]["predicted_proba"],
                                     name="Large Loss", opacity=0.6, nbinsx=50,
                                     marker_color="#F44336"))
    fig_hist.add_vline(x=TIER_THRESHOLDS["HIGH"], line_dash="dash", line_color="red",
                        annotation_text="HIGH threshold")
    fig_hist.add_vline(x=TIER_THRESHOLDS["ELEVATED"], line_dash="dash", line_color="orange",
                        annotation_text="ELEVATED threshold")
    fig_hist.update_layout(barmode="overlay", height=350,
                            xaxis_title="P(Large Loss)", yaxis_title="Count")
    st.plotly_chart(fig_hist, use_container_width=True)

with r2c2:
    st.subheader("Large Loss Rate by Line of Business")
    lob_stats = (filtered.groupby("line_of_business")
                 .agg(claims=("large_loss", "size"), large_losses=("large_loss", "sum")))
    lob_stats["rate"] = lob_stats["large_losses"] / lob_stats["claims"]
    lob_stats = lob_stats.sort_values("rate", ascending=True)
    fig_lob = px.barh(lob_stats, x="rate", y=lob_stats.index,
                       color="rate", color_continuous_scale="RdYlGn_r",
                       labels={"rate": "Large Loss Rate", "y": ""})
    fig_lob.update_layout(height=350, coloraxis_showscale=False)
    st.plotly_chart(fig_lob, use_container_width=True)

# ---------------------------------------------------------------------------
# Row 3: Feature distributions for HIGH vs STANDARD
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Feature Comparison: HIGH vs STANDARD Tier")

compare_feat = st.selectbox("Select feature", NUMERIC, index=NUMERIC.index("report_lag_days"))

r3c1, r3c2 = st.columns(2)
with r3c1:
    high_vals = filtered[filtered["triage_tier"] == "HIGH"][compare_feat].dropna()
    std_vals = filtered[filtered["triage_tier"] == "STANDARD"][compare_feat].dropna()

    fig_comp = go.Figure()
    fig_comp.add_trace(go.Histogram(x=std_vals, name="STANDARD", opacity=0.5,
                                     marker_color="#4CAF50", nbinsx=40))
    fig_comp.add_trace(go.Histogram(x=high_vals, name="HIGH", opacity=0.5,
                                     marker_color="#F44336", nbinsx=40))
    fig_comp.update_layout(barmode="overlay", height=300,
                            xaxis_title=compare_feat, yaxis_title="Count")
    st.plotly_chart(fig_comp, use_container_width=True)

with r3c2:
    summary_stats = pd.DataFrame({
        "HIGH": high_vals.describe(),
        "STANDARD": std_vals.describe(),
    }).round(2)
    st.dataframe(summary_stats, use_container_width=True)

# ---------------------------------------------------------------------------
# Row 4: Monthly trends
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Monthly Trends")

monthly = (filtered.assign(loss_date=pd.to_datetime(filtered["loss_date"]))
           .assign(month=lambda df: df["loss_date"].dt.to_period("M").astype(str))
           .groupby("month")
           .agg(claims=("large_loss", "size"),
                large_losses=("large_loss", "sum"),
                mean_score=("predicted_proba", "mean"))
           .reset_index())
monthly["large_loss_rate"] = monthly["large_losses"] / monthly["claims"]

r4c1, r4c2 = st.columns(2)
with r4c1:
    fig_monthly = px.line(monthly, x="month", y="claims", markers=True,
                           labels={"month": "Month", "claims": "Claims"})
    fig_monthly.update_layout(height=300)
    st.plotly_chart(fig_monthly, use_container_width=True)

with r4c2:
    fig_rate = px.line(monthly, x="month", y="large_loss_rate", markers=True,
                        labels={"month": "Month", "large_loss_rate": "Large Loss Rate"},
                        color_discrete_sequence=["#F44336"])
    fig_rate.update_layout(height=300)
    st.plotly_chart(fig_rate, use_container_width=True)

# ---------------------------------------------------------------------------
# Row 5: Claim-level drill-down
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Claim-Level Drill-Down")
st.caption("Showing the top 100 claims by predicted probability in the current filter.")

display_cols = ["claim_id", "policy_id", "triage_tier", "predicted_proba", "reason_codes",
                "large_loss", "incurred_12m", "line_of_business", "cause_of_loss",
                "injury_flag", "attorney_at_fnol", "report_lag_days", "insured_state"]

top_claims = (filtered[display_cols]
              .sort_values("predicted_proba", ascending=False)
              .head(100)
              .reset_index(drop=True))

st.dataframe(
    top_claims.style
    .background_gradient(subset=["predicted_proba"], cmap="RdYlGn_r")
    .format({"predicted_proba": "{:.3f}", "incurred_12m": "${:,.0f}"}),
    use_container_width=True,
    height=400,
)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    f"**Model**: GBM classifier | **Threshold**: HIGH >= {TIER_THRESHOLDS['HIGH']}, "
    f"ELEVATED >= {TIER_THRESHOLDS['ELEVATED']} | "
    f"**Large loss**: incurred >= ${LARGE_LOSS_THRESHOLD:,} | "
    f"**Data**: {len(data):,} synthetic claims"
)
