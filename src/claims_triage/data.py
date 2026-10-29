"""Synthetic commercial P&C data for the FNOL claims triage project.

Everything here is fictional: programs, MGAs, insureds and losses are generated from
random distributions with a fixed seed, so results are reproducible and safe to publish.

Two tables are produced:

* ``policies``: one row per policy (the grain of the policy feature table).
* ``claims``: one row per claim, with fields known at First Notice of Loss (FNOL) and the
  12-month incurred amount used as the training target.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LARGE_LOSS_THRESHOLD = 50_000

# code, program name, MGA, line of business, states, base log-severity
PROGRAMS = [
    ("CYB01", "Cyber Shield SMB", "Northgate Underwriters", "Cyber", ["CA", "TX", "NY", "IL"], 8.6),
    ("HOM02", "Coastal Homeowners", "Bayline Specialty", "Homeowners", ["FL", "TX", "LA"], 8.5),
    ("CAU03", "Regional Fleet Auto", "Ridgeway Transport Risk", "Commercial Auto", ["TX", "OK", "KS"], 8.9),
    ("IMR04", "Contractors Equipment", "Summit Marine Partners", "Inland Marine", ["CO", "UT", "AZ"], 8.3),
    ("GLC05", "Artisan Contractors GL", "Keystone Risk Partners", "General Liability", ["PA", "OH", "NJ"], 8.8),
    ("WCP06", "Mountain Workers Comp", "Granite Peak MGA", "Workers Compensation", ["WY", "CO", "MT"], 8.7),
]

CAUSES = {
    "Cyber": ["Ransomware", "Business Email Compromise", "Data Breach"],
    "Homeowners": ["Wind", "Water Damage", "Fire", "Theft"],
    "Commercial Auto": ["Collision", "Cargo", "Liability BI"],
    "Inland Marine": ["Theft", "Equipment Damage"],
    "General Liability": ["Bodily Injury", "Property Damage", "Products"],
    "Workers Compensation": ["Strain", "Fall", "Laceration", "Struck By"],
}
# extra log-severity by cause (unlisted causes get 0)
CAUSE_EFFECT = {"Ransomware": 0.6, "Fire": 0.7, "Liability BI": 0.6, "Bodily Injury": 0.5,
                "Fall": 0.3, "Products": 0.4, "Theft": -0.3, "Water Damage": -0.1, "Laceration": -0.4}
INJURY_LINES = {"Commercial Auto", "General Liability", "Workers Compensation"}


def generate_policies(n: int = 20_000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(PROGRAMS), n)
    prog = [PROGRAMS[i] for i in idx]
    revenue = np.exp(rng.normal(13.6, 1.1, n)).round(-3)          # right-skewed, median ~ $800k
    years = np.round(rng.gamma(2.2, 5.0, n)).astype(float)
    tiv = (revenue * np.exp(rng.normal(0.3, 0.9, n))).round(-3)     # total insured value
    df = pd.DataFrame({
        "policy_id": [f"POL-{i:06d}" for i in range(1, n + 1)],
        "program_code": [p[0] for p in prog],
        "program_name": [p[1] for p in prog],
        "mga_name": [p[2] for p in prog],
        "line_of_business": [p[3] for p in prog],
        "insured_state": [rng.choice(p[4]) for p in prog],
        "policy_limit": rng.choice([250_000, 500_000, 1_000_000, 2_000_000, 5_000_000], n,
                                   p=[0.15, 0.25, 0.35, 0.15, 0.10]),
        "deductible": rng.choice([0, 1_000, 2_500, 5_000, 10_000], n, p=[0.1, 0.35, 0.3, 0.15, 0.1]),
        "annual_revenue": revenue,
        "years_in_business": years,
        "total_insured_value": tiv,
        "prior_claims_3y": rng.poisson(0.45, n),
    })
    # Missing values, as in real bordereaux: revenue and years are often not reported
    df.loc[rng.random(n) < 0.08, "annual_revenue"] = np.nan
    df.loc[rng.random(n) < 0.12, "years_in_business"] = np.nan
    return df


def generate_claims(policies: pd.DataFrame, n: int = 40_000, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pol = policies.sample(n=n, replace=True, random_state=seed).reset_index(drop=True)
    lob = pol["line_of_business"].to_numpy()
    base = pol["program_code"].map({p[0]: p[5] for p in PROGRAMS}).to_numpy()

    cause = np.array([rng.choice(CAUSES[x]) for x in lob])
    is_injury_line = np.isin(lob, list(INJURY_LINES))
    injury = (rng.random(n) < np.where(is_injury_line, 0.35, 0.03)).astype(int)
    attorney = (rng.random(n) < 0.04 + 0.22 * injury).astype(int)
    report_lag = np.round(rng.exponential(6, n)).astype(int)
    late = rng.random(n) < 0.02                                    # a few very late reports (outliers)
    report_lag[late] += rng.integers(120, 720, late.sum())
    loss_date = pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 1277, n), unit="D")

    rev = pol["annual_revenue"].fillna(pol["annual_revenue"].median()).to_numpy()
    state = pol["insured_state"].to_numpy()
    years = pol["years_in_business"].fillna(10).to_numpy()
    tiv_ratio = pol["total_insured_value"].to_numpy() / pol["policy_limit"].to_numpy()
    lob_liab = np.isin(lob, ["General Liability", "Commercial Auto"])
    prior = pol["prior_claims_3y"].to_numpy()
    log_sev = (
        base - 0.30
        + 0.30 * np.log(rev / 800_000)
        + 0.55 * injury
        + 0.60 * attorney
        + 0.90 * injury * attorney                                  # interactions: trees should find these
        + 0.80 * attorney * lob_liab
        + 1.30 * ((cause == "Wind") & np.isin(state, ["FL", "LA"]))
        + 1.00 * ((cause == "Fire") & (tiv_ratio > 2))              # underinsured fire losses hit the limit
        + 0.80 * ((years < 3) & (prior >= 2))                       # new business with a bad history
        + 0.35 * (years < 3)                                        # step effect: new businesses
        + np.where((report_lag >= 15) & (report_lag <= 60), 0.55,   # non-monotonic lag effect
                   np.where(report_lag > 180, -0.40, 0.0))
        + 0.10 * prior
        + np.vectorize(lambda c: CAUSE_EFFECT.get(c, 0.0))(cause)
        + rng.normal(0, 0.70, n)
    )
    incurred = np.minimum(np.exp(log_sev), pol["policy_limit"].to_numpy()).round(2)

    claims = pd.DataFrame({
        "claim_id": [f"CLM-{i:06d}" for i in range(1, n + 1)],
        "policy_id": pol["policy_id"],
        "loss_date": loss_date.date,
        "loss_month": loss_date.month.astype("int64"),   # int64 like the other integer inputs (pandas gives int32)
        "cause_of_loss": cause,
        "report_lag_days": report_lag,
        "injury_flag": injury,
        "attorney_at_fnol": attorney,
        "incurred_12m": incurred,
    })
    claims["large_loss"] = (claims["incurred_12m"] >= LARGE_LOSS_THRESHOLD).astype(int)
    # A handful of cause_of_loss values are not reported at FNOL
    claims.loc[rng.random(n) < 0.03, "cause_of_loss"] = None
    return claims


def generate(n_policies: int = 20_000, n_claims: int = 40_000, seed: int = 7):
    policies = generate_policies(n_policies, seed)
    claims = generate_claims(policies, n_claims, seed + 4)
    return policies, claims
