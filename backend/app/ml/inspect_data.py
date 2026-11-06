"""
ml/inspect_data.py
------------------
Inspects every dataset in backend/data/raw/ and prints / saves a Markdown report.

Usage (from backend/ directory, with venv active):
    python -m app.ml.inspect_data

Output:
    - Console: full report
    - docs/DATASET_REPORT.md: saved Markdown

Rule: never invent data, never treat missing values as zero risk.
"""
from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parents[2]          # backend/
RAW_DIR     = BACKEND_DIR / "data" / "raw"
DOCS_DIR    = BACKEND_DIR.parent / "docs"
REPORT_PATH = DOCS_DIR / "DATASET_REPORT.md"

DOCS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def section(title: str) -> str:
    return f"\n## {title}\n"

def subsection(title: str) -> str:
    return f"\n### {title}\n"

def _missing_pct(series: pd.Series) -> str:
    n = series.isna().sum()
    pct = 100 * n / max(len(series), 1)
    return f"{n} ({pct:.1f}%)"

def describe_df(df: pd.DataFrame, label_col: str | None = None) -> str:
    lines: list[str] = []
    lines.append(f"- **Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns")
    lines.append(f"- **Duplicate rows:** {df.duplicated().sum():,}")

    lines.append("\n**Column dtypes & missing values:**\n")
    lines.append("| Column | dtype | Missing |")
    lines.append("|--------|-------|---------|")
    for col in df.columns:
        lines.append(f"| `{col}` | {df[col].dtype} | {_missing_pct(df[col])} |")

    if label_col and label_col in df.columns:
        lines.append("\n**Label distribution:**\n")
        vc = df[label_col].value_counts(dropna=False)
        lines.append("| Label | Count | % |")
        lines.append("|-------|-------|---|")
        for lbl, cnt in vc.items():
            lines.append(f"| `{lbl}` | {cnt:,} | {100*cnt/len(df):.1f}% |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dataset 1: Instagram Fake Spammer Genuine Accounts  (Kaggle – manual DL)
# ---------------------------------------------------------------------------

INSTA_TRAIN = RAW_DIR / "train.csv"
INSTA_TEST  = RAW_DIR / "test.csv"

# Column name mapping from dataset → our scanner form fields
INSTA_COL_MAP = {
    "profile pic":           "has_profile_pic",
    "nums/length username":  "username_num_ratio    [derived]",
    "fullname words":        "fullname_words         [derived]",
    "nums/length fullname":  "fullname_num_ratio     [derived]",
    "name==username":        "name_eq_username       [derived]",
    "description length":    "bio                    [proxy: length only]",
    "external URL":          "url_in_bio             [derived]",
    "private":               "private                [not in form]",
    "#posts":                "posts",
    "#followers":            "followers",
    "#follows":              "following",
    "fake":                  "LABEL",
}

SCANNER_FORM_FIELDS = {
    "followers", "following", "posts",
    "account_age_days", "has_profile_pic", "bio", "post_text", "username",
}

INSTA_DIRECTLY_MATCHED = {
    "profile pic"  : "has_profile_pic",
    "#posts"       : "posts",
    "#followers"   : "followers",
    "#follows"     : "following",
    "description length": "bio (length proxy)",
    "external URL" : "url_in_bio",
}

INSTA_MISSING_VS_FORM = ["account_age_days", "post_text", "platform"]


def inspect_instagram() -> str:
    lines: list[str] = []
    lines.append(subsection("Dataset A – Instagram Fake Spammer Genuine Accounts"))
    lines.append(textwrap.dedent("""\
        - **Source:** Kaggle – [free4ever1/instagram-fake-spammer-genuine-accounts](https://www.kaggle.com/datasets/free4ever1/instagram-fake-spammer-genuine-accounts)
        - **Licence:** CC BY-SA 4.0
        - **Files:** `train.csv`, `test.csv`
        - **Label definition:** `fake=1` → spammer / fake; `fake=0` → genuine
        - **Platform scope:** Instagram only
    """))

    if not INSTA_TRAIN.exists():
        lines.append(textwrap.dedent("""\
            > ⚠️ **MANUAL DOWNLOAD REQUIRED**
            >
            > The Instagram dataset requires a Kaggle account.
            > 1. Visit: https://www.kaggle.com/datasets/free4ever1/instagram-fake-spammer-genuine-accounts
            > 2. Click **Download** (you may need to accept terms).
            > 3. Extract `train.csv` and `test.csv` into:
            >    ```
            >    backend/data/raw/train.csv
            >    backend/data/raw/test.csv
            >    ```
            > 4. Re-run this script.
            >
            > **Known columns (from published research):**
            > `profile pic`, `nums/length username`, `fullname words`,
            > `nums/length fullname`, `name==username`, `description length`,
            > `external URL`, `private`, `#posts`, `#followers`, `#follows`, `fake`
        """))
        lines.append("\n**Feature mapping (based on documented columns):**\n")
        lines.append("| Dataset column | Scanner form field | Status |")
        lines.append("|----------------|--------------------|--------|")
        for ds_col, form_field in INSTA_DIRECTLY_MATCHED.items():
            lines.append(f"| `{ds_col}` | `{form_field}` | ✅ Present |")
        lines.append(f"| `private` | *(not in form)* | ➕ Could add |")
        lines.append(f"| `nums/length username` | *(derived from username)* | 🔧 Derivable |")
        for mf in INSTA_MISSING_VS_FORM:
            lines.append(f"| *(absent)* | `{mf}` | ❌ Missing |")
        return "\n".join(lines)

    # File exists – inspect for real
    df_train = pd.read_csv(INSTA_TRAIN)
    df_test  = pd.read_csv(INSTA_TEST) if INSTA_TEST.exists() else pd.DataFrame()
    df_all   = pd.concat([df_train, df_test], ignore_index=True) if not df_test.empty else df_train

    lines.append(f"\n**Train file:** `{INSTA_TRAIN.name}` | **Test file:** `{INSTA_TEST.name if INSTA_TEST.exists() else 'not found'}`\n")
    lines.append(describe_df(df_all, label_col="fake"))

    # Feature mapping
    lines.append("\n**Feature mapping:**\n")
    lines.append("| Dataset column | Scanner form field | Status |")
    lines.append("|----------------|--------------------|--------|")
    for ds_col in df_all.columns:
        mapped = INSTA_COL_MAP.get(ds_col, "*(no direct map)*")
        status = "✅ Present" if ds_col in INSTA_DIRECTLY_MATCHED else ("🔧 Derivable" if "derived" in mapped.lower() else "➕ Could add")
        if ds_col == "fake":
            status = "🏷️ Label"
        lines.append(f"| `{ds_col}` | `{mapped}` | {status} |")
    for mf in INSTA_MISSING_VS_FORM:
        lines.append(f"| *(absent)* | `{mf}` | ❌ Missing from dataset |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dataset 2: SMS Spam Collection (UCI / Kaggle – already downloaded)
# ---------------------------------------------------------------------------

SMS_FILE = RAW_DIR / "SMSSpamCollection"

def inspect_sms() -> str:
    lines: list[str] = []
    lines.append(subsection("Dataset B – UCI SMS Spam Collection (NLP module)"))
    lines.append(textwrap.dedent("""\
        - **Source:** UCI ML Repository – [SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
        - **Direct URL:** `https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip`
        - **Licence:** Creative Commons Attribution (open access, academic use)
        - **Files:** `SMSSpamCollection` (tab-separated, no header)
        - **Label definition:** `spam` → spam/phishing SMS; `ham` → legitimate SMS
        - **Purpose for SocialShield:** Train TF-IDF spam classifier for `post_text` / bio NLP analysis
    """))

    if not SMS_FILE.exists():
        lines.append("> ❌ File `SMSSpamCollection` not found in `backend/data/raw/`. Run the download command.")
        return "\n".join(lines)

    df = pd.read_csv(
        SMS_FILE,
        sep="\t",
        header=None,
        names=["label", "text"],
        encoding="utf-8",
    )

    lines.append(describe_df(df, label_col="label"))

    # Text stats
    df["text_length"] = df["text"].str.len()
    lines.append("\n**Text length statistics:**\n")
    lines.append("| Stat | Value |")
    lines.append("|------|-------|")
    for stat, val in df["text_length"].describe().items():
        lines.append(f"| {stat} | {val:.1f} |")

    lines.append("\n**Sample spam messages (first 3):**\n")
    for _, row in df[df["label"] == "spam"].head(3).iterrows():
        preview = row["text"][:100].replace("|", "\\|")
        lines.append(f"- `{preview}…`")

    lines.append("\n**Applicability to SocialShield:**")
    lines.append("- ✅ SMS spam language patterns transfer well to social media spam/phishing posts")
    lines.append("- ⚠️ SMS context differs from social post context (no hashtags, no @mentions in raw data)")
    lines.append("- ✅ Large enough (5,574 msgs) to train a TF-IDF classifier with good generalisation")
    lines.append("- ⚠️ Label is 'spam' not 'fake account' — only the NLP submodule uses this dataset")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Synthetic fallback for Instagram (created if train.csv absent)
# ---------------------------------------------------------------------------

SYNTH_FILE = RAW_DIR / "synthetic_SAMPLE_ONLY.csv"

def create_synthetic_instagram() -> str:
    """
    Create a minimal synthetic sample ONLY for interface/pipeline testing.
    Clearly named and labelled – must never be used for real-world metrics.
    """
    rng = np.random.default_rng(42)
    n = 500

    # Fake accounts heuristics (from literature, not from real data)
    fake_mask = np.zeros(n, dtype=bool)
    fake_mask[:200] = True  # 200 fake, 300 genuine
    rng.shuffle(fake_mask)

    df = pd.DataFrame({
        "profile pic":          rng.integers(0, 2, n).astype(int),
        "nums/length username":  rng.uniform(0, 0.6, n).round(3),
        "fullname words":        rng.integers(0, 5, n),
        "nums/length fullname":  rng.uniform(0, 0.4, n).round(3),
        "name==username":        rng.integers(0, 2, n),
        "description length":    rng.integers(0, 150, n),
        "external URL":          rng.integers(0, 2, n),
        "private":               rng.integers(0, 2, n),
        "#posts":                rng.integers(0, 300, n),
        "#followers":            rng.integers(0, 5000, n),
        "#follows":              rng.integers(0, 7500, n),
        "fake":                  fake_mask.astype(int),
    })

    # Apply loose statistical bias toward fake-looking fake accounts
    df.loc[fake_mask, "profile pic"]          = rng.integers(0, 2, fake_mask.sum()).astype(df["profile pic"].dtype)
    df.loc[fake_mask, "#followers"]           = rng.integers(0, 300, fake_mask.sum())
    df.loc[fake_mask, "#follows"]             = rng.integers(500, 7500, fake_mask.sum())
    df.loc[fake_mask, "description length"]   = rng.integers(0, 20, fake_mask.sum())
    df.loc[fake_mask, "#posts"]               = rng.integers(0, 20, fake_mask.sum())

    df.to_csv(SYNTH_FILE, index=False)
    return f"Created synthetic sample: {SYNTH_FILE} ({n} rows)"


# ---------------------------------------------------------------------------
# Main report builder
# ---------------------------------------------------------------------------

def build_report() -> str:
    lines: list[str] = []
    lines.append("# SocialShield AI – Dataset Report")
    lines.append(f"\n*Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}*\n")

    lines.append("> [!CAUTION]")
    lines.append("> Outputs from models trained on these datasets are **risk indicators only**.")
    lines.append("> A `fake=1` label means the account was classified as a spammer/fake in the")
    lines.append("> original dataset — it does **not** prove the account owner's real-world identity.")

    # -----------------------------------------------------------------------
    lines.append(section("1. Dataset Comparison"))
    lines.append("""\
| Property | Dataset A (Instagram) | Dataset B (SMS Spam) |
|----------|-----------------------|----------------------|
| **Purpose** | Account classification (ML) | NLP spam detection |
| **Source** | Kaggle (free4ever1) | UCI ML Repository |
| **Licence** | CC BY-SA 4.0 | CC Attribution |
| **Size (approx)** | ~700–1,100 rows | 5,574 SMS messages |
| **Label** | `fake` (0/1) | `spam`/`ham` |
| **Label definition** | Spammer/inauthentic vs genuine | Spam SMS vs legitimate SMS |
| **Platform** | Instagram only | SMS (cross-applicable) |
| **Has followers** | ✅ `#followers` | ❌ |
| **Has following** | ✅ `#follows` | ❌ |
| **Has posts** | ✅ `#posts` | ❌ |
| **Has account age** | ❌ Not present | ❌ |
| **Has profile pic** | ✅ `profile pic` (binary) | ❌ |
| **Has bio** | ✅ `description length` (length only, no raw text) | ❌ |
| **Has post text** | ❌ Not present | ✅ `text` column |
| **Has username** | ✅ Ratio features derived | ❌ |
| **Has URL in bio** | ✅ `external URL` (binary) | ❌ |
| **Bot ≠ Fake caveat** | Labelled as 'fake/spammer' — not bot detection | Spam ≠ fake identity |
| **Credentials required** | ✅ Kaggle login | ❌ Public direct download |
""")

    # -----------------------------------------------------------------------
    lines.append(section("2. Instagram Dataset Detail"))
    lines.append(inspect_instagram())

    synth_note = ""
    if not INSTA_TRAIN.exists():
        synth_msg = create_synthetic_instagram()
        synth_note = textwrap.dedent(f"""
> [!WARNING]
> **SYNTHETIC DATA NOTICE — NOT REAL-WORLD PERFORMANCE**
>
> `train.csv` was not found. A synthetic sample has been generated at:
> `backend/data/raw/synthetic_SAMPLE_ONLY.csv`
>
> This file exists **only** for interface and pipeline testing. Any metric
> (accuracy, F1, AUC) computed from it is **meaningless as a real-world measure**.
> Replace it with the real Kaggle dataset before claiming any performance figures.
>
> {synth_msg}
""")
        lines.append(synth_note)

    # -----------------------------------------------------------------------
    lines.append(section("3. SMS / NLP Dataset Detail"))
    lines.append(inspect_sms())

    # -----------------------------------------------------------------------
    lines.append(section("4. TwiBot-20 Assessment"))
    lines.append(textwrap.dedent("""\
        TwiBot-20 was evaluated and **excluded** for the following reasons:

        | Reason | Detail |
        |--------|--------|
        | **Access restriction** | Requires direct email request to lead author (shangbin@cs.washington.edu); no public download |
        | **Platform scope** | Twitter/X only — does not cover Instagram, Facebook, YouTube |
        | **Label definition mismatch** | Labels *bot vs human*, not *fake/spammer vs genuine* |
        | **Format complexity** | JSON with nested tweets and graph neighbours — significant preprocessing overhead for a 12-hour hackathon |
        | **Conclusion** | Use Instagram dataset for account-level ML; re-evaluate TwiBot-22 for a future multi-platform expansion |
    """))

    # -----------------------------------------------------------------------
    lines.append(section("5. Label Limitations & Ethical Notes"))
    lines.append(textwrap.dedent("""\
        | Limitation | Explanation |
        |------------|-------------|
        | `fake=1` ≠ proved fake identity | Dataset labellers used heuristics; ground truth is noisy |
        | No account age in Instagram data | A key inauthentic-account signal is absent — model is weaker on recency |
        | No raw bio text | Only bio length is available; actual NLP on bio content is not possible from this data |
        | No post text | NLP on posts trained separately (SMS proxy); transfer may not be perfect |
        | Instagram-only training | Applying the model to Facebook/X/YouTube is **out-of-distribution**; risk scores must carry a caveat |
        | Class imbalance risk | Dataset is roughly balanced (verify after download); if not, apply SMOTE or class weights |
        | Scraping provenance | Kaggle dataset was scraped pre-API-restriction; exact collection date unknown |
    """))

    # -----------------------------------------------------------------------
    lines.append(section("6. Proposed Training Feature Set"))
    lines.append(textwrap.dedent("""\
        The intersection of **available in Instagram dataset** ∩ **collectable in scanner form**:

        | # | Feature name (in code) | Source in dataset | Source in scanner form | Notes |
        |---|------------------------|-------------------|-----------------------|-------|
        | 1 | `has_profile_pic` | `profile pic` (0/1) | `has_profile_pic` | Direct match |
        | 2 | `followers` | `#followers` | `followers` | Direct match |
        | 3 | `following` | `#follows` | `following` | Direct match |
        | 4 | `posts` | `#posts` | `posts` | Direct match |
        | 5 | `bio_length` | `description length` | derived from `bio` | Proxy — length only |
        | 6 | `has_url_in_bio` | `external URL` (0/1) | derived from `bio` text | Derivable |
        | 7 | `following_ratio` | derived: `#follows / (#followers+1)` | derived | High-signal ratio |
        | 8 | `username_num_ratio` | `nums/length username` | derived from `username` | Derivable |
        | 9 | `name_eq_username` | `name==username` (0/1) | *(no fullname in form)* | ⚠️ Drop unless form adds fullname |
        | 10 | `is_private` | `private` (0/1) | *(not in form)* | ➕ Optional form field |

        **Features excluded (not in both sources):**
        - `account_age_days` — absent from dataset; present in form → use as a heuristic-only signal, not in ML model
        - `post_text` — absent from Instagram dataset; NLP from SMS dataset instead
        - `platform` — dataset is Instagram-only; encode as categorical if multi-platform data added later

        > [!IMPORTANT]
        > **Awaiting your approval on this feature set before writing any preprocessing code.**
        > Please confirm, modify, or add features. Once approved, Step 3 will implement the pipeline.
    """))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("SocialShield AI – Dataset Inspection")
    print("=" * 70)

    report = build_report()

    # Save to docs/
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"\n✅ Report saved to: {REPORT_PATH}")
    print("\n" + "=" * 70)
    print(report)
