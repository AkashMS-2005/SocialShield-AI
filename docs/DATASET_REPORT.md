# SocialShield AI – Dataset Report

*Generated: 2026-11-06 06:14 UTC*

> [!CAUTION]
> Outputs from models trained on these datasets are **risk indicators only**.
> A `fake=1` label means the account was classified as a spammer/fake in the
> original dataset — it does **not** prove the account owner's real-world identity.

## 1. Dataset Comparison

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


## 2. Instagram Dataset Detail


### Dataset A – Instagram Fake Spammer Genuine Accounts

- **Source:** Kaggle – [free4ever1/instagram-fake-spammer-genuine-accounts](https://www.kaggle.com/datasets/free4ever1/instagram-fake-spammer-genuine-accounts)
- **Licence:** CC BY-SA 4.0
- **Files:** `train.csv`, `test.csv`
- **Label definition:** `fake=1` → spammer / fake; `fake=0` → genuine
- **Platform scope:** Instagram only

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


**Feature mapping (based on documented columns):**

| Dataset column | Scanner form field | Status |
|----------------|--------------------|--------|
| `profile pic` | `has_profile_pic` | ✅ Present |
| `#posts` | `posts` | ✅ Present |
| `#followers` | `followers` | ✅ Present |
| `#follows` | `following` | ✅ Present |
| `description length` | `bio (length proxy)` | ✅ Present |
| `external URL` | `url_in_bio` | ✅ Present |
| `private` | *(not in form)* | ➕ Could add |
| `nums/length username` | *(derived from username)* | 🔧 Derivable |
| *(absent)* | `account_age_days` | ❌ Missing |
| *(absent)* | `post_text` | ❌ Missing |
| *(absent)* | `platform` | ❌ Missing |

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
> Created synthetic sample: C:\Akash M S\SocialShield AI\backend\data\raw\synthetic_SAMPLE_ONLY.csv (500 rows)


## 3. SMS / NLP Dataset Detail


### Dataset B – UCI SMS Spam Collection (NLP module)

- **Source:** UCI ML Repository – [SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
- **Direct URL:** `https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip`
- **Licence:** Creative Commons Attribution (open access, academic use)
- **Files:** `SMSSpamCollection` (tab-separated, no header)
- **Label definition:** `spam` → spam/phishing SMS; `ham` → legitimate SMS
- **Purpose for SocialShield:** Train TF-IDF spam classifier for `post_text` / bio NLP analysis

- **Shape:** 5,572 rows × 2 columns
- **Duplicate rows:** 403

**Column dtypes & missing values:**

| Column | dtype | Missing |
|--------|-------|---------|
| `label` | object | 0 (0.0%) |
| `text` | object | 0 (0.0%) |

**Label distribution:**

| Label | Count | % |
|-------|-------|---|
| `ham` | 4,825 | 86.6% |
| `spam` | 747 | 13.4% |

**Text length statistics:**

| Stat | Value |
|------|-------|
| count | 5572.0 |
| mean | 80.5 |
| std | 59.9 |
| min | 2.0 |
| 25% | 36.0 |
| 50% | 62.0 |
| 75% | 122.0 |
| max | 910.0 |

**Sample spam messages (first 3):**

- `Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entr…`
- `FreeMsg Hey there darling it's been 3 week's now and no word back! I'd like some fun you up for it s…`
- `WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim…`

**Applicability to SocialShield:**
- ✅ SMS spam language patterns transfer well to social media spam/phishing posts
- ⚠️ SMS context differs from social post context (no hashtags, no @mentions in raw data)
- ✅ Large enough (5,574 msgs) to train a TF-IDF classifier with good generalisation
- ⚠️ Label is 'spam' not 'fake account' — only the NLP submodule uses this dataset

## 4. TwiBot-20 Assessment

TwiBot-20 was evaluated and **excluded** for the following reasons:

| Reason | Detail |
|--------|--------|
| **Access restriction** | Requires direct email request to lead author (shangbin@cs.washington.edu); no public download |
| **Platform scope** | Twitter/X only — does not cover Instagram, Facebook, YouTube |
| **Label definition mismatch** | Labels *bot vs human*, not *fake/spammer vs genuine* |
| **Format complexity** | JSON with nested tweets and graph neighbours — significant preprocessing overhead for a 12-hour hackathon |
| **Conclusion** | Use Instagram dataset for account-level ML; re-evaluate TwiBot-22 for a future multi-platform expansion |


## 5. Label Limitations & Ethical Notes

| Limitation | Explanation |
|------------|-------------|
| `fake=1` ≠ proved fake identity | Dataset labellers used heuristics; ground truth is noisy |
| No account age in Instagram data | A key inauthentic-account signal is absent — model is weaker on recency |
| No raw bio text | Only bio length is available; actual NLP on bio content is not possible from this data |
| No post text | NLP on posts trained separately (SMS proxy); transfer may not be perfect |
| Instagram-only training | Applying the model to Facebook/X/YouTube is **out-of-distribution**; risk scores must carry a caveat |
| Class imbalance risk | Dataset is roughly balanced (verify after download); if not, apply SMOTE or class weights |
| Scraping provenance | Kaggle dataset was scraped pre-API-restriction; exact collection date unknown |


## 6. Proposed Training Feature Set

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
