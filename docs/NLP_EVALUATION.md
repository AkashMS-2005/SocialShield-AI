# SocialShield AI – NLP Model Evaluation Report

*Generated: 2026-11-06 06:50 UTC*

## 1. Dataset & Task
- **Dataset:** UCI SMS Spam Collection (`SMSSpamCollection`)
- **Total Records:** 5,572 messages
- **Class Balance:** 4,825 legitimate (ham, 86.6%) vs 747 spam/phishing (13.4%)
- **Train / Test Split:** Stratified 80/20 (4,457 train, 1,115 test)
- **Model Architecture:** Scikit-learn `Pipeline([TfidfVectorizer(ngram_range=(1, 2), max_features=10000), LogisticRegression(class_weight='balanced')])`

## 2. Held-Out Test Set Performance

| Metric | Score | Explanation |
|--------|-------|-------------|
| **Accuracy** | `0.9794` | Overall fraction of correct predictions |
| **Precision** | `0.9257` | When flagged as spam-like, probability it is truly spam |
| **Recall** | `0.9195` | Fraction of actual spam caught |
| **F1-Score** | `0.9226` | Harmonic mean of precision and recall |
| **ROC-AUC** | `0.9897` | Discrimination ability across all thresholds |
| **Brier Score** | `0.0185` | Probability calibration error (lower is better, 0.0 is perfect) |

### Confusion Matrix (Test Set, N = 1115)

| | Predicted Non-Spam | Predicted Spam |
|---|---|---|
| **Actual Ham (Non-spam)** | 955 | 11 |
| **Actual Spam** | 12 | 137 |

## 3. Practical Interpretability & Top Predictive n-grams
The Logistic Regression coefficients enable direct interpretability:
- High positive coefficients identify phrases strongly associated with promotional spam or phishing (e.g., `free`, `claim`, `win`, `urgent`, `prize`, `txt`, `call now`).
- In inference, multiplying the TF-IDF feature weights of a new input text by these coefficients surfaces the exact words driving the spam score.

## 4. Ethical Limitations
1. **Context Transferability:** The model is trained on SMS messages. While mobile phishing patterns overlap heavily with social media spam (DMs, deceptive bios), social posts also feature platform-specific hashtags, mentions, and emoji clusters.
2. **Spam-Like ≠ Fake Account:** A genuine user might post a promotional link, and an automated bot might quote neutral text. Text analysis provides an indicator signal, **not proof of account inauthenticity**.
3. **Missing Text:** When an account has no bio or posts, text analysis returns `not_available` with an explicit limitation flag — missing text is **never** silently treated as safe.
