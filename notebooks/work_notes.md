#### Notes 1: how to read `cv_lr_df` (Logistic Regression, 5-fold CV)

**The table**
- **1 row = 1 fold.** Each fold trains the whole Pipeline on 4/5 of `X_train` and scores the remaining 1/5 (≈ 1,056 customers).
- **`test_*` = the validation fold**, *not* `X_test`. That's sklearn's naming; the real test set stays untouched.
- `fit_time` / `score_time`: seconds to train / to predict + score. Useful later to compare slow vs fast models.

**`.agg(["mean", "std"]).T`**
- **mean** = the score to expect on new customers. Use it to **compare models**.
- **std** = how much the score moves between folds, i.e. how **stable** (trustworthy) the mean is.
- Rule of thumb: if two models' means differ by **less than about one std**, the difference may just be noise from how the folds were split. Don't call a winner on it.

**Is a small std good?** Yes. It means the score doesn't depend on which customers ended up in which fold. For ≈ 1,000 rows per fold:

| std | Reading |
|---|---|
| < 0.02 | Very stable |
| 0.02 – 0.05 | Normal for this data size (our P, R, F1, PR-AUC ≈ 0.03–0.05) |
| > 0.05 | Unstable: check the data or the model before trusting the mean |

**What counts as a good score?**

| Metric | Random model | Our LR | Good on this dataset | Warning sign |
|---|---|---|---|---|
| ROC-AUC | 0.50 | **0.85 ± 0.02** | 0.83 – 0.86 (typical tuned models) | > 0.90 → suspect leakage |
| PR-AUC | 0.265 (= churn rate) | **0.66 ± 0.05** | 0.60 – 0.70 | Near 0.265 → model learned nothing |
| Precision / Recall / F1 | depends on threshold | 0.68 / 0.53 / 0.59 | No fixed range: they change with the threshold (phase 6) | n/a |

ROC-AUC and PR-AUC don't use a cut-off, so they get "good ranges". Precision, recall and F1 here use the default 0.5 cut-off, so they are only a first look.
