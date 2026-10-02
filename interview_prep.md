# Interview Prep: Telco Customer Churn

Questions from each phase, with my answer, how to improve it, and a model answer.

---

## Phase 1: EDA (categorical associations)

### Q1. With 7,043 rows, why is the chi-squared p-value close to useless for ranking features, and what do you use instead?

**My answer**

I didn't answer this one. I hadn't run a chi-squared test directly; it's only used inside my `cramers_v` function.

**Tip to improve**

- `cramers_v` runs the chi-squared test inside it and prints its p-value by default (`verbose=True`). So I *was* looking at chi-squared results, and I should be able to say what they mean.
- The phrase interviewers listen for: **"statistical significance is not practical significance."**

**Model answer**

> The chi-squared test of independence only asks whether there is *any* association, not how strong it is. The test statistic grows with sample size: for the same effect, χ² roughly scales with n. With 7,043 rows, even a tiny real difference gives a very small p-value. `Contract` vs `Churn` gives p ≈ 10⁻²⁵⁸, but in practice it is no more "significant" than a weak feature.
>
> So I use the p-value only as a **gate**: is there any evidence of a relationship at all? I rank features by **effect size**. Cramér's V does that because it divides χ² by n and by the table's degrees of freedom, so it doesn't grow with sample size and stays on a 0–1 scale. `gender` is the contrast: V = 0.000 and a high p-value, so it fails even the gate.

---

### Q2. Why compute Theil's U when you already have Cramér's V? Give a concrete example where the asymmetry matters.

**My answer**

> I use Theil's U because it gives the entropy of the target variable with respect to the independent variable. Cramér's V doesn't have that, as it only tells about the association and if it's dependent.

**Tip to improve**

1. Theil's U doesn't "give the entropy." It gives the **fraction of the target's uncertainty (entropy) that is removed by knowing the feature**. U(Churn | Contract) = 0.17 means "knowing the contract removes 17% of my uncertainty about churn."
2. Cramér's V doesn't tell you "if it's dependent." That's the **p-value's** job. V tells you **how strong** the association is.
3. Missing keyword: **direction**. Cramér's V is **symmetric**, so V(A, B) = V(B, A). Theil's U is **asymmetric**: U(A | B) ≠ U(B | A).
4. Have a concrete example ready (below).

**Model answer**

> Cramér's V tells me how strongly two categorical variables are associated, but it is symmetric, so it can't tell me which variable predicts which. Theil's U (the uncertainty coefficient) is U(Y | X) = (H(Y) − H(Y | X)) / H(Y): the fraction of the uncertainty in Y removed by knowing X. It is asymmetric, so I can see when A fully determines B but B only partly determines A.
>
> For example, take `HasInternet` (whether `InternetService` is not "No") and `OnlineSecurity`. Cramér's V is **1.000**, a perfect association. But Theil's U shows:
> - U(HasInternet | OnlineSecurity) = **1.000**. If you know OnlineSecurity, you know HasInternet for certain, because "No internet service" means No and both "Yes" and "No" mean Yes.
> - U(OnlineSecurity | HasInternet) = **0.504**. Knowing the customer has internet still leaves you guessing whether they bought security (3,498 No vs 2,019 Yes).
>
> For feature selection I only need one direction, **U(Churn | feature)**, because that is what the model uses. That is the `Churn` row of my Theil's U heatmap.

**Supporting numbers (from my data)**

Crosstab of `OnlineSecurity` vs `HasInternet`:

| OnlineSecurity | HasInternet = No | HasInternet = Yes |
|---|---|---|
| No | 0 | 3,498 |
| No internet service | 1,526 | 0 |
| Yes | 0 | 2,019 |

Feature vs target:

| Feature | Cramér's V | U(Churn \| feature) | U(feature \| Churn) |
|---|---|---|---|
| Contract | 0.410 | **0.170** | 0.099 |
| PaymentMethod | 0.303 | **0.077** | 0.033 |
| gender | 0.000 | 0.000 | 0.000 |

⚠️ Don't compare U values with V values directly; they're on different scales. Contract's U = 0.17 is not "weaker" than its V = 0.41. Compare U only with other U values.

---

## Imbalanced target (26.5% churn)

### Q3. If you train Logistic Regression with and without `class_weight="balanced"`, would you expect ROC-AUC to change much? Which metrics would change, and why?

**My answer**

I didn't answer this one; I asked for the model answer.

**Tip to improve**

- Separate the two things a classifier gives you: a **ranking** of customers (who is riskier than whom) and a **decision** (who gets flagged at a threshold).
- ROC-AUC and PR-AUC judge the ranking only. Precision, recall, F1, accuracy and the confusion matrix judge the decision at a threshold.
- Class weights mostly move the probabilities, not the order, so ask yourself which metrics depend on the order and which depend on the cut-off.
- Key phrase: **"At this ratio, imbalance is mainly an evaluation and threshold problem, not a data problem."**

**Model answer**

> ROC-AUC should barely change. It only measures **ranking**: the probability that a random churner gets a higher score than a random non-churner. For logistic regression, `class_weight="balanced"` mostly shifts the **intercept** upward (the coefficients change only a little, through regularisation). Adding a constant inside the sigmoid raises every score but keeps their order, so ROC-AUC and PR-AUC stay almost the same.
>
> What *does* change is everything that depends on a **threshold**. At the default 0.5, the weighted model flags many more customers: **recall goes up, precision goes down**, F1 shifts, and accuracy drops. The **predicted probabilities** also get inflated: the average predicted churn probability rises well above the true 26.5% rate, so the model is no longer calibrated.
>
> So for LR, class weighting and threshold tuning do nearly the same job. Since I pick the threshold from out-of-fold predictions anyway, weighting adds little. Its main side effect is that I can't read the output as a true probability unless I recalibrate. Tree models are different: class weights change the split criterion, so the trees themselves change and AUC can move more. That's why I compare with and without weights instead of assuming they help.

**Supporting numbers (from my data)**

Logistic Regression, 5-fold stratified out-of-fold predictions (one-hot + scaled, `TotalCharges` dropped):

| `class_weight` | ROC-AUC | PR-AUC | Mean predicted p | Accuracy @0.5 | Precision @0.5 | Recall @0.5 | F1 @0.5 |
|---|---|---|---|---|---|---|---|
| None | 0.843 | 0.651 | 0.266 | 0.802 | 0.652 | 0.541 | 0.592 |
| balanced | 0.843 | 0.650 | **0.415** | 0.750 | 0.518 | **0.794** | 0.627 |

- The ranking metrics are identical, so weighting didn't make the model better at telling churners apart.
- At 0.5, recall jumps from 0.54 to 0.79 and precision falls from 0.65 to 0.52. That is the same trade-off you'd get by lowering the threshold on the unweighted model.
- The mean predicted probability goes from 0.266 (matches the true rate, so calibrated) to 0.415 (inflated). This matters for the Streamlit app, which shows a churn probability.

**Related points (imbalance plan for this project)**

- **EDA:** only be aware of the imbalance. Compare each group's churn rate against the 26.5% baseline; don't resample anything.
- **Split / CV:** `stratify=y` and `StratifiedKFold`, so every fold keeps about 26.5% churn.
- **Metrics:** not accuracy (the "always No" `DummyClassifier` scores 73.5%). Use PR-AUC, whose random baseline is 0.265, not 0.5.
- **Models:** `class_weight="balanced"` (LR, DT) and `scale_pos_weight ≈ 2.77` (XGB), compared with and without.
- **Threshold:** chosen from out-of-fold predictions on train (max F2, or recall ≥ 0.75 with the best precision), because missing a churner costs more than an unneeded retention offer.
- **Test set:** never resampled or re-weighted. If SMOTE were ever used, it would go inside an `imblearn` Pipeline so it runs on training folds only.
