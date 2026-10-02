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
