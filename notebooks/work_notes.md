# 1. How to read `cv_lr_df` (Logistic Regression, 5-fold CV)

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

---

# 2. How a Decision Tree chooses its root feature?

The tree uses a **greedy search** for the split that best separates churners from non-churners.

### Step 1: Measure how "mixed" a group is (impurity)

sklearn uses **Gini impurity** by default:  
$\text{Gini} = 1 - p(\text{churn})^2 - p(\text{stay})^2$

* **A pure group** (everyone stays, or everyone churns) has $\text{Gini} = 0$.
* **A 50/50 mix** has $\text{Gini} = 0.5$, the worst possible for two classes.
* **All training customers** ($26.5\%$ churn) have $\text{Gini} = 1 - 0.265^2 - 0.735^2 \approx \mathbf{0.39}$.

### Step 2: Try every possible split

The tree tries every column of the preprocessed data:

* **One-hot columns** (e.g. `Contract_Month-to-month`) have only one possible split: is it 0 or 1?
* **Numeric columns** (`tenure`, `MonthlyCharges`): the tree sorts the values and tries a threshold between each pair of neighbouring values, e.g. "$\text{tenure} \le 5.5$?", "$\text{tenure} \le 6.5$?" and so on.

### Step 3: Score each split

Each split creates a left and a right group. The tree computes the **weighted average Gini** of the two groups (weighted by their sizes) and how much it fell from the parent's Gini:  
$\text{Gini decrease} = \text{Gini}(\text{parent}) - [\text{n\_left}/\text{n} \times \text{Gini}(\text{left}) + \text{n\_right}/\text{n} \times \text{Gini}(\text{right})]$

### Step 4: Pick the largest decrease

The split with the biggest decrease becomes the root.

**A rough example (approximate churn rates from the EDA):** "Contract = month-to-month?" separates a group with $\sim 43\%$ churn from a group with $\sim 7\%$ churn. That leaves both groups much purer than the $26.5\%$ mix, which gives a large Gini decrease. That's why `Contract_Month-to-month` (or `tenure`) is the most likely root.

### Step 5: Repeat inside each group

The same search runs again inside each child group. It stops when it hits `max_depth` or `min_samples_leaf`, or when no split improves purity.

### Three interview points

1. **The search is greedy.** The tree picks the best split *right now* and never looks ahead to check whether a weaker first split would allow better splits later. So a single tree is not guaranteed to be optimal.
2. **There are other criteria.** `criterion="entropy"` (information gain) is the alternative. It almost always picks the same splits as Gini, and Gini is slightly cheaper to compute.
3. **Root $\ne$ "most important feature" overall.** The root is the best *single* split on *all* customers. Lower splits only act on subgroups. Feature importance adds up the Gini decreases over every split where a feature is used. It's related to the single-feature AUC scan you did in L5 (Contract at 0.739), but it's a different measure.

# 3. Is onehotencoder needed for tree model types like decision tree, random forest or xgboost?

**Short answer:** tree models need *some* encoding unless the library handles categories natively, but it doesn't have to be one-hot.

| Model | Needs encoding? | Options |
| :--- | :--- | :--- |
| **Decision Tree, Random Forest**<br>(sklearn) | **Yes.** They only accept numbers and fail on strings like `"Month-to-month"`. | One-hot or `OrdinalEncoder` |
| **XGBoost** | Optional | Native categorical support with `enable_categorical=True` plus pandas `category` dtype (hist tree method) |
| **LightGBM** | Optional | Native via `categorical_feature` |
| **CatBoost** | Optional | Native via `cat_features`, which is its main strength |

**One-hot vs ordinal for trees:**

* `OrdinalEncoder` maps each level to an integer, e.g. Month-to-month=0, One year=1, Two year=2. The order is fake for nominal features such as `PaymentMethod`, but a tree can still isolate any level with a few splits ("$\le 0.5$", then "$\le 1.5$"). It also produces fewer columns. For LR it would be wrong, because LR would read the numbers as a linear scale.
* **One-hot** gives one 0/1 column per level, so every split is "this level vs the rest". It's clean and readable. With high-cardinality features (hundreds of levels) it creates many sparse columns, but here the largest feature has only 4 levels, so it's fine.

**Why this project uses one-hot for trees in cycle 1:**

1. **Fair comparison.** All tree models get the same input, so score differences come from the model, not the encoding. The spec says this for step 7.
2. **Readable rules.** `plot_tree` shows `Contract_Month-to-month <= 0.5` instead of `Contract <= 0.5` on a made-up integer scale.
3. **Low cardinality.** It costs almost nothing here.

Cycle 2 can test **native categorical handling** (CatBoost's `cat_features`, LightGBM, XGBoost's `enable_categorical`) to see whether it beats one-hot. That's already noted in the spec for CatBoost.