"""Association measures between categorical variables, used in the EDA notebooks."""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, entropy
import matplotlib.pyplot as plt
import seaborn as sns


def cramers_v(x, y, bias_correction=True, alpha=0.05, verbose=True):
    """
    Cramer's V between two categorical series.
    :param verbose: If True, prints the hypothesis test results. If False, suppresses output.
    """
    x_name = getattr(x, 'name', 'x')
    y_name = getattr(y, 'name', 'y')

    crosstab = pd.crosstab(x, y)                       
    r, k = crosstab.shape
    if min(r, k) < 2:                                  
        return np.nan, np.nan
        
    chi2, p_value, _, _ = chi2_contingency(crosstab, correction=False)  
    
    # --- VERBOSE SWITCH ADDED HERE ---
    if verbose:
        if p_value < alpha:
            print(f"p-value ({p_value:.5f}) < {alpha}: Reject the Null Hypothesis. '{x_name}' and '{y_name}' are DEPENDENT.")
        else:
            print(f"p-value ({p_value:.5f}) >= {alpha}: Fail to reject the Null Hypothesis. '{x_name}' and '{y_name}' are INDEPENDENT.")
    # ---------------------------------
        
    n = crosstab.to_numpy().sum()
    phi2 = chi2 / n
    if not bias_correction:
        return np.sqrt(phi2 / min(k - 1, r - 1)), p_value
        
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    denom = min(kcorr - 1, rcorr - 1)
    
    if denom <= 0:
        return np.nan, p_value
        
    return np.sqrt(phi2corr / denom), p_value


def theils_u(x, y):
    """
    Theil's U (uncertainty coefficient) U(y|x): the fraction of the uncertainty (entropy) of y
    that is removed by knowing x. Asymmetric: theils_u(x, y) != theils_u(y, x). Range [0, 1].
        U(y|x) = (H(y) - H(y|x)) / H(y)
    :param x: the independent variable (predictor/feature)
    :param y: the dependent variable (target/being predicted)
    """
    # Create crosstab where rows are the target (y) and cols are the predictor (x)
    crosstab = pd.crosstab(y, x).to_numpy()            
    n = crosstab.sum()
    
    h_y = entropy(crosstab.sum(axis=1) / n)            # H(y) - Entropy of the target
    
    if h_y == 0:                                       # y is constant -> nothing left to explain
        return 1.0
        
    p_x = crosstab.sum(axis=0) / n                     # Probabilities of the predictor classes
    
    # H(y|x) - Conditional entropy of y given x
    h_y_given_x = sum(p * entropy(col / col.sum()) for p, col in zip(p_x, crosstab.T))  
    
    return (h_y - h_y_given_x) / h_y



def cramer_matrix(data, columns, plot=True):
    """
    Generates a pairwise association matrix using Cramer's V (symmetric).
    """
    # Build the matrix using the updated cramers_v function (verbose=False to suppress prints)
    matrix = pd.DataFrame(
        [[cramers_v(data[r], data[c], verbose=False)[0] for c in columns] for r in columns],
        index=columns, columns=columns
    ).astype(float)
    
    if plot:
        plt.figure(figsize=(10, 8))
        
        # Cramer's V is symmetric, so we mask the redundant upper triangle
        mask = np.triu(np.ones_like(matrix, dtype=bool))
        
        sns.heatmap(matrix, 
                    mask=mask, 
                    annot=True, 
                    fmt=".3f", 
                    cmap="YlGnBu", 
                    linewidths=1, 
                    linecolor='white',
                    cbar_kws={"shrink": .8})
        
        plt.title("Cramer's V Correlation Heatmap (Symmetric)")
        plt.tight_layout()
        plt.show()
        
    return


def theil_matrix(data, columns, plot=True):
    """
    Generates a pairwise association matrix using Theil's U (asymmetric).
    Cell [row, col] = U(row | col), i.e., how well the column (predictor) predicts the row (target).
    """
    # Build the matrix using the updated theils_u function (c = predictor/x, r = target/y)
    matrix = pd.DataFrame(
        [[theils_u(x=data[c], y=data[r]) for c in columns] for r in columns],
        index=columns, columns=columns
    ).astype(float)
    
    if plot:
        plt.figure(figsize=(10, 8))
        
        # Theil's U is asymmetric, so we DO NOT mask the upper triangle. 
        # The full square is required to see both U(x|y) and U(y|x).
        sns.heatmap(matrix, 
                    annot=True, 
                    fmt=".3f", 
                    cmap="YlGnBu", 
                    linewidths=1, 
                    linecolor='white',
                    cbar_kws={"shrink": .8})
        
        plt.title("Theil's U Predictive Heatmap (Asymmetric)\nHow well the X-axis predicts the Y-axis")
        plt.tight_layout()
        plt.show()
        
    return