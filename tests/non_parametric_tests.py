import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exploring_dataset.visualisation import plot_visualizations

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import time
import numpy as np
from tabulate import tabulate
import scikit_posthocs as sp


def wilcoxon_test(df, subject_col, dependent_col, timepoint_col, timepoint1, timepoint2):
    """
    Parameters:
    - df: DataFrame with data
    - subject_col: identifier for subjects (to match pairs)
    - dependent_col: the numerical column to test (e.g., 'tumor_size')
    - timepoint_col: column indicating timepoints (e.g., 'timepoint')
    - timepoint1: first timepoint (string)
    - timepoint2: second timepoint (string)
    """
    print(f"\nWilcoxon Signed-Rank Test between {timepoint1} and {timepoint2} for {dependent_col}:")
    
    # Pivot to align timepoints side-by-side
    pivoted = df[df[timepoint_col].isin([timepoint1, timepoint2])].pivot(
        index=subject_col,
        columns=timepoint_col,
        values=dependent_col
    ).dropna()

    if len(pivoted) < 2:
        print("Not enough paired samples for Wilcoxon test.")
        return
    
    stat, p_value = stats.wilcoxon(pivoted[timepoint1], pivoted[timepoint2])
    print(tabulate(
        [[stat, p_value, "Significant" if p_value < 0.05 else "Not significant"]],
        headers=["Statistic", "P-Value", "Result"],
        tablefmt="grid"
    ))
    if p_value < 0.05:
        print("Reject the null hypothesis: significant difference between timepoints.")
    else:
        print("Fail to reject the null hypothesis: no significant difference between timepoints.")
    pivoted['diff'] = pivoted['month_6'] - pivoted['baseline']
    print(f"Median difference: {pivoted['diff'].median()}") 
    
    return {
        'n_pairs': len(pivoted),
        "statistic": stat,
        "p_value": p_value,
    }


def friedman_test(df, subject_col, dependent_col, timepoint_col, timepoints, alpha=0.05):
    """
     Parameters:
    - df: DataFrame with data
    - subject_col: identifier for subjects (to align repeated measures)
    - dependent_col: the numerical column to test (e.g., 'tumor_size')
    - timepoint_col: column indicating timepoints (e.g., 'timepoint')
    - timepoints: list of timepoints to include (e.g., ['baseline', 'month_6', 'month_12'])
    """
    print(f"\nFriedman Test across {', '.join(timepoints)} for {dependent_col}:")

    # Pivot to align timepoints side-by-side
    pivoted = df[df[timepoint_col].isin(timepoints)].pivot(
        index=subject_col,
        columns=timepoint_col,
        values=dependent_col
    ).dropna()

    if pivoted.shape[0] < 10:
        print("Not enough complete cases to perform the test.")
        return None

    stat, p_value = stats.friedmanchisquare(*[pivoted[tp] for tp in timepoints])

    print(tabulate(
        [[stat, p_value, "Significant" if p_value < 0.05 else "Not significant"]],
        headers=["Statistic", "P-Value", "Result"],
        tablefmt="grid"
    ))

    friedman_results = {
        'n_subjects': pivoted.shape[0],
        'statistic': round(float(stat), 4),
        'p_value': round(float(p_value), 6),
        'posthoc': None
    }

    if p_value < alpha:
        print("H-0 rejected: performing Nemenyi post-hoc test.")
        long_df = pivoted.reset_index().melt(
            id_vars=subject_col,
            value_vars=timepoints,
            var_name='timepoint',
            value_name='value'
        )
        long_df = pivoted.reset_index().melt(
            id_vars=subject_col,
            value_vars=timepoints,
            var_name='timepoint',
            value_name='value'
        )
        nemenyi_result = sp.posthoc_nemenyi_friedman(pivoted[timepoints]) # timepoint as group, subject_id as individual

        print("Nemenyi post-hoc p-values:")
        print(tabulate(
            nemenyi_result,
            headers=nemenyi_result.columns,
            showindex=True,
            tablefmt="grid",
            floatfmt=".4f"
        ))

        friedman_results['posthoc'] = nemenyi_result
        plot_visualizations(long_df, nemenyi_result, dependent_col)
    else:
        print("H-0 not rejected: no significant difference detected. No need for post-hoc tests.")

    return friedman_results


