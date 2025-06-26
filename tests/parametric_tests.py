import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import time
import numpy as np
from tabulate import tabulate


def ttest_independent(df, dependent_col, independent_col):
    """
    df: dataframe z danymi
    dependent_col: zmienna zależna (ilościowa)
    independent_col: zmienna niezależna (dichotomiczna: 2 grupy)
    """

    """
    Założenia testu t-Studenta dla prób niezależnych:
        1. Pomiar ilościowy zmiennej zależnej —
        2. Zmienna niezależna dychotomiczna —
        3. Normalność rozkładu w grupach - Test Shapiro-Wilk
        4. Jednorodność wariancji - Test Levene\'a
        5. Równoliczność osób w grupach - Test niezależności chi-kwadrat
    """

    print(f"\n\nT-test for {dependent_col} by {independent_col}:")
    # Step 1: Check if the independent variable has exactly 2 groups
    groups = df[independent_col].dropna().unique()
    if len(groups) != 2:
        print(f"The independent variable '{independent_col}' is NOT dichotomous (found {len(groups)} groups).")
        return
    
    group1 = df[df[independent_col] == groups[0]][dependent_col].dropna()
    group2 = df[df[independent_col] == groups[1]][dependent_col].dropna()

    # Step 2: Shapiro-Wilk test for normality
    stat1, p1 = stats.shapiro(group1)
    stat2, p2 = stats.shapiro(group2)
    print(f"Shapiro-Wilk Test results:")
    print(f"- {groups[0]}: p={p1:.4f} {'Normal' if p1 > 0.05 else 'Not normal'}")
    print(f"- {groups[1]}: p={p2:.4f} {'Normal' if p2 > 0.05 else 'Not normal'}")

    # Step 3: Levene's test for homogeneity of variances
    levene_stat, levene_p = stats.levene(group1, group2)
    print(f"Levene's Test (homogeneity of variances): p={levene_p:.4f} {'Homogeneous' if levene_p > 0.05 else 'Not homogeneous'}")
    
    # Step 4: Chi-square test (equality of group sizes)
    observed = [len(group1), len(group2)]
    expected = [sum(observed) / 2] * 2
    chi2_stat, chi2_p = stats.chisquare(observed, f_exp=expected)
    print(f"Chi-square Test (equality of group sizes): p={chi2_p:.4f} {'Equal sizes' if chi2_p > 0.05 else 'Unequal sizes'}")

    # Check if all assumptions are satisfied
    if p1 > 0.05 and p2 > 0.05 and levene_p > 0.05 and chi2_p > 0.05:
        # Step 5: Perform the independent t-test
        equal_var = levene_p > 0.05  # Use Levene's test result to determine equal variance assumption
        t_stat, t_p = stats.ttest_ind(group1, group2, equal_var=equal_var)
        print("\nIndependent t-test results:")
        print(type)
        print(tabulate(
            [[f"{t_stat:.10f}", f"{t_p:.20f}", "Significant difference" if t_p < 0.05 else "No significant difference"]],
            headers=["t-statistic", "p-value", "Conclusion"],
            tablefmt="grid"
        ))

        return {
            'group1_n': int(len(group1)),
            'group2_n': int(len(group2)),
            'shapiro_p_group1': round(float(p1), 4),
            'shapiro_p_group2': round(float(p2), 4),
            'levene_p': round(float(levene_p), 4),
            'chi2_p': round(float(chi2_p), 4),
            't_stat': round(float(t_stat), 4),
            't_p': round(float(t_p), 4),
            'equal_var_assumed': equal_var
        }
    else:
        print("\nT-test cannot be performed as one or more assumptions are not satisfied.")
        return None
    

def anova_test(df, dependent_col, independent_col):
    """
    df: dataframe z danymi
    dependent_col: zmienna zależna (ilościowa)
    independent_col: zmienna niezależna (więcej niż 2 grupy)
    """

    print(f"\n\nANOVA for {dependent_col} by {independent_col}:")
    # Step 1: Check the number of groups
    groups = df[independent_col].dropna().unique()
    if len(groups) < 2:
        print(f"The independent variable '{independent_col}' does not have enough groups for ANOVA.")
        return

    # preparing groups for ANOVA
    data_groups = []
    shapiro_ps = []
    print(f"Shapiro-Wilk Test (normality within groups):")
    for group in groups:
        group_data = df[df[independent_col] == group][dependent_col].dropna()
        if len(group_data) < 3:
            print(f"- {group}: Not enough data ({len(group_data)}) to test.")
            continue
        data_groups.append(group_data)
        stat, p = stats.shapiro(group_data)
        shapiro_ps.append(p)
        print(f"- {group}: p={p:.4f} {'Normal' if p > 0.05 else 'Not normal'}")

    # Step 2: Levene's test (homogeneity of variances)
    levene_stat, levene_p = stats.levene(*data_groups)
    print(f"Levene's Test (homogeneity of variances): p={levene_p:.4f} {'Homogeneous' if levene_p > 0.05 else 'Not homogeneous'}")
    # Perform ANOVA only if all assumptions are satisfied
    trouble_shapiro = False
    for p in shapiro_ps:
        if p < 0.05:
            trouble_shapiro = True
    if all([levene_p > 0.05]) and not trouble_shapiro:
        f_stat, p_value = stats.f_oneway(*data_groups)
        print(f"\nANOVA Test (one-way) results:")
        print(tabulate(
            [[f"{f_stat:.10f}", f"{p_value:.20f}", "Significant differences" if p_value < 0.05 else "No significant differences"]],
            headers=["F-statistic", "p-value", "Conclusion"],
            tablefmt="grid"
        ))

        return {
            'groups': list(groups),
            'levene_p': round(float(levene_p), 4),
            'anova_f': round(float(f_stat), 4),
            'anova_p': round(float(p_value), 4)
        }
    else:
        print("\nANOVA cannot be performed as one or more assumptions are not satisfied.")
        return None
    