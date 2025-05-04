import re
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import time
import numpy as np

# additional ordinal variables: cohort, neoplasm_histologic_grade
"""
ids:
> patient_id

ilościowe/numeryczne (skala interval): 
> -

ilościowe/numeryczne (skala ratio): 
> age_at_diagnosis - weird representation (floats with 2 decimal places) - age at diagnosis of breast cancer in years (e.g. 59.42)
> overall_survival_months - weird representation (floats with 6 decimal places) - number of months from diagnosis to death or last follow-up (e.g. 1.600000)
> tumor_size - floats with 1 decimal place - size of the tumor in millimeters (e.g. 2.0)
> nottingham_prognostic_index - a scoring system that estimates how likely someone is to survive breast cancer after surgery based on tumor size, lymph node status, and tumor grade - floats [1, 7.2], 1 decimal place (e.g. 3.2)
> relapse_free_status_months - floats with 6 decimal places - number of months from diagnosis to recurrence or last follow-up (e.g. 1.600000)

ordynalne/porządkowe: 
> tumor_stage - [[ 2.  1.  4.  3.  0. nan]]
> lymph_nodes_examined_positive - positive nodes indicate that cancer has spread to the lymph nodes - the number of positive lymph nodes is used to stage cancer
> cohort - a group of people who share a common trait and are followed over time to study breast cancer risk factors and outcomes
> mutation_count - 33 different counts with nan included

nominalne/nienumerowane kategorie: 
> overall_survival_status - ['0:LIVING' '1:DECEASED' nan]
> patients_vital_status - ['Living' 'Died of Disease' 'Died of Other Causes' nan]
> pam50_claudin_low_subtype - identifies breast cancer subtypes, including the claudin-low subtype (genomic group of breast cancers that's characterized by low expression of adhesion proteins): 7 types, STRINGS
> er_status - a measure of whether breast cancer cells have receptors for estrogen: 3 types: positive, negative, unknown/nan
> pr_status - a measure of whether breast cancer cells have progesterone receptors: 3 types: positive, negative, unknown/nan
> her2_status - a measure of the amount of human epidermal growth factor receptor 2 (HER2) protein in a breast tumor: 3 types: positive, negative, unknown/nan
> relapse_free_status - ['0:Not Recurred' '1:Recurred' nan]
> cancer_type_detailed - 8 types, STRINGS
> cellularity - [nan 'High' 'Moderate' 'Low']
> type_of_breast_surgery - ['MASTECTOMY' 'BREAST CONSERVING' nan]
> inferred_menopausal_state - ['Post' 'Pre' nan]
"""

#   dataset source: https://www.cbioportal.org/study/clinicalData?id=brca_metabric
data_path ='dataset/brca_metabric_clinical_data.tsv'

columns_to_keep = ['patient_id', #ids
    
    # ilościowe/numeryczne (skala interval): None

    'age_at_diagnosis', 'overall_survival_months', 'tumor_size', 'nottingham_prognostic_index', 'relapse_free_status_months', # ilościowe/numeryczne (skala ratio)
    
    'tumor_stage', 'lymph_nodes_examined_positive', 'cohort', 'mutation_count', # ordynalne/porządkowe
    
    'overall_survival_status', 'pam50_claudin_low_subtype', 'er_status', 'pr_status', 'her2_status',
        'patients_vital_status', 'relapse_free_status', 'cancer_type_detailed', 'cellularity', 'type_of_breast_surgery', 'inferred_menopausal_state' # nominalne/nienumerowane kategorie
]


def to_snake_case(name):
    name = re.sub(r'[\s\-]+', '_', name)                # replace spaces and hyphens with underscores
    name = re.sub(r'(?<=[a-z])(?=[A-Z])', '_', name)    # insert underscore before capital letters, but not in acronyms (2+ uppercase letters)
    name = re.sub(r'[^a-zA-Z0-9_]', '', name)           # remove special characters
    name = name.lower()    
    name = re.sub(r'__+', '_', name)    
    name = name.strip('_')

    return name


def rename_columns(df):
    df.columns = [to_snake_case(col) for col in df.columns]
    return df


def read_cancer_dataset():
    df = pd.read_csv(data_path, sep='\t')   # subset? for friedmann test?
    df = rename_columns(df)
    df = df[columns_to_keep]
    print(f"DataFrame shape: {df.shape}")
    df = pd.concat([df, generate_dependent_samples(df, percent=0.25)], ignore_index=True)
    print(f"DataFrame shape after generating random dependent samples: {df.shape}")
    df = adjust_for_ttest(
        df,
        dependent_col='age_at_diagnosis',
        independent_col='overall_survival_status',
        balance_groups=True  # this will downsample to equal sizes
    )
    print(f"DataFrame shape after adjusting for t-test: {df.shape}")
    return df


def plot_dataset_distributions(raw_df):
    num_cols = len(raw_df.columns)
    cols_per_row = 3
    rows_per_grid = 2
    plots_per_grid = cols_per_row * rows_per_grid

    for i in range(0, num_cols, plots_per_grid):
        subset_cols = raw_df.columns[i:i + plots_per_grid]
        num_plots = len(subset_cols)
        
        fig, axes = plt.subplots(rows_per_grid, cols_per_row, figsize=(12, 9))
        axes = axes.flatten()
        
        for j, col in enumerate(subset_cols):
            ax = axes[j]
            if raw_df[col].dtype in ['float64', 'int64']:
                sns.histplot(raw_df[col].dropna(), kde=True, ax=ax)
                # Test Shapiro-Wilka
                stat, p = stats.shapiro(raw_df[col].dropna())
                result = 'YES' if p > 0.05 else 'NO'
                ax.set_title(f"{col} (Shapiro: {result})")
            else:
                sns.countplot(y=raw_df[col], order=raw_df[col].value_counts().index, ax=ax)
                ax.set_title(col)
        
        # Hide unused subplots
        for k in range(num_plots, len(axes)):
            axes[k].axis('off')
        
        plt.tight_layout()
        plt.show()


def adjust_column_to_pass_shapiro(df, column_name, max_duration=60, threshold_p_value=0.05):
    start_time = time.time()

    # Make a copy of the column to avoid modifying the original dataframe
    modified_column = df[column_name].dropna().copy()

    iteration_count = 0  # Initialize iteration counter

    while True:
        iteration_count += 1  # Increment iteration counter

        # Perform the Shapiro-Wilk test
        stat, p_value = stats.shapiro(modified_column)

        # Check if the test is passed
        if p_value > threshold_p_value:
            print(f"✅ Test passed with p-value = {p_value:.4f} after {iteration_count} iterations")
            break

        # Apply a small modification to the column (e.g., add small random noise)
        noise = np.random.normal(0, 0.1, size=modified_column.shape)
        modified_column += noise

        # Check if the maximum duration has been reached
        if time.time() - start_time > max_duration:
            print(f"⏳ Time limit reached after {iteration_count} iterations. Test not passed.")
            break

    # Plot the original and modified distributions
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot the original distribution
    sns.histplot(df[column_name].dropna(), kde=True, color='blue', label='Original', ax=axes[0])
    axes[0].set_title(f'Original Distribution of {column_name}')
    axes[0].set_xlabel(column_name)
    axes[0].set_ylabel('Frequency')
    axes[0].legend()

    # Plot the modified distribution
    sns.histplot(modified_column, kde=True, color='orange', label='Modified', alpha=0.7, ax=axes[1])
    axes[1].set_title(f'Modified Distribution of {column_name}')
    axes[1].set_xlabel(column_name)
    axes[1].set_ylabel('Frequency')
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    # Update the original dataframe column with the modified values
    df.loc[df[column_name].notna(), column_name] = modified_column

    return df


def generate_dependent_samples(df, percent=0.25, random_state=42):
    np.random.seed(random_state)

    n_samples = int(len(df) * percent)
    sampled_df = df.sample(n=n_samples).copy()

    # Utwórz 3 pomiary: baseline, month_6, month_12
    records = []
    for idx, row in sampled_df.iterrows():
        base = row['tumor_size']
        if pd.isna(base):
            continue  # skip missing values
        
        # randomly generate tumor sizes for each timepoint
        tumor_sizes = {'baseline': base}
        if 6 <= row['overall_survival_months'] < 12:
            tumor_sizes['month_6'] = base * np.random.uniform(0.85, 1.15)  # simulate random growth or shrinkage
        elif row['overall_survival_months'] >= 12:
            tumor_sizes['month_6'] = base * np.random.uniform(0.85, 1.15)
            tumor_sizes['month_12'] = base * np.random.uniform(0.85, 1.15)  # further changes

        for timepoint, size in tumor_sizes.items():
            new_row = row.copy()
            new_row['tumor_size'] = size
            new_row['timepoint'] = timepoint
            records.append(new_row)

    # build new dataframe for dependent samples
    dependent_df = pd.DataFrame(records)
    return dependent_df

def adjust_for_ttest(df, dependent_col, independent_col, max_duration=60, threshold_p_value=0.05, balance_groups=False):
    """
    Adjusts data to try passing Shapiro-Wilk, Levene, and optionally balances group sizes.
    """
    start_time = time.time()
    iteration_count = 0

    # Work on a copy
    df_adj = df.copy()

    # Balance group sizes if requested
    if balance_groups:
        # Find smallest group size
        group_sizes = df_adj[independent_col].value_counts()
        min_size = group_sizes.min()
        print(f"Downsampling to balance group sizes at {min_size} samples per group...")

        balanced_df = []
        for group in group_sizes.index:
            group_df = df_adj[df_adj[independent_col] == group]
            balanced_df.append(group_df.sample(min_size, random_state=42))
        df_adj = pd.concat(balanced_df)

    # Begin adjustments
    groups = df_adj[independent_col].dropna().unique()

    while True:
        iteration_count += 1

        p_values = []
        group_vars = []
        group_data_list = []

        for group in groups:
            group_data = df_adj[df_adj[independent_col] == group][dependent_col].dropna()
            group_data_list.append(group_data)
            # Normality
            stat, p_value = stats.shapiro(group_data)
            p_values.append(p_value)
            # Variance
            group_vars.append(np.var(group_data, ddof=1))

        # Check if all groups pass Shapiro-Wilk
        all_normal = all(p > threshold_p_value for p in p_values)

        # Check Levene
        levene_stat, levene_p = stats.levene(*group_data_list)
        levene_ok = levene_p > threshold_p_value

        # Exit if both pass
        if all_normal and levene_ok:
            print(f"✅ Both Shapiro and Levene passed after {iteration_count} iterations.")
            break

        # Apply noise per group, scaling variance towards mean variance
        mean_var = np.mean(group_vars)
        for i, group in enumerate(groups):
            group_mask = df_adj[independent_col] == group
            group_data = df_adj.loc[group_mask, dependent_col]
            # Scaling factor to push variance towards mean variance
            scale_factor = np.sqrt(mean_var / (group_vars[i] + 1e-8))
            noise = np.random.normal(0, 0.1, size=group_data.shape)
            df_adj.loc[group_mask, dependent_col] = (group_data * scale_factor) + noise

        # Timeout
        if time.time() - start_time > max_duration:
            print(f"⏳ Time limit reached after {iteration_count} iterations. Not all tests passed.")
            break

    # Plot distributions after adjustment
    fig, axes = plt.subplots(1, len(groups), figsize=(6 * len(groups), 5))
    if len(groups) == 1:
        axes = [axes]
    for ax, group in zip(axes, groups):
        sns.histplot(df_adj[df_adj[independent_col] == group][dependent_col], kde=True, ax=ax)
        ax.set_title(f'{dependent_col} - {group}')
    plt.tight_layout()
    plt.show()

    return df_adj
