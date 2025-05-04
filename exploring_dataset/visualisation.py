import matplotlib.pyplot as plt
import seaborn as sns


def plot_visualizations(long_df, bonferroni_dunn_result, dependent_col):
    """
    This function handles the visualizations for the Friedman test and Bonferroni-Dunn post-hoc results.
    
    Parameters:
    - long_df: DataFrame containing the long format data for plotting
    - bonferroni_dunn_result: DataFrame containing the Bonferroni-Dunn post-hoc p-values
    - dependent_col: the dependent variable being analyzed (e.g., 'tumor_size')
    """
    # Visualization 1: Boxplot for visualizing the data distribution at each timepoint
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=long_df, x='timepoint', y='value', hue='timepoint', palette='Set2', legend=False)
    plt.title(f'{dependent_col} distribution across timepoints')
    plt.xlabel('Timepoint')
    plt.ylabel(dependent_col)
    plt.show()

    # Visualization 2: Heatmap for the pairwise comparison p-values
    plt.figure(figsize=(8, 6))
    sns.heatmap(bonferroni_dunn_result, annot=True, cmap='coolwarm', fmt='.4f', cbar_kws={'label': 'p-value'})
    plt.title('Pairwise Comparison p-values (Bonferroni-Dunn)')
    plt.xlabel('Timepoints')
    plt.ylabel('Timepoints')
    plt.show()

