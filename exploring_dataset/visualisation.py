import matplotlib.pyplot as plt
import seaborn as sns


def plot_visualizations(long_df, posthoc_result, dependent_col):
    # Visualization 1: Boxplot for visualizing the data distribution at each timepoint
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=long_df, x='timepoint', y='value', hue='timepoint', palette='Set2', legend=False)
    plt.title(f'{dependent_col} distribution across timepoints')
    plt.xlabel('Timepoint')
    plt.ylabel(dependent_col)
    plt.show()

    # Visualization 2: Heatmap for the pairwise comparison p-values
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(posthoc_result, annot=True, cmap='coolwarm', fmt='.4f', cbar_kws={'label': 'p-value'})
    # plt.title('Pairwise Comparison p-values (Nemenyi)')
    # plt.xlabel('Timepoints')
    # plt.ylabel('Timepoints')
    # plt.show()
