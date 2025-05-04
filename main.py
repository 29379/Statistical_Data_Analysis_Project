from utils import (
    read_cancer_dataset, plot_dataset_distributions, adjust_column_to_pass_shapiro,
    generate_dependent_samples,adjust_for_ttest
)
from tests.parametric_tests import ttest_independent, anova
from tests.non_parametric_tests import wilcoxon_test, friedman_test
from exploring_dataset.descripting import descriptive_statistics

from tabulate import tabulate

def main():
    df = read_cancer_dataset()
    
    desc = descriptive_statistics(df)
    desc.to_excel('results/descriptive_statistics.xlsx', index=True)

    t_test_results = ttest_independent(df, dependent_col='age_at_diagnosis', independent_col='overall_survival_status')
    if t_test_results:
        print("\nFull t-test results:")
        print(tabulate(
            t_test_results.items(),
            headers=["Metric", "Value"],
            tablefmt="grid"
        ))
    
    anova_results = anova(df, dependent_col='age_at_diagnosis', independent_col='cellularity')
    if anova_results:
        print("\nFull ANOVA results:")
        print(tabulate(
            anova_results.items(),
            headers=["Metric", "Value"],
            tablefmt="grid"
        ))
    

if __name__ == "__main__":
    main()