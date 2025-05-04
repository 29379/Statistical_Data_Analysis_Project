from utils import (
    read_cancer_dataset, plot_dataset_distributions, adjust_column_to_pass_shapiro,
    generate_dependent_samples,adjust_for_ttest
)
from tests.parametric_tests import ttest_independent, anova_test
from tests.non_parametric_tests import wilcoxon_test, friedman_test
from exploring_dataset.descripting import descriptive_statistics

from tabulate import tabulate
import pandas as pd

def main():
    df = read_cancer_dataset()
    
    desc = descriptive_statistics(df)
    desc.to_excel('results/descriptive_statistics.xlsx', index=True)

    print("\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")

    t_test_results = ttest_independent(df, dependent_col='age_at_diagnosis', independent_col='overall_survival_status')
    if t_test_results:
        print("\nFull t-test results:")
        print(tabulate(
            t_test_results.items(),
            headers=["Metric", "Value"],
            tablefmt="grid"
        ))
    
    print("\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")

    anova_results = anova_test(df, dependent_col='age_at_diagnosis', independent_col='cellularity')
    if anova_results:
        print("\nFull ANOVA results:")
        print(tabulate(
            anova_results.items(),
            headers=["Metric", "Value"],
            tablefmt="grid"
        ))

    print("\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")

    wilcoxon_results = wilcoxon_test(
        df,
        subject_col='patient_id',
        dependent_col='tumor_size',
        timepoint_col='timepoint',
        timepoint1='baseline',
        timepoint2='month_6'
    )
    if wilcoxon_results:
        print("\nWilcoxon results:")
        print(tabulate(wilcoxon_results.items(), headers=["Metric", "Value"], tablefmt="grid"))

    print("\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")

    friedman_results = friedman_test(
        df,
        subject_col='patient_id',
        dependent_col='tumor_size',
        timepoint_col='timepoint',
        timepoints=['baseline', 'month_6', 'month_12']
    )
    if friedman_results:
        print("\nFriedman results:")
        scalar_results = {k: round(v, 6) if isinstance(v, float) else v for k, v in friedman_results.items() if not isinstance(v, pd.DataFrame)}
        print(tabulate(scalar_results.items(), headers=["Metric", "Value"], tablefmt="grid"))

    

if __name__ == "__main__":
    main()