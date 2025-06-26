from utils import (
    read_cancer_dataset
)
from tests.parametric_tests import ttest_independent, anova_test
from tests.non_parametric_tests import wilcoxon_test, friedman_test
from exploring_dataset.descripting import descriptive_statistics

from tabulate import tabulate
import pandas as pd
from utils import MODIFIED_FILE_NAME
from lifeline import survival_plot
from training import predict_survival


def main():
    fails_count = 0
    df = read_cancer_dataset()
    
    desc = descriptive_statistics(df)
    desc.to_excel('results/descriptive_statistics.xlsx', index=True)

    print("\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")

    t_test_results = ttest_independent(df, dependent_col='age_at_diagnosis', independent_col='overall_survival_status')
    if t_test_results is not None:
        print("\nFull t-test results:")
        print(tabulate(
            t_test_results.items(),
            headers=["Metric", "Value"],
            tablefmt="grid"
        ))
    else:
        print("Failed at t-Student")
        fails_count += 1
    
    print("\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")

    anova_results = anova_test(df, dependent_col='age_at_diagnosis', independent_col='cellularity')
    if anova_results is not None:
        print("\nFull ANOVA results:")
        print(tabulate(
            anova_results.items(),
            headers=["Metric", "Value"],
            tablefmt="grid"
        ))
    else:
        print("Failed at ANOVA")
        fails_count += 1

    print("\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")

    if fails_count == 0:
        df.to_csv(MODIFIED_FILE_NAME, sep='\t', index=False)
        print(f"Saved modified dataset to {MODIFIED_FILE_NAME}")

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

    print("\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")

    survival_plot(df)

    print("\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")

    predict_survival(df)
    

if __name__ == "__main__":
    main()