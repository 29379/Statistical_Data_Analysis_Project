import pandas as pd
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
from copy import deepcopy

def survival_plot(dataframe):
    df = deepcopy(dataframe)
    df['event'] = df['overall_survival_status'].map(lambda x: 1 if 'DECEASED' in str(x) else 0)
    df['duration'] = df['overall_survival_months']

    kmf = KaplanMeierFitter()

    kmf.fit(durations=df['duration'], event_observed=df['event'])

    plt.figure(figsize=(8, 5))
    kmf.plot(ci_show=True)
    plt.xlabel("Czas przeżycia (miesiące)")
    plt.ylabel("Prawdopodobieństwo przeżycia")
    plt.grid(True)
    plt.show()
