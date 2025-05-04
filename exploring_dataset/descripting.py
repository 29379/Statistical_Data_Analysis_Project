import pandas as pd

def descriptive_statistics(df):
    opis = {}

    numerical_statistics = {}
    numerical_variables = df.select_dtypes(include=['number']).columns
    for col in numerical_variables:
        numerical_statistics[col] = {
            'typ': 'ilościowa (numeryczna)',
            'count': df[col].count(),
            'mean': df[col].mean(),
            'std': df[col].std(),
            'min': df[col].min(),
            '25%': df[col].quantile(0.25),
            '50% (median)': df[col].median(),
            '75%': df[col].quantile(0.75),
            'max': df[col].max(),
            'skewness': df[col].skew(),
            'kurtosis': df[col].kurtosis()
        }
    
    categorical_statistics = {}
    categorical_values = df.select_dtypes(include=['object', 'category', 'bool']).columns
    for col in categorical_values:
        categorical_statistics[col] = {
            'typ': 'kategoryczna (nominalna/porządkowa)',
            'count': df[col].count(),
            'unique': df[col].nunique(),
            'top (najczęstsza)': df[col].mode()[0] if not df[col].mode().empty else None,
            'freq (liczba najczęstszej)': df[col].value_counts().iloc[0] if not df[col].value_counts().empty else None,
            'value_counts': df[col].value_counts().to_dict()
        }
    
    print("Numerical Statistics:")
    print(pd.DataFrame(numerical_statistics).T.to_markdown())
    print("\n\nCategorical Statistics:")
    print(pd.DataFrame(categorical_statistics).T.drop(columns=['value_counts']).to_markdown())
    
    combined_statistics = {**numerical_statistics, **categorical_statistics}
    return pd.DataFrame(combined_statistics).T