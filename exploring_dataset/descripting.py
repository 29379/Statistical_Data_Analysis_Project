import pandas as pd

def descriptive_statistics(df):
    numerical_statistics = {}
    numerical_variables = df.select_dtypes(include=['number']).columns
    for col in numerical_variables:
        numerical_statistics[col] = {
            'type': 'ilościowa (numeryczna)',
            'count': round(df[col].count(), 2),
            'mean': round(df[col].mean(), 2),
            'std': round(df[col].std(), 2),
            'min': round(df[col].min(), 2),
            '25%': round(df[col].quantile(0.25), 2),
            '50% (median)': round(df[col].median(), 2),
            '75%': round(df[col].quantile(0.75), 2),
            'max': round(df[col].max(), 2),
            'skewness': round(df[col].skew(), 2),
            'kurtosis': round(df[col].kurtosis(), 2)
        }

    categorical_statistics = {}
    categorical_values = df.select_dtypes(include=['object', 'category', 'bool']).columns
    for col in categorical_values:
        categorical_statistics[col] = {
            'type': 'kategoryczna (nominalna/porządkowa)',
            'count': round(df[col].count(), 2), 
            'unique': round(df[col].nunique(), 2),  
            'top (najczęstsza)': df[col].mode()[0] if not df[col].mode().empty else None,
            'freq (liczba najczęstszej)': round(df[col].value_counts().iloc[0], 2) if not df[col].value_counts().empty else None,
            'value_counts': {k: round(v, 2) for k, v in df[col].value_counts().to_dict().items()} 
        }
    
    print("Numerical Statistics:")
    print(pd.DataFrame(numerical_statistics).T.to_markdown())
    print("\n\nCategorical Statistics:")
    print(pd.DataFrame(categorical_statistics).T.drop(columns=['value_counts']).to_markdown())
    
    combined_statistics = {**numerical_statistics, **categorical_statistics}
    return pd.DataFrame(combined_statistics).T
