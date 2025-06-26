from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_curve, roc_auc_score,
    accuracy_score, precision_score, recall_score,
    confusion_matrix
)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def compute_metrics(y_test, y_pred, y_probs):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    auc = roc_auc_score(y_test, y_probs)
    print(f"\nAccuracy     : {accuracy:.3f}")
    print(f"Precision    : {precision:.3f}")
    print(f"Recall       : {recall:.3f}")
    print(f"Specificity  : {specificity:.3f}")
    print(f"AUC (ROC)    : {auc:.3f}")
    return accuracy, precision, recall, specificity, auc


def predict_survival(df):
    df['event'] = df['overall_survival_status'].map(lambda x: 1 if 'DECEASED' in x else 0)

    numeric_features = ['age_at_diagnosis', 'tumor_size', 'nottingham_prognostic_index',
                 'relapse_free_status_months', 'tumor_stage', 'lymph_nodes_examined_positive',
                 'cohort', 'mutation_count']
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    # categorical_transformer = Pipeline(steps=[
    #     ('imputer', SimpleImputer(strategy='most_frequent')),
    #     ('encoder', OneHotEncoder(drop='first'))
    # ])
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        # ('cat', categorical_transformer, categorical_features)
    ])
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000))
    ])
    X = df[numeric_features] # + categorical_features
    y = df['event']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    pipeline.fit(X_train, y_train)

    y_probs = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)

    accuracy, precision, recall, specificity, auc_score = compute_metrics(y_test, y_pred, y_probs)
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)

    plt.figure(figsize=(8, 5))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc_score:.2f})", color='blue')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
