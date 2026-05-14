"""
Supply Chain Analysis — Part 3: Model Training & Evaluation


"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from collections import Counter
from warnings import filterwarnings

from sklearn.ensemble          import RandomForestClassifier
from sklearn.model_selection   import train_test_split, cross_val_score
from sklearn.metrics           import (classification_report,
                                       precision_score, recall_score,
                                       accuracy_score, f1_score,
                                       confusion_matrix, roc_auc_score,
                                       roc_curve)
from imblearn.over_sampling    import SMOTE

filterwarnings('ignore')



#  Load Data 

CLEAN_PATH = r'C:\Supply chain analysis\Data\clean_supply_chain.csv'

df = pd.read_csv(CLEAN_PATH)
print(f"Loaded: {df.shape[0]:,} rows  {df.shape[1]} columns")


# 1.  FEATURE SELECTION


FEATURE_COLS = [
    'Type',
    'Days for shipment (scheduled)',
    'Category Name',
    'Customer Segment',
    'Shipping Mode',
    'Order Status',
    'Department Name',
    'Order Region',
    'order_month',
    'order_day',
    'order_hour',
]

TARGET_COL = 'Late_delivery_risk'

X = df[FEATURE_COLS].copy()
y = df[TARGET_COL].copy()

print(f"\nTarget distribution:\n{y.value_counts()}")


# 2.  FREQUENCY ENCODING FOR CATEGORICAL COLUMNS


cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
print(f"\nCategorical columns to encode: {cat_cols}")

for col in cat_cols:
    freq_map      = X[col].value_counts(normalize=True)
    X[f'{col}_freq'] = X[col].map(freq_map)

X_encoded = X.drop(columns=cat_cols)
print(f"Encoded feature matrix shape: {X_encoded.shape}")
print(f"Features: {X_encoded.columns.tolist()}")


# 3.  TRAIN / TEST SPLIT


X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nTrain size : {X_train.shape[0]:,}")
print(f"Test  size : {X_test.shape[0]:,}")


# 4.  HANDLE CLASS IMBALANCE WITH SMOTE


print(f"\nClass distribution before SMOTE: {Counter(y_train)}")

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print(f"Class distribution after  SMOTE: {Counter(y_train_sm)}")


# 5.  MODEL TRAINING


rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)
rf_model.fit(X_train_sm, y_train_sm)
print("\nRandom Forest training complete.")

# 6.  MODEL EVALUATION


def evaluate_model(model, X_test, y_test):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n" + "="*55)
    print("  MODEL EVALUATION — Random Forest Classifier")
    print("="*55)
    print(classification_report(y_test, y_pred))
    print(f"  Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision : {precision_score(y_test, y_pred):.4f}")
    print(f"  Recall    : {recall_score(y_test, y_pred):.4f}")
    print(f"  F1 Score  : {f1_score(y_test, y_pred):.4f}")
    print(f"  ROC-AUC   : {roc_auc_score(y_test, y_proba):.4f}")

    return y_pred, y_proba

y_pred, y_proba = evaluate_model(rf_model, X_test, y_test)

# 5-fold cross-validation
cv_scores = cross_val_score(rf_model, X_encoded, y, cv=5, scoring='f1')
print(f"\n5-Fold CV F1 Scores : {cv_scores.round(4)}")
print(f"Mean CV F1          : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")




