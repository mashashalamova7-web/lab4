"""
run_experiments.py — запуск экспериментов для отчёта.
Этот файл НЕ используется градером.
"""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Загрузка данных
df = pd.read_csv("german.csv", sep=";")
X = df.drop(columns=["Creditability"])
y = df["Creditability"]

if set(y.unique()) == {1, 2}:
    y = y.map({1: 0, 2: 1})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
)

preprocessor = Pipeline([("scaler", StandardScaler())])

results = {}

models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=5, random_state=RANDOM_SEED
    ),
    "ExtraTrees": ExtraTreesClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=5, random_state=RANDOM_SEED
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        min_samples_leaf=10, random_state=RANDOM_SEED
    ),
    "HistGradientBoosting": HistGradientBoostingClassifier(
        max_iter=300, max_depth=6, learning_rate=0.05,
        min_samples_leaf=20, l2_regularization=0.1,
        max_bins=255, random_state=RANDOM_SEED
    ),
    "AdaBoost": AdaBoostClassifier(
        n_estimators=100, learning_rate=0.1, random_state=RANDOM_SEED
    ),
    "MLPClassifier": MLPClassifier(
        hidden_layer_sizes=(128, 64, 32), activation="relu",
        solver="adam", alpha=0.001, max_iter=500,
        random_state=RANDOM_SEED, early_stopping=True
    ),
    "Stacking": StackingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=100, max_depth=8, random_state=RANDOM_SEED)),
            ("gb", GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=RANDOM_SEED)),
            ("et", ExtraTreesClassifier(n_estimators=100, max_depth=8, random_state=RANDOM_SEED)),
        ],
        final_estimator=LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
        cv=5,
    ),
}

for name, clf in models.items():
    pipe = Pipeline([("pre", preprocessor), ("clf", clf)])
    pipe.fit(X_train, y_train)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = pipe.predict(X_test)

    results[name] = {
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
    }
    print(f"{name:25s} | ROC-AUC: {results[name]['roc_auc']:.4f}")

# Сохранение JSON
with open("experiment_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\nРезультаты сохранены в experiment_results.json")
print("\nТаблица:")
print(pd.DataFrame(results).T.to_string())

best = max(results, key=lambda k: results[k]["roc_auc"])
print(f"\nЛучшая модель: {best} (ROC-AUC = {results[best]['roc_auc']})")