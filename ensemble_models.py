"""Эксперимент с ансамблем или полносвязной нейронной сетью."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from grader_contracts.ensembles import EnsembleInput, EnsembleResult


def run_ensemble_experiment(data: EnsembleInput) -> EnsembleResult:
    """Обучите выбранную модель и верните вероятности, метки и метрики на test.

    Разрешены Random Forest, Extra Trees, AdaBoost, Gradient Boosting,
    HistGradientBoosting, Voting/Stacking или полносвязная MLP. Не используйте
    test_target при обучении модели или подборе её параметров.
    """
    # Используем random_seed из контракта
    seed = data.random_seed

    X_train = data.train_features
    X_test = data.test_features
    y_train = data.train_target
    y_test = data.test_target

    # Приводим к numpy массивам (на случай если пришли pandas DataFrame)
    if hasattr(X_train, "values"):
        X_train = X_train.values
    if hasattr(X_test, "values"):
        X_test = X_test.values
    if hasattr(y_train, "values"):
        y_train = y_train.values
    if hasattr(y_test, "values"):
        y_test = y_test.values

    # МОДЕЛЬ: HistGradientBoostingClassifier
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", HistGradientBoostingClassifier(
            max_iter=300,
            max_depth=6,
            learning_rate=0.05,
            min_samples_leaf=20,
            l2_regularization=0.1,
            max_bins=255,
            random_state=seed,
        )),
    ])

    # Обучение ТОЛЬКО на train (test_target не используется!)
    model.fit(X_train, y_train)

    # Предсказания на test
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # Метрики
    roc_auc = float(roc_auc_score(y_test, y_proba))
    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))

    return EnsembleResult(
        model_name="HistGradientBoostingClassifier",
        probabilities=y_proba,
        predictions=y_pred,
        roc_auc=roc_auc,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
    )