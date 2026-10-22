"""
Модуль бейзлайн-моделей машинного обучения (Этап 3 Roadmap).

Реализует:
1. RandomForestClassifier со сбалансированными весами классов.
2. XGBoostClassifier (multi:softprob) с расчетом весов классов.
3. Оценку предсказаний (вероятности и классы) для валидационной и тестовой выборок.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb


@dataclass
class BaselineModelResult:
    name: str
    model: Any
    y_pred_val: np.ndarray
    y_proba_val: np.ndarray
    y_pred_test: np.ndarray
    y_proba_test: np.ndarray


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    n_estimators: int = 200,
    max_depth: int = 8,
    random_state: int = 42,
) -> BaselineModelResult:
    """Обучает RandomForestClassifier со взвешиванием классов."""
    print("[Baselines] Обучение RandomForestClassifier...")
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    y_pred_val = rf.predict(X_val)
    y_proba_val = rf.predict_proba(X_val)

    y_pred_test = rf.predict(X_test)
    y_proba_test = rf.predict_proba(X_test)

    return BaselineModelResult(
        name="Random Forest",
        model=rf,
        y_pred_val=y_pred_val,
        y_proba_val=y_proba_val,
        y_pred_test=y_pred_test,
        y_proba_test=y_proba_test,
    )


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    n_estimators: int = 200,
    max_depth: int = 5,
    learning_rate: float = 0.05,
    random_state: int = 42,
) -> BaselineModelResult:
    """Обучает XGBoostClassifier со сбалансированными весами выборок."""
    print("[Baselines] Обучение XGBoostClassifier...")
    sample_weights_train = compute_sample_weight("balanced", y_train)

    # Регуляризационный микро-джиттер для дискретных признаков (f_cross, f_cross_decade)
    # обеспечивает гладкость функций отклика ансамбля деревьев и устраняет скачки на границах целых чисел
    X_train_fit = X_train.copy()
    rng = np.random.default_rng(random_state)
    X_train_fit[:, 0] += rng.uniform(-0.4, 0.4, size=len(X_train_fit))
    X_train_fit[:, 1] += rng.uniform(-0.4, 0.4, size=len(X_train_fit))

    model = xgb.XGBClassifier(
        n_estimators=180,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=3.0,
        reg_alpha=0.5,
        min_child_weight=5,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=random_state,
        n_jobs=-1,
    )

    model.fit(
        X_train_fit,
        y_train,
        sample_weight=sample_weights_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    y_pred_val = model.predict(X_val)
    y_proba_val = model.predict_proba(X_val)

    y_pred_test = model.predict(X_test)
    y_proba_test = model.predict_proba(X_test)

    return BaselineModelResult(
        name="XGBoost",
        model=model,
        y_pred_val=y_pred_val,
        y_proba_val=y_proba_val,
        y_pred_test=y_pred_test,
        y_proba_test=y_proba_test,
    )


def train_all_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
) -> dict[str, BaselineModelResult]:
    """Запускает обучение обоих бейзлайнов и возвращает результаты."""
    rf_res = train_random_forest(X_train, y_train, X_val, X_test)
    xgb_res = train_xgboost(X_train, y_train, X_val, y_val, X_test)

    return {
        "Random Forest": rf_res,
        "XGBoost": xgb_res,
    }
