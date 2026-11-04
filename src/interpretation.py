"""
Модуль интерпретации моделей с использованием SHAP (Этап 6 Roadmap).

Реализует:
1. Расчет Shapley значений с помощью shap.TreeExplainer для бейзлайнов.
2. Построение графика SHAP Summary Plot (Beeswarm) для ключевого класса 2 («Аварийное»).
3. Построение столбчатой диаграммы глобальной важности признаков.
4. Экспорт графиков в высоком разрешении для научной статьи, доказывающих,
   что именно циклы замерзания-оттаивания (F_cross) определяют риск пучинообразования.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def compute_shap_analysis(
    model: Any,
    X_sample: np.ndarray,
    feature_names: list[str],
    target_class: int = 2,
    save_dir: str | Path = "results/figures",
) -> np.ndarray:
    """Выполняет расчет SHAP-значений и строит графики для публикации."""
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[SHAP] Расчет значений SHAP для класса {target_class} («Аварийное»)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Обработка разных форматов возврата SHAP для мультиклассовой классификации:
    # Вариант 1: список массивов [class_0, class_1, class_2]
    # Вариант 2: 3D-массив (n_samples, n_features, n_classes)
    if isinstance(shap_values, list):
        shap_target = shap_values[target_class]
    elif len(shap_values.shape) == 3:
        shap_target = shap_values[:, :, target_class]
    else:
        shap_target = shap_values

    # Понятные русскоязычные названия признаков для графиков в статье
    russian_feature_labels = {
        "f_cross": "F_cross (циклы замерзания 30д)",
        "f_cross_decade": "F_cross_decade (циклы за 10д)",
        "cumulative_cycles": "Кумулятивные циклы замерзания (сезон)",
        "t_sum": "T_sum (сумма отр. темп.)",
        "t_avg": "T_avg (средняя темп. 30д)",
        "snow_avg": "Снежный покров (средн.)",
        "snow_max": "Снежный покров (макс.)",
        "moisture_avg": "Влажность грунта",
        "day_sin": "День года (sin)",
        "day_cos": "День года (cos)",
    }
    pretty_names = [russian_feature_labels.get(f, f) for f in feature_names]

    # 1. SHAP Beeswarm Summary Plot для класса 2
    plt.figure(figsize=(9, 6))
    shap.summary_plot(
        shap_target,
        features=X_sample,
        feature_names=pretty_names,
        show=False,
        plot_type="dot",
    )
    plt.title(
        f"SHAP-анализ важности признаков: Вклад в класс «Аварийное» (Y=2)",
        fontsize=12,
        fontweight="bold",
        pad=15,
    )
    plt.tight_layout()
    summary_path = out_dir / "shap_summary_class2.png"
    plt.savefig(summary_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SHAP] График Beeswarm сохранен: {summary_path.resolve()}")

    # 2. Bar Plot средней абсолютной важности
    mean_abs_shap = np.mean(np.abs(shap_target), axis=0)
    sorted_idx = np.argsort(mean_abs_shap)

    plt.figure(figsize=(9, 5))
    y_pos = np.arange(len(sorted_idx))
    plt.barh(y_pos, mean_abs_shap[sorted_idx], color="#d95f02", alpha=0.85, edgecolor="black")
    plt.yticks(y_pos, [pretty_names[i] for i in sorted_idx], fontsize=10)
    plt.xlabel("Среднее абсолютное значение SHAP (влияние на класс «Аварийное»)", fontsize=10)
    plt.title("Рейтинг физических признаков по влиянию на аварийность", fontsize=11, fontweight="bold")
    plt.grid(True, axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()
    bar_path = out_dir / "shap_feature_importance_bar.png"
    plt.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SHAP] График важности признаков сохранен: {bar_path.resolve()}")

    return shap_target
