"""
Модуль оценки качества моделей (Этап 5 Roadmap).

Реализует:
1. Расчет Precision, Recall, F1 для каждого класса (особенно для класса 2 «Аварийное»).
2. Macro F1 и Weighted F1.
3. Построение и сохранение матриц ошибок (Confusion Matrix).
4. Экспорт сводной таблицы сравнения бейзлайнов и CNN+LSTM в формате CSV и Markdown.
"""

from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score

CLASS_NAMES = ["0: Норма", "1: Преддефектное", "2: Аварийное"]


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    split_name: str = "Test",
) -> dict[str, float | str]:
    """Вычисляет детальные метрики классификации."""
    prec_per_class = precision_score(y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0)
    rec_per_class = recall_score(y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0)

    macro_f1 = f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=[0, 1, 2], average="weighted", zero_division=0)

    # Оценка для каждого класса
    c0_f1 = f1_per_class[0]
    c1_f1 = f1_per_class[1]
    c2_prec = prec_per_class[2]
    c2_rec = rec_per_class[2]
    c2_f1 = f1_per_class[2]

    return {
        "Model": model_name,
        "Split": split_name,
        "Precision_Class_0": round(float(prec_per_class[0]), 4),
        "Recall_Class_0": round(float(rec_per_class[0]), 4),
        "F1_Class_0": round(float(f1_per_class[0]), 4),
        "Precision_Class_1": round(float(prec_per_class[1]), 4),
        "Recall_Class_1": round(float(rec_per_class[1]), 4),
        "F1_Class_1": round(float(f1_per_class[1]), 4),
        "Precision_Class_2 (Аварийное)": round(float(c2_prec), 4),
        "Recall_Class_2 (Аварийное)": round(float(c2_rec), 4),
        "F1_Class_2 (Аварийное)": round(float(c2_f1), 4),
        "Macro_F1": round(float(macro_f1), 4),
        "Weighted_F1": round(float(weighted_f1), 4),
    }


def plot_confusion_matrices(
    results_dict: dict[str, tuple[np.ndarray, np.ndarray]],
    save_path: str | Path = "results/figures/confusion_matrices.png",
) -> None:
    """Строит сравнительные матрицы ошибок для нескольких моделей.

    Параметры:
        results_dict: словарь {model_name: (y_true, y_pred)}
    """
    n_models = len(results_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for ax, (name, (y_true, y_pred)) in zip(axes, results_dict.items()):
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            ax=ax,
            cbar=False,
        )
        ax.set_title(f"Матрица ошибок: {name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Предсказанный класс", fontsize=10)
        ax.set_ylabel("Истинный класс", fontsize=10)

    plt.tight_layout()
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(p, dpi=300)
    plt.close()
    print(f"[Evaluate] График матриц ошибок сохранен: {p.resolve()}")


def generate_comparison_table(
    metrics_list: list[dict[str, float | str]],
    output_dir: str | Path = "results/tables",
) -> pd.DataFrame:
    """Генерирует сводную таблицу сравнения моделей для научной статьи."""
    df_metrics = pd.DataFrame(metrics_list)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    csv_path = out / "comparison_metrics.csv"
    md_path = out / "comparison_metrics.md"

    df_metrics.to_csv(csv_path, index=False)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Сравнительная таблица эффективности моделей прогнозирования пучинообразования\n\n")
        try:
            f.write(df_metrics.to_markdown(index=False))
        except ImportError:
            # Ручное форматирование markdown таблицы если tabulate не установлен
            headers = list(df_metrics.columns)
            header_line = "| " + " | ".join(headers) + " |"
            sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
            data_lines = []
            for _, row in df_metrics.iterrows():
                row_str = "| " + " | ".join(str(val) for val in row) + " |"
                data_lines.append(row_str)
            f.write("\n".join([header_line, sep_line] + data_lines) + "\n")

    print(f"[Evaluate] Сводная таблица сохранена: {csv_path} и {md_path}")
    return df_metrics
