"""
Модуль расчета практического физического порога тревоги (Этап 7 Roadmap).

Реализует:
1. Анализ зависимости вероятности аварийного состояния от количества переходов через 0 °C (F_cross / F_cross_decade).
2. Вывод и формулировку инженерного «Критерия тревоги» (порога срабатывания предиктивной системы).
3. Построение калибровочной кривой порога F_cross vs P(Y=2).
4. Построение графика прогнозной вероятности риска во времени для зимне-весеннего сезона (Подраздел 5.3 статьи).
"""

from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compute_alarm_threshold(
    df_eval: pd.DataFrame,
    fcross_col: str = "f_cross_decade",
    proba_col: str = "proba_defect",
    target_col: str = "target_y",
    alarm_prob_threshold: float = 0.50,
    save_path: str | Path = "results/figures/threshold_alarm_curve.png",
) -> dict[str, float | int | str]:
    """Анализирует зависимость риска от F_cross и вычисляет критический порог тревоги."""
    work_df = df_eval.dropna(subset=[fcross_col, proba_col]).copy()
    work_df[fcross_col] = work_df[fcross_col].round().astype(int)

    is_defect_mode = ("defect" in proba_col) or ("class_1" in proba_col)
    actual_calc = (lambda s: (s >= 1).mean()) if is_defect_mode else (lambda s: (s == 2).mean())

    # Группировка по числу переходов
    stats = (
        work_df.groupby(fcross_col)
        .agg(
            count=(proba_col, "count"),
            mean_pred_risk=(proba_col, "mean"),
            actual_risk=(target_col, actual_calc),
        )
        .reset_index()
    )

    # Поиск порога срабатывания: наименьшее значение F_cross, где риск превышает порог тревоги
    alarm_rows = stats[stats["mean_pred_risk"] >= alarm_prob_threshold]
    if not alarm_rows.empty:
        critical_fcross = int(alarm_rows.iloc[0][fcross_col])
        empirical_risk = float(alarm_rows.iloc[0]["actual_risk"])
        predicted_risk = float(alarm_rows.iloc[0]["mean_pred_risk"])
    else:
        # Если порог 50% не достигнут, берем точку максимального риска
        max_row = stats.loc[stats["mean_pred_risk"].idxmax()]
        critical_fcross = int(max_row[fcross_col])
        empirical_risk = float(max_row["actual_risk"])
        predicted_risk = float(max_row["mean_pred_risk"])

    # Построение графика
    fig, ax1 = plt.subplots(figsize=(8.5, 5))

    color_pred = "#d95f02"
    color_actual = "#1f78b4"

    label_pred = "Модельная вероятность деформаций (%)" if is_defect_mode else "Модельная вероятность аварийности (%)"
    label_act = "Фактическая доля дефектов (%)" if is_defect_mode else "Фактическая доля аварий (%)"

    ax1.plot(
        stats[fcross_col],
        stats["mean_pred_risk"] * 100,
        marker="o",
        linewidth=2.5,
        color=color_pred,
        label=label_pred,
    )
    ax1.plot(
        stats[fcross_col],
        stats["actual_risk"] * 100,
        marker="s",
        linestyle="--",
        linewidth=2,
        color=color_actual,
        label=label_act,
    )

    # Линия тревоги
    ax1.axhline(
        alarm_prob_threshold * 100,
        color="red",
        linestyle=":",
        linewidth=2,
        label=f"Порог тревоги ({int(alarm_prob_threshold * 100)}%)",
    )
    ax1.axvline(
        critical_fcross,
        color="darkred",
        linestyle="-.",
        linewidth=1.8,
        label=f"Критический порог: F_cross ≥ {critical_fcross}",
    )

    chart_title = (
        "Зависимость риска пучинообразования от циклов замерзания-оттаивания"
        if is_defect_mode
        else "Зависимость риска аварийного состояния от циклов замерзания-оттаивания"
    )
    ax1.set_title(chart_title, fontsize=12, fontweight="bold", pad=12)
    ax1.set_xlabel(f"Количество переходов температуры через 0 °C за декаду ({fcross_col})", fontsize=10)
    ax1.set_ylabel("Вероятность пучинообразования (%)", fontsize=10)
    ax1.set_ylim(-5, 105)
    ax1.set_xticks(stats[fcross_col].values)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left", frameon=True)

    plt.tight_layout()
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(p, dpi=300)
    plt.close()
    print(f"[Threshold] Кривая порога тревоги сохранена: {p.resolve()}")

    recommendation = (
        f"При количестве переходов через 0 °C более {critical_fcross} раз за декаду "
        f"прогнозируемая вероятность образования пучины превышает {int(predicted_risk * 100)}% "
        f"(фактическая частота дефектов: {int(empirical_risk * 100)}%)."
    )

    return {
        "feature_name": fcross_col,
        "critical_threshold_cycles": critical_fcross,
        "predicted_risk_percent": round(predicted_risk * 100, 1),
        "actual_risk_percent": round(empirical_risk * 100, 1),
        "recommendation_text": recommendation,
    }


def plot_seasonal_risk_timeline(
    df_eval: pd.DataFrame,
    start_date: str = "2023-10-01",
    end_date: str = "2024-05-31",
    proba_col: str = "proba_class_2",
    save_path: str | Path = "results/figures/seasonal_risk_timeline.png",
) -> None:
    """Строит динамику риска деформации дорожного полотна за зимне-весенний период."""
    work_df = df_eval.copy()
    work_df["date"] = pd.to_datetime(work_df["date"])

    season_df = work_df[
        (work_df["date"] >= pd.to_datetime(start_date)) & (work_df["date"] <= pd.to_datetime(end_date))
    ].sort_values("date")

    if season_df.empty:
        # Если диапазон вне тестового набора, берем последние 200 дней
        season_df = work_df.iloc[-220:].sort_values("date")

    fig, (ax_temp, ax_risk) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # 1. График температуры и циклов перехода
    ax_temp.plot(season_df["date"], season_df["t_air"], color="royalblue", lw=1.5, label="Температура воздуха T_air (°C)")
    ax_temp.axhline(0, color="gray", linestyle="--", lw=1, alpha=0.7)
    ax_temp.fill_between(
        season_df["date"],
        season_df["t_air"],
        0,
        where=(season_df["t_air"] <= 0),
        color="skyblue",
        alpha=0.35,
        label="Область отрицательных температур",
    )
    ax_temp.set_ylabel("T, °C", fontsize=10)
    ax_temp.set_title("Динамика метеоусловий и прогнозируемого риска пучинообразования", fontsize=12, fontweight="bold")
    ax_temp.grid(True, linestyle=":", alpha=0.6)
    ax_temp.legend(loc="lower left")

    # 2. График прогнозного риска
    if proba_col in season_df.columns:
        risk_vals = season_df[proba_col] * 100
    else:
        risk_vals = np.zeros(len(season_df))

    ax_risk.plot(season_df["date"], risk_vals, color="#e41a1c", lw=2.5, label="Вероятность класса «Аварийное» (P(Y=2))")
    ax_risk.axhline(50, color="darkred", linestyle=":", lw=1.8, label="Порог тревоги (50%)")
    ax_risk.set_ylabel("Риск, %", fontsize=10)
    ax_risk.set_xlabel("Дата", fontsize=10)
    ax_risk.set_ylim(-5, 105)
    ax_risk.grid(True, linestyle=":", alpha=0.6)
    ax_risk.legend(loc="upper left")

    plt.tight_layout()
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(p, dpi=300)
    plt.close()
    print(f"[Threshold] График сезонной динамики риска сохранен: {p.resolve()}")
