"""
Модуль практического применения обученных моделей PuchinaAI (Inference & Decision Support).

Позволяет по текущим/историческим данным телеметрии дорожного полотна:
1. Выполнить автоматический расчет физических признаков (F_cross, T_sum, близость к 0 °C и т.д.).
2. Сформировать прогноз состояния полотна Y на горизонт (по умолчанию tau = 30 дней):
   - Класс 0: «Норма» (деформация < 1.5 см) — штатный режим;
   - Класс 1: «Преддефектное состояние» (1.5 <= деформация < 3.0 см) — требуется плановый осмотр;
   - Класс 2: «АВАРИЙНОЕ СОСТОЯНИЕ / ТРЕВОГА» (деформация >= 3.0 см) — риск разрушения покрытия по ГОСТ Р 50597-2017.
3. Вычислить вероятности каждого класса P(Y=0), P(Y=1), P(Y=2) и ожидаемую глубину деформации.
4. Выделить дни, на которых нейросеть сфокусировала внимание (ZeroCrossingAttention).
5. Построить график прогноза и сохранить детальный отчет (в единичном или пакетном режиме).
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from src.features import compute_physical_features, extract_sequential_windows
from src.models.hybrid_nn import ZeroCrossingAttention


CLASS_NAMES = {
    0: "Норма (Класс 0)",
    1: "Преддефект (Класс 1)",
    2: "АВАРИЯ / ТРЕВОГА (Класс 2)",
}

CLASS_DESCRIPTIONS = {
    0: "Штатная эксплуатация: колейность/пучение < 1.5 см, трещины < 1.5 м.",
    1: "Внимание: колейность 1.5–3.0 см. Рекомендуется мониторинг и плановый дорожный осмотр.",
    2: "КРИТИЧЕСКИЙ РИСК: деформация >= 3.0 см по ГОСТ Р 50597-2017. Необходимы ограничения или ремонт.",
}

# Средние эмпирические деформации рабочего слоя по классам (см)
CLASS_HEAVE_ESTIMATE = {0: 0.6, 1: 2.1, 2: 3.8}


@dataclass
class PredictionReport:
    """Результат прогноза риска деформаций."""
    forecast_date: pd.Timestamp
    target_date: pd.Timestamp
    horizon_days: int
    predicted_class: int
    predicted_label: str
    probabilities: dict[str, float]
    expected_heave_cm: float
    recommendation: str
    model_name: str
    attention_weights: np.ndarray | None
    df_forecast: pd.DataFrame


def load_model_for_inference(
    model_type: Literal["hybrid_nn", "xgboost", "rf"] = "hybrid_nn",
    horizon_days: int = 30,
    models_dir: str | Path = "saved_models",
) -> tuple[any, any | None]:
    """Загружает модель и вспомогательный экстрактор внимания (при наличии)."""
    m_dir = Path(models_dir)
    if model_type == "hybrid_nn":
        model_path = m_dir / f"hybrid_nn_{horizon_days}d.keras"
        if not model_path.exists():
            raise FileNotFoundError(f"Файл нейросети не найден: {model_path.resolve()}")
        model = tf.keras.models.load_model(model_path)
        # Экстрактор внимания
        try:
            att_layer = model.get_layer("zero_crossing_attention")
            att_model = tf.keras.Model(inputs=model.input, outputs=att_layer.output[1])
        except Exception:
            att_model = None
        return model, att_model

    elif model_type == "xgboost":
        model_path = m_dir / f"xgb_model_{horizon_days}d.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Файл XGBoost не найден: {model_path.resolve()}")
        return joblib.load(model_path), None

    elif model_type in ["rf", "random_forest"]:
        model_path = m_dir / f"rf_model_{horizon_days}d.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Файл Random Forest не найден: {model_path.resolve()}")
        return joblib.load(model_path), None

    else:
        raise ValueError(f"Неизвестный тип модели: {model_type}. Доступны: 'hybrid_nn', 'xgboost', 'rf'")


def predict_future_risk(
    data: str | Path | pd.DataFrame,
    model_type: Literal["hybrid_nn", "xgboost", "rf"] = "hybrid_nn",
    horizon_days: int = 30,
    site_id: str | None = None,
    as_of_date: str | pd.Timestamp | None = None,
    models_dir: str | Path = "saved_models",
    save_plot_path: str | Path | None = "results/figures/latest_forecast.png",
) -> PredictionReport:
    """Выполняет расчет и прогноз риска деформаций полотна на horizon_days вперед.

    Параметры:
        data: путь к CSV файлу телеметрии или готовый DataFrame.
        model_type: используемая модель ('hybrid_nn', 'xgboost', 'rf').
        horizon_days: горизонт упреждения (по умолчанию 30 дней).
        site_id: идентификатор поста (если в файле несколько участков).
        as_of_date: дата, на которую строится прогноз (по умолчанию последняя доступная дата).
        models_dir: директория с весами моделей.
        save_plot_path: путь для сохранения наглядного графика прогноза.
    """
    if isinstance(data, (str, Path)):
        df_raw = pd.read_csv(data)
    else:
        df_raw = data.copy()

    df_raw["date"] = pd.to_datetime(df_raw["date"])

    # Фильтрация по посту
    if site_id is not None and "site_id" in df_raw.columns:
        df_site = df_raw[df_raw["site_id"] == site_id].copy()
        if df_site.empty:
            raise ValueError(f"Пост '{site_id}' не найден в данных. Доступные: {list(df_raw['site_id'].unique())}")
    elif "site_id" in df_raw.columns and df_raw["site_id"].nunique() > 1:
        first_site = df_raw["site_id"].iloc[0]
        print(f"[Inference] В данных несколько постов, выбран для прогноза: {first_site}")
        df_site = df_raw[df_raw["site_id"] == first_site].copy()
    else:
        df_site = df_raw.copy()

    df_site = df_site.sort_values("date").reset_index(drop=True)

    # Определение даты прогноза
    if as_of_date is not None:
        target_forecast_date = pd.to_datetime(as_of_date)
        df_site = df_site[df_site["date"] <= target_forecast_date].copy()
        if df_site.empty:
            raise ValueError(f"Нет данных до даты {as_of_date}")
    
    forecast_date = df_site["date"].max()
    future_date = forecast_date + pd.Timedelta(days=horizon_days)

    # 1. Расчет физических параметров
    feat_df = compute_physical_features(df_site, window_days=30, decade_days=10)

    # 2. Подготовка признаков в зависимости от типа модели
    model, att_model = load_model_for_inference(model_type, horizon_days=horizon_days, models_dir=models_dir)

    attention_vector = None

    if model_type == "hybrid_nn":
        seq_length = 45
        if len(feat_df) < seq_length:
            raise ValueError(
                f"Для нейросети требуется минимум {seq_length} последовательных дней истории, доступно: {len(feat_df)} дней."
            )

        X_seq, indices = extract_sequential_windows(feat_df, seq_length=seq_length)
        # Берем последнее окно, оканчивающееся на forecast_date
        last_window = X_seq[-1:]  # Shape: (1, seq_length, num_features)

        proba = model.predict(last_window, verbose=0)[0]
        pred_class = int(np.argmax(proba))

        if att_model is not None:
            raw_att = att_model.predict(last_window, verbose=0)[0]  # Shape: (seq_length, 1)
            attention_vector = raw_att.flatten()

    else:
        # Табличные модели (XGBoost, Random Forest)
        tabular_cols = [
            "f_cross",
            "f_cross_decade",
            "cumulative_cycles",
            "t_sum",
            "t_avg",
            "snow_avg",
            "snow_max",
            "moisture_avg",
            "day_sin",
            "day_cos",
        ]
        last_row = feat_df.iloc[-1:]
        X_tab = last_row[tabular_cols].values.astype(np.float32)

        proba = model.predict_proba(X_tab)[0]
        pred_class = int(np.argmax(proba))

    proba_dict = {
        "P(Норма, Класс 0)": float(proba[0]),
        "P(Преддефект, Класс 1)": float(proba[1]),
        "P(Авария, Класс 2)": float(proba[2]),
    }

    # Ожидаемая расчетная деформация (взвешенная сумма)
    expected_heave = float(sum(proba[c] * CLASS_HEAVE_ESTIMATE[c] for c in range(3)))

    report = PredictionReport(
        forecast_date=forecast_date,
        target_date=future_date,
        horizon_days=horizon_days,
        predicted_class=pred_class,
        predicted_label=CLASS_NAMES[pred_class],
        probabilities=proba_dict,
        expected_heave_cm=round(expected_heave, 2),
        recommendation=CLASS_DESCRIPTIONS[pred_class],
        model_name=model_type,
        attention_weights=attention_vector,
        df_forecast=feat_df,
    )

    if save_plot_path is not None:
        plot_inference_forecast(report, save_path=save_plot_path)

    return report


def predict_batch(
    data: str | Path | pd.DataFrame,
    model_type: Literal["hybrid_nn", "xgboost", "rf"] = "hybrid_nn",
    horizon_days: int = 30,
    site_id: str | None = None,
    output_csv: str | Path = "results/tables/batch_predictions.csv",
    models_dir: str | Path = "saved_models",
) -> pd.DataFrame:
    """Выполняет пакетный расчет прогнозов по всем дням временного ряда."""
    if isinstance(data, (str, Path)):
        df_raw = pd.read_csv(data)
    else:
        df_raw = data.copy()

    df_raw["date"] = pd.to_datetime(df_raw["date"])

    if site_id is not None and "site_id" in df_raw.columns:
        df_work = df_raw[df_raw["site_id"] == site_id].copy()
    else:
        df_work = df_raw.copy()

    # Обработка постов и сезонов
    group_keys = []
    if "site_id" in df_work.columns:
        group_keys.append("site_id")
    if "season" in df_work.columns:
        group_keys.append("season")

    model, _ = load_model_for_inference(model_type, horizon_days=horizon_days, models_dir=models_dir)

    tabular_cols = [
        "f_cross",
        "f_cross_decade",
        "cumulative_cycles",
        "t_sum",
        "t_avg",
        "snow_avg",
        "snow_max",
        "moisture_avg",
        "day_sin",
        "day_cos",
    ]

    all_results = []

    if group_keys:
        grouped = df_work.groupby(group_keys)
    else:
        grouped = [(None, df_work)]

    for _, group in grouped:
        g = group.sort_values("date").reset_index(drop=True)
        feat_g = compute_physical_features(g, window_days=30, decade_days=10)

        if model_type == "hybrid_nn":
            seq_length = 45
            if len(feat_g) < seq_length:
                continue
            X_seq, valid_idx = extract_sequential_windows(feat_g, seq_length=seq_length)
            probas = model.predict(X_seq, verbose=0)
            aligned_dates = feat_g.loc[valid_idx, "date"].values
            res_df = feat_g.loc[valid_idx].copy()
        else:
            X_tab = feat_g[tabular_cols].values.astype(np.float32)
            probas = model.predict_proba(X_tab)
            res_df = feat_g.copy()

        pred_classes = np.argmax(probas, axis=1)
        res_df["horizon_days"] = horizon_days
        res_df["target_date"] = res_df["date"] + pd.Timedelta(days=horizon_days)
        res_df["pred_class"] = pred_classes
        res_df["pred_label"] = [CLASS_NAMES[c] for c in pred_classes]
        res_df["proba_class_0"] = probas[:, 0].round(4)
        res_df["proba_class_1"] = probas[:, 1].round(4)
        res_df["proba_class_2"] = probas[:, 2].round(4)
        res_df["expected_heave_cm"] = (
            probas[:, 0] * CLASS_HEAVE_ESTIMATE[0] +
            probas[:, 1] * CLASS_HEAVE_ESTIMATE[1] +
            probas[:, 2] * CLASS_HEAVE_ESTIMATE[2]
        ).round(2)
        res_df["alarm_triggered"] = probas[:, 2] >= 0.50

        all_results.append(res_df)

    final_df = pd.concat(all_results, ignore_index=True)

    out_p = Path(output_csv)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(out_p, index=False)
    print(f"[Inference] Пакетный прогноз сохранен: {out_p.resolve()} ({len(final_df)} строк)")

    return final_df


def plot_inference_forecast(report: PredictionReport, save_path: str | Path) -> None:
    """Визуализирует недавнюю предысторию и прогноз риска в будущее на +tau дней."""
    df_hist = report.df_forecast.tail(45).copy()
    dates = df_hist["date"].values
    t_air = df_hist["t_air"].values
    cycles = df_hist["is_zero_crossing"].values

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=False, gridspec_kw={"height_ratios": [1.4, 1.0]})

    # Верхний график: Температура воздуха и моменты циклов 0 °C
    ax1.plot(dates, t_air, color="#1f77b4", lw=2, label="Температура воздуха T_air (°C)")
    ax1.axhline(0, color="gray", linestyle="--", alpha=0.7)
    ax1.fill_between(dates, t_air, 0, where=(t_air <= 0), color="skyblue", alpha=0.3, label="Область мороза")

    cycle_dates = dates[cycles == 1]
    cycle_temps = t_air[cycles == 1]
    if len(cycle_dates) > 0:
        ax1.scatter(cycle_dates, cycle_temps, color="gold", edgecolor="black", s=70, zorder=5, label="Переходы через 0 °C (F_cross)")

    ax1.axvline(report.forecast_date, color="darkred", linestyle=":", lw=1.8, label=f"Дата анализа: {report.forecast_date.strftime('%Y-%m-%d')}")
    ax1.set_title(f"Оперативный мониторинг телеметрии и упреждающий прогноз на {report.target_date.strftime('%Y-%m-%d')} (+{report.horizon_days} дней)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("T_air, °C", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(loc="lower left", fontsize=9)

    # Нижний график: Вероятностный профиль прогноза дефектов
    class_colors = ["#2ca02c", "#ff7f0e", "#d62728"]
    probs = [
        report.probabilities["P(Норма, Класс 0)"] * 100,
        report.probabilities["P(Преддефект, Класс 1)"] * 100,
        report.probabilities["P(Авария, Класс 2)"] * 100,
    ]
    labels = ["Норма (Класс 0)\n< 1.5 см", "Преддефект (Класс 1)\n1.5 – 3.0 см", "Авария / Тревога (Класс 2)\n>= 3.0 см"]

    bars = ax2.barh(labels, probs, color=class_colors, alpha=0.85, height=0.55)
    ax2.set_xlim(0, 105)
    ax2.set_xlabel("Прогнозная вероятность состояния полотна, %", fontsize=10)
    ax2.axvline(50, color="black", linestyle="--", lw=1.2, alpha=0.6, label="Порог тревоги (50%)")
    ax2.grid(True, axis="x", linestyle=":", alpha=0.5)

    for bar, p in zip(bars, probs):
        ax2.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2.0, f"{p:.1f}%", va="center", fontweight="bold", fontsize=10)

    # Текстовый статус
    status_text = (
        f"ИТОГОВЫЙ ПРОГНОЗ: {report.predicted_label}\n"
        f"Ожидаемая деформация: ~{report.expected_heave_cm} см\n"
        f"Рекомендация: {report.recommendation}"
    )
    plt.figtext(0.12, -0.04, status_text, fontsize=10, bbox=dict(boxstyle="round,pad=0.5", facecolor="#f7f7f7", edgecolor="gray"))

    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Inference] График прогноза сохранен: {p.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="PuchinaAI: Оперативный прогноз риска деформаций дорожного полотна")
    parser.add_argument("--data", type=str, default="data/processed/dataset_defect_1_labelled.csv", help="Путь к файлу телеметрии CSV")
    parser.add_argument("--model", type=str, default="hybrid_nn", choices=["hybrid_nn", "xgboost", "rf"], help="Тип модели для прогноза")
    parser.add_argument("--horizon", type=int, default=30, help="Горизонт прогноза в днях (30, 60, 90)")
    parser.add_argument("--site", type=str, default=None, help="Идентификатор поста (например 'КП1', 'КП2', 'ОП1')")
    parser.add_argument("--date", type=str, default=None, help="Дата прогноза YYYY-MM-DD (по умолчанию последняя дата в файле)")
    parser.add_argument("--batch", action="store_true", help="Пакетный расчет по всем датам датасета с экспортом в CSV")
    parser.add_argument("--plot", type=str, default="results/figures/latest_forecast.png", help="Путь для сохранения графика")
    parser.add_argument("--output", type=str, default="results/tables/batch_predictions.csv", help="Путь для сохранения batch CSV")

    args = parser.parse_args()

    print("=" * 80)
    print("ПУЧИНА-AI: СИСТЕМА ОПЕРАТИВНОГО ПРЕДИКТИВНОГО МОНИТОРИНГА")
    print(f"Модель: {args.model.upper()} | Горизонт: {args.horizon} дней")
    print("=" * 80)

    if args.batch:
        df_batch = predict_batch(
            data=args.data,
            model_type=args.model,
            horizon_days=args.horizon,
            site_id=args.site,
            output_csv=args.output,
        )
        print(f"\n[СВОДКА ПАКЕТНОГО ПРОГНОЗА]")
        print(f"  • Всего рассчитано дней прогноза: {len(df_batch)}")
        print(f"  • Распределение прогнозов:")
        for cls_id, name in CLASS_NAMES.items():
            count = (df_batch["pred_class"] == cls_id).sum()
            pct = count / len(df_batch) * 100
            print(f"      - {name:25s}: {count:5d} ({pct:5.1f}%)")
        alarm_cnt = df_batch["alarm_triggered"].sum()
        print(f"  • Дней с превышением порога тревоги (P2 >= 50%): {alarm_cnt} ({alarm_cnt / len(df_batch) * 100:.1f}%)")
        print(f"  • Файл сохранен в: {args.output}")

    else:
        report = predict_future_risk(
            data=args.data,
            model_type=args.model,
            horizon_days=args.horizon,
            site_id=args.site,
            as_of_date=args.date,
            save_plot_path=args.plot,
        )

        print(f"\n[РЕЗУЛЬТАТЫ ПРОГНОЗА]")
        print(f"  • Дата расчета:          {report.forecast_date.strftime('%Y-%m-%d')}")
        print(f"  • Дата упреждения:       {report.target_date.strftime('%Y-%m-%d')} (+{report.horizon_days} дней)")
        print(f"  • Прогнозный класс Y:    {report.predicted_label}")
        print(f"  • Ожидаемая деформация:  ~{report.expected_heave_cm} см")
        print(f"  • Вероятности классов:")
        for k, v in report.probabilities.items():
            print(f"      - {k:25s}: {v * 100:.1f}%")
        print(f"  • Регламентная рекомендация:")
        print(f"      {report.recommendation}")
    print("=" * 80)


if __name__ == "__main__":
    main()
