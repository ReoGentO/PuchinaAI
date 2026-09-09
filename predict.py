"""
Модуль практического применения и оперативного прогнозирования рисков (Inference).

Использование в эксплуатационной дорожной службе:
1. Загрузка обученной модели (XGBoost, Random Forest или Hybrid CNN+LSTM).
2. Расчет оперативных физических факторов (Fcross, Tsum, Tavg, снег, влажность)
   по метеоданным за последние 30-60 дней + прогнозу на 7 дней вперед.
3. Выдача класса риска, вероятностного профиля и инженерных рекомендаций
   по предупреждению пучинообразования.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
from tensorflow import keras

# Безопасный вывод на Windows PowerShell / cmd
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.features import compute_physical_features


CLASS_DESCRIPTIONS = {
    0: {
        "status": "[OK] НОРМА (Безопасно)",
        "depth": "< 1.0 см",
        "action": "Штатный режим эксплуатации. Плановый визуальный контроль.",
    },
    1: {
        "status": "[!] ПРЕДДЕФЕКТНОЕ СОСТОЯНИЕ (Внимание)",
        "depth": "1.0 – 3.0 см",
        "action": "Усиленный мониторинг участка. Контроль водоотвода (очистка кюветов и водопропускных труб от льда).",
    },
    2: {
        "status": "[ТРЕВОГА] АВАРИЙНАЯ ОПАСНОСТЬ (Риск пучины)",
        "depth": "> 3.0 см или разрыв покрытия",
        "action": "Срочные превентивные меры: введение временного ограничения осевой нагрузки грузового транспорта, инструментальный замер прогиба полотна, подготовка локального ремонта.",
    },
}


def run_operational_inference(
    data_path: str = "data/synthetic/simulated_weather_heave_timeseries.csv",
    model_type: str = "xgb",
    horizon_days: int = 30,
    reference_date: str | None = None,
) -> dict:
    """Выполняет оперативный расчет рисков для заданного дня наблюдения."""
    models_dir = Path("saved_models")
    xgb_path = models_dir / f"xgb_model_{horizon_days}d.joblib"
    rf_path = models_dir / f"rf_model_{horizon_days}d.joblib"
    nn_path = models_dir / f"hybrid_nn_{horizon_days}d.keras"

    # 1. Загрузка данных
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # 2. Выбор даты наблюдения
    if reference_date is not None:
        ref_dt = pd.to_datetime(reference_date)
        idx_matches = df[df["date"] <= ref_dt]
        if idx_matches.empty:
            raise ValueError(f"Дата {reference_date} раньше доступного диапазона в данных.")
        curr_idx = idx_matches.index[-1]
    else:
        # По умолчанию берем день в зимне-весенний период с активными переходами через ноль
        curr_idx = len(df) - 100

    target_row = df.iloc[curr_idx]
    obs_date = target_row["date"].strftime("%Y-%m-%d")

    # 3. Расчет признаков за предшествующие 30 дней
    feat_df = compute_physical_features(df, window_days=30, decade_days=10)
    row_feat = feat_df.iloc[curr_idx]

    tab_features = [
        "f_cross",
        "f_cross_decade",
        "t_sum",
        "t_avg",
        "snow_avg",
        "snow_max",
        "moisture_avg",
        "day_sin",
        "day_cos",
    ]
    X_vec = row_feat[tab_features].values.astype(np.float32).reshape(1, -1)

    # 4. Прогноз модели
    if model_type.lower() == "xgb" and xgb_path.exists():
        model = joblib.load(xgb_path)
        probas = model.predict_proba(X_vec)[0]
    elif model_type.lower() == "rf" and rf_path.exists():
        model = joblib.load(rf_path)
        probas = model.predict_proba(X_vec)[0]
    elif model_type.lower() in ["nn", "cnn_lstm"] and nn_path.exists():
        model = keras.models.load_model(nn_path)
        # Формирование последовательности 60 дней
        if curr_idx < 59:
            raise ValueError("Недостаточно истории (нужно минимум 60 дней) для нейросети.")
        sub_seq = df.iloc[curr_idx - 59 : curr_idx + 1]
        t_air = sub_seq["t_air"].values
        snow = sub_seq["snow_depth"].values if "snow_depth" in sub_seq else np.zeros(60)
        doy = (sub_seq["date"].dt.dayofyear.values - 1) / 365.25
        X_seq = np.stack([t_air, snow, doy], axis=-1).astype(np.float32).reshape(1, 60, 3)
        probas = model.predict(X_seq, verbose=0)[0]
    else:
        # Если сохраненных моделей еще нет, запускаем быстрое обучение или используем доступный xgb
        raise FileNotFoundError(
            f"Модель {model_type} для горизонта {horizon_days} дней не найдена в {models_dir}. "
            f"Сначала запустите: python main.py --horizon {horizon_days}"
        )

    pred_class = int(np.argmax(probas))
    p0, p1, p2 = probas[0] * 100, probas[1] * 100, probas[2] * 100

    f_cross_30d = row_feat["f_cross"]
    f_cross_10d = row_feat["f_cross_decade"]
    t_sum_val = row_feat["t_sum"]
    t_avg_val = row_feat["t_avg"]

    # 5. Оценка порогов и тревоги
    is_alarm = (f_cross_10d >= 4) or (p2 >= 40.0)

    report = {
        "observation_date": obs_date,
        "forecast_horizon_days": horizon_days,
        "target_date_approx": (row_feat["date"] + pd.Timedelta(days=horizon_days)).strftime("%Y-%m-%d"),
        "predicted_class": pred_class,
        "class_info": CLASS_DESCRIPTIONS[pred_class],
        "probabilities": {"P_normal": round(p0, 1), "P_subdefect": round(p1, 1), "P_emergency": round(p2, 1)},
        "physical_factors": {
            "f_cross_decade (циклы за 10д)": int(f_cross_10d),
            "f_cross_30d (циклы за 30д)": int(f_cross_30d),
            "t_sum (градусо-дни холода)": round(float(t_sum_val), 1),
            "t_avg (средняя T за 30д)": round(float(t_avg_val), 1),
            "current_t_air": round(float(row_feat["t_air"]), 1),
        },
        "alarm_triggered": is_alarm,
    }

    _print_operational_report(report)
    return report


def _print_operational_report(r: dict) -> None:
    print("\n" + "=" * 70)
    print("[ОТЧЕТ] ОПЕРАТИВНОЕ ЗАКЛЮЧЕНИЕ ПРЕДИКТИВНОГО МОНИТОРИНГА ПУЧИНООБРАЗОВАНИЯ")
    print("=" * 70)
    print(f"* Дата фиксации метеоусловий: {r['observation_date']}")
    print(f"* Горизонт упреждения:        {r['forecast_horizon_days']} дней (целевой период: ~{r['target_date_approx']})")
    print(f"* Текущая температура:       {r['physical_factors']['current_t_air']} °C")
    print(f"* Циклов замерзания/оттайки:  {r['physical_factors']['f_cross_decade (циклы за 10д)']} за декаду, {r['physical_factors']['f_cross_30d (циклы за 30д)']} за 30 дней")
    print(f"* Сумма отр. температур Tsum: {r['physical_factors']['t_sum (градусо-дни холода)']} °C·дней")
    print("-" * 70)
    print(f"ПРОГНОЗ СОСТОЯНИЯ: {r['class_info']['status']}")
    print(f"* Ожидаемая глубина деформации: {r['class_info']['depth']}")
    print(f"* Распределение вероятностей:")
    print(f"    - Класс 0 (Норма):        {r['probabilities']['P_normal']}%")
    print(f"    - Класс 1 (Преддефект):   {r['probabilities']['P_subdefect']}%")
    print(f"    - Класс 2 (Аварийное):    {r['probabilities']['P_emergency']}%")
    print("-" * 70)
    if r["alarm_triggered"]:
        print("[!] ВНИМАНИЕ: СРАБОТАЛ КРИТЕРИЙ ТРЕВОГИ ПО ПУЧИНООБРАЗОВАНИЮ!")
    print(f"РЕКОМЕНДАЦИЯ СЛУЖБЕ ЭКСПЛУАТАЦИИ:\n  -> {r['class_info']['action']}")
    print("=" * 70 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="PuchinaAI Operational Inference")
    parser.add_argument("--data", type=str, default="data/synthetic/simulated_weather_heave_timeseries.csv")
    parser.add_argument("--horizon", type=int, default=30, help="Горизонт (30, 60, 90)")
    parser.add_argument("--model", type=str, default="xgb", choices=["xgb", "rf", "cnn_lstm"])
    parser.add_argument("--date", type=str, default=None, help="Дата среза в формате ГГГГ-ММ-ДД")

    args = parser.parse_args()
    run_operational_inference(
        data_path=args.data,
        model_type=args.model,
        horizon_days=args.horizon,
        reference_date=args.date,
    )


if __name__ == "__main__":
    main()
