"""
Модуль расчета физических признаков пучинообразования (Feature Engineering).

Рассчитывает:
1. Табличные кумулятивные признаки за скользящее окно (W=30 дней):
   - F_cross: число переходов температуры через 0 °C (циклы замерзания-оттаивания).
   - T_sum: кумулятивная сумма отрицательных температур за окно (градусо-дни промерзания).
   - T_avg: средняя температура воздуха за окно.
   - Snow_avg, Snow_max: средняя и максимальная высота снежного покрова.
   - Moisture_avg / Precip_sum: влажность грунта и сумма осадков.
   - Day_sin, Day_cos: синусно-косинусное кодирование дня года.
   - F_cross_decade: число переходов через 0 °C за последнюю декаду (10 дней).

2. 3D-тензоры временных последовательностей для нейросети (Batch, 60 дней, 3 признака):
   - [T_air, Snow, Day_of_year_norm].
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def compute_zero_crossings(series: np.ndarray) -> np.ndarray:
    """Вычисляет бинарный массив моментов перехода температуры через 0 °C."""
    n = len(series)
    crossings = np.zeros(n, dtype=int)
    for i in range(1, n):
        prev_t = series[i - 1]
        curr_t = series[i]
        if (prev_t < 0 and curr_t >= 0) or (prev_t > 0 and curr_t <= 0):
            crossings[i] = 1
    return crossings


def compute_physical_features(
    df: pd.DataFrame,
    window_days: int = 30,
    decade_days: int = 10,
) -> pd.DataFrame:
    """Рассчитывает табличные физические признаки по скользящим окнам.

    Параметры:
        df: DataFrame с колонками ['date', 't_air', 'snow_depth', ...].
        window_days: размер основного скользящего окна (W = 30 дней).
        decade_days: размер декадного окна для анализа порогов (10 дней).

    Возвращает:
        DataFrame с добавленными кумулятивными физическими признаками.
    """
    res = df.copy()
    t_air = res["t_air"].values
    n = len(res)

    # 1. Индикатор переходов через 0 °C
    crossings = compute_zero_crossings(t_air)
    res["is_zero_crossing"] = crossings

    # F_cross за окно W
    res["f_cross"] = (
        pd.Series(crossings).rolling(window=window_days, min_periods=1).sum().values
    )

    # F_cross_decade (за последние 10 дней)
    res["f_cross_decade"] = (
        pd.Series(crossings).rolling(window=decade_days, min_periods=1).sum().values
    )

    # 2. T_sum — сумма отрицательных температур за окно (сумма градусов холода)
    t_neg = np.where(t_air < 0, t_air, 0.0)
    res["t_sum"] = (
        pd.Series(t_neg).rolling(window=window_days, min_periods=1).sum().values
    )

    # 3. T_avg — средняя температура за окно
    res["t_avg"] = (
        pd.Series(t_air).rolling(window=window_days, min_periods=1).mean().round(2).values
    )

    # 4. Снежный покров и влажность
    if "snow_depth" in res.columns:
        res["snow_avg"] = (
            res["snow_depth"].rolling(window=window_days, min_periods=1).mean().round(2)
        )
        res["snow_max"] = (
            res["snow_depth"].rolling(window=window_days, min_periods=1).max().round(2)
        )
    else:
        res["snow_avg"] = 0.0
        res["snow_max"] = 0.0

    if "precipitation" in res.columns:
        res["precip_sum"] = (
            res["precipitation"].rolling(window=window_days, min_periods=1).sum().round(2)
        )
    else:
        res["precip_sum"] = 0.0

    if "ground_moisture" in res.columns:
        res["moisture_avg"] = (
            res["ground_moisture"].rolling(window=window_days, min_periods=1).mean().round(2)
        )

    # 5. Циклическое кодирование времени года
    dates = pd.to_datetime(res["date"])
    day_of_year = dates.dt.dayofyear.values
    res["day_of_year"] = day_of_year
    res["day_sin"] = np.sin(2 * np.pi * day_of_year / 365.25).round(4)
    res["day_cos"] = np.cos(2 * np.pi * day_of_year / 365.25).round(4)

    return res


def extract_sequential_windows(
    df: pd.DataFrame,
    seq_length: int = 60,
    feature_cols: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Извлекает 3D-тензор временных окон фиксированной длины для нейросети.

    Параметры:
        df: подготовленный DataFrame.
        seq_length: длина временного окна в днях (60 дней).
        feature_cols: список признаков для включения в тензор.
                      По умолчанию: ['t_air', 'snow_depth', 'day_of_year_norm'].

    Возвращает:
        (X_seq, valid_indices):
        - X_seq: массив размерности (N - seq_length + 1, seq_length, num_features).
        - valid_indices: соответствующие индексы строк исходного df.
    """
    if feature_cols is None:
        feature_cols = ["t_air", "snow_depth", "day_of_year_norm"]

    work_df = df.copy()
    if "day_of_year_norm" not in work_df.columns:
        doy = pd.to_datetime(work_df["date"]).dt.dayofyear.values
        work_df["day_of_year_norm"] = (doy - 1) / 365.25

    if "snow_depth" not in work_df.columns:
        work_df["snow_depth"] = 0.0

    feat_matrix = work_df[feature_cols].values.astype(np.float32)
    n_samples = len(work_df)

    if n_samples < seq_length:
        raise ValueError(f"Длина датасета ({n_samples}) меньше длины последовательности ({seq_length}).")

    windows = []
    valid_indices = []

    for i in range(seq_length - 1, n_samples):
        window = feat_matrix[i - seq_length + 1 : i + 1]
        windows.append(window)
        valid_indices.append(i)

    return np.array(windows, dtype=np.float32), np.array(valid_indices, dtype=int)
