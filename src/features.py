"""
Модуль расчета физических признаков пучинообразования (Feature Engineering).

Рассчитывает:
1. Табличные кумулятивные признаки за скользящее окно (W=30 дней):
   - F_cross: число переходов температуры через 0 °C (циклы замерзания-оттаивания).
   - F_cross_decade: число переходов через 0 °C за последнюю декаду (10 дней).
   - cumulative_cycles: кумулятивное число циклов за текущий сезон.
   - T_sum: кумулятивная сумма отрицательных температур за окно (градусо-дни промерзания).
   - T_avg: средняя температура воздуха за окно.
   - Snow_avg, Snow_max: средняя и максимальная высота снежного покрова.
   - Moisture_avg: влажность грунта.
   - Day_sin, Day_cos: синусно-косинусное кодирование дня года.

2. 3D-тензоры временных последовательностей для нейросети (Batch, Seq_len, Features):
   - [T_air_norm, daily_cycle_flag, zero_cross_proximity, cum_cycles_norm, T_ground_0.2_norm, Snow_norm, Moisture_norm].
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
    if "daily_cycle_flag" in res.columns:
        crossings = res["daily_cycle_flag"].values.astype(int)
    else:
        crossings = compute_zero_crossings(t_air)
    res["is_zero_crossing"] = crossings

    # F_cross за окно W (30 дней)
    res["f_cross"] = (
        pd.Series(crossings).rolling(window=window_days, min_periods=1).sum().values
    )

    # F_cross_decade (за последние 10 дней)
    res["f_cross_decade"] = (
        pd.Series(crossings).rolling(window=decade_days, min_periods=1).sum().values
    )

    # Кумулятивные циклы за сезон
    if "cumulative_cycles" not in res.columns:
        res["cumulative_cycles"] = pd.Series(crossings).cumsum().values

    # Непрерывная близость температуры к точке фазового перехода (0 °C)
    res["zero_cross_proximity"] = np.exp(-np.abs(t_air) / 3.0).round(4)

    # 2. T_sum — сумма отрицательных температур за окно (градусо-дни холода)
    t_neg = np.where(t_air < 0, t_air, 0.0)
    res["t_sum"] = (
        pd.Series(t_neg).rolling(window=window_days, min_periods=1).sum().round(2).values
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
    elif "w_ground_0.5" in res.columns:
        res["ground_moisture"] = np.round(
            0.35 * res["w_ground_0.5"] + 0.30 * res.get("w_ground_1.0", 25.0) +
            0.20 * res.get("w_ground_1.5", 25.0) + 0.15 * res.get("w_ground_2.0", 25.0),
            1
        )
        res["moisture_avg"] = (
            res["ground_moisture"].rolling(window=window_days, min_periods=1).mean().round(2)
        )
    else:
        res["ground_moisture"] = 22.0
        res["moisture_avg"] = 22.0

    # 5. Циклическое кодирование времени года
    dates = pd.to_datetime(res["date"])
    day_of_year = dates.dt.dayofyear.values
    res["day_of_year"] = day_of_year
    res["day_sin"] = np.sin(2 * np.pi * day_of_year / 365.25).round(4)
    res["day_cos"] = np.cos(2 * np.pi * day_of_year / 365.25).round(4)

    return res


def extract_sequential_windows(
    df: pd.DataFrame,
    seq_length: int = 45,
    feature_cols: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Извлекает 3D-тензор временных окон фиксированной длины для нейросети.

    Параметры:
        df: подготовленный DataFrame.
        seq_length: длина временного окна в днях (по умолчанию 45 дней).
        feature_cols: список признаков для включения в тензор.
                      По умолчанию нормализованные физические каналы.

    Возвращает:
        (X_seq, valid_indices):
        - X_seq: массив размерности (N_windows, seq_length, num_features).
        - valid_indices: соответствующие индексы строк исходного df.
    """
    work_df = df.copy()

    # Подготовка нормализованных признаков
    if "t_air_norm" not in work_df.columns:
        work_df["t_air_norm"] = (work_df["t_air"] / 20.0).astype(np.float32)

    if "daily_cycle_flag" not in work_df.columns:
        work_df["daily_cycle_flag"] = compute_zero_crossings(work_df["t_air"].values).astype(np.float32)

    if "zero_cross_proximity" not in work_df.columns:
        work_df["zero_cross_proximity"] = np.exp(-np.abs(work_df["t_air"]) / 3.0).astype(np.float32)

    if "cum_cycles_norm" not in work_df.columns:
        if "cumulative_cycles" in work_df.columns:
            work_df["cum_cycles_norm"] = (work_df["cumulative_cycles"] / 50.0).astype(np.float32)
        else:
            work_df["cum_cycles_norm"] = (work_df["daily_cycle_flag"].cumsum() / 50.0).astype(np.float32)

    if "t_g02_norm" not in work_df.columns:
        if "t_ground_0.2" in work_df.columns:
            work_df["t_g02_norm"] = (work_df["t_ground_0.2"] / 10.0).astype(np.float32)
        else:
            work_df["t_g02_norm"] = 0.0

    if "snow_norm" not in work_df.columns:
        if "snow_depth" in work_df.columns:
            work_df["snow_norm"] = (work_df["snow_depth"] / 50.0).astype(np.float32)
        else:
            work_df["snow_norm"] = 0.0

    if "moisture_norm" not in work_df.columns:
        if "ground_moisture" in work_df.columns:
            work_df["moisture_norm"] = (work_df["ground_moisture"] / 35.0).astype(np.float32)
        else:
            work_df["moisture_norm"] = 0.65

    default_features = [
        "t_air_norm",
        "daily_cycle_flag",
        "zero_cross_proximity",
        "cum_cycles_norm",
        "t_g02_norm",
        "snow_norm",
        "moisture_norm",
    ]

    if feature_cols is None:
        feature_cols = default_features

    windows = []
    valid_indices = []

    # Группировка по (site_id, season) для исключения сквозных окон через летние перерывы
    group_keys = []
    if "site_id" in work_df.columns:
        group_keys.append("site_id")
    if "season" in work_df.columns:
        group_keys.append("season")

    if group_keys:
        for _, group in work_df.groupby(group_keys):
            g_indices = group.index.values
            g_feat = group[feature_cols].values.astype(np.float32)
            if len(g_feat) < seq_length:
                continue
            for i in range(seq_length - 1, len(g_feat)):
                windows.append(g_feat[i - seq_length + 1 : i + 1])
                valid_indices.append(g_indices[i])
        return np.array(windows, dtype=np.float32), np.array(valid_indices, dtype=int)

    feat_matrix = work_df[feature_cols].values.astype(np.float32)
    n_samples = len(work_df)

    if n_samples < seq_length:
        raise ValueError(f"Длина датасета ({n_samples}) меньше длины последовательности ({seq_length}).")

    for i in range(seq_length - 1, n_samples):
        window = feat_matrix[i - seq_length + 1 : i + 1]
        windows.append(window)
        valid_indices.append(i)

    return np.array(windows, dtype=np.float32), np.array(valid_indices, dtype=int)
