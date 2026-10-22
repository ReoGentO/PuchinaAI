"""
Модуль построения датасетов с временными лагами и строго хронологическим разбиением.

Реализует:
1. Этап 1: Инженерия временных меток.
   - Сдвиг целевой переменной на горизонты прогнозирования tau in {30, 60, 90} дней.
   - Создание 3 отдельных датасетов под каждый горизонт.
2. Этап 5: Временная кросс-валидация и разбиение выборки:
   - Первые 70% по датам — Train
   - Следующие 15% по датам — Validation
   - Последние 15% по датам — Test
   (Без случайного перемешивания, исключая утечку данных из будущего).
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from src.features import compute_physical_features, extract_sequential_windows


@dataclass
class DatasetSplits:
    """Контейнер для хранения обучающей, валидационной и тестовой выборок."""
    horizon_days: int
    train_dates: tuple[pd.Timestamp, pd.Timestamp]
    val_dates: tuple[pd.Timestamp, pd.Timestamp]
    test_dates: tuple[pd.Timestamp, pd.Timestamp]

    # Табличные признаки для XGBoost / RandomForest
    feature_names: list[str]
    X_train_tab: np.ndarray
    y_train: np.ndarray
    X_val_tab: np.ndarray
    y_val: np.ndarray
    X_test_tab: np.ndarray
    y_test: np.ndarray

    # Временные последовательности для CNN+LSTM (Batch, Seq_len, Features)
    seq_feature_names: list[str]
    X_train_seq: np.ndarray
    X_val_seq: np.ndarray
    X_test_seq: np.ndarray

    # Метаданные (исходные даты строк тестовой выборки для графиков динамики)
    df_test: pd.DataFrame


def build_lagged_dataset(
    df: pd.DataFrame,
    horizon_days: int = 90,
    window_days: int = 30,
    seq_length: int = 60,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Формирует датасет для заданного горизонта прогноза tau."""
    # 1. Расчет физических признаков и сдвиг лага
    if "site_id" in df.columns:
        site_dfs = []
        for s, group in df.groupby("site_id"):
            group_sorted = group.sort_values("date").reset_index(drop=True)
            group_feat = compute_physical_features(group_sorted, window_days=window_days)
            group_feat["target_y"] = group_feat["defect_class"].shift(-horizon_days)
            site_dfs.append(group_feat)
        feat_df = pd.concat(site_dfs, ignore_index=True)
    else:
        feat_df = compute_physical_features(df, window_days=window_days)
        feat_df["target_y"] = feat_df["defect_class"].shift(-horizon_days)

    # Исключаем последние дни, для которых нет будущего горизонта
    valid_df = feat_df.dropna(subset=["target_y"]).reset_index(drop=True)
    valid_df["target_y"] = valid_df["target_y"].astype(int)

    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        filename = out_path / f"dataset_horizon_{horizon_days}d.csv"
        valid_df.to_csv(filename, index=False)
        print(f"[DatasetBuilder] Сохранен датасет для лага {horizon_days} дней: {filename}")

    return valid_df


def prepare_chronological_splits(
    df: pd.DataFrame,
    horizon_days: int = 90,
    seq_length: int = 60,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    tabular_features: list[str] | None = None,
    sequential_features: list[str] | None = None,
    output_dir: str | Path | None = "data/processed",
) -> DatasetSplits:
    """Создает строго хронологическое разбиение данных (70% / 15% / 15%)."""
    if tabular_features is None:
        tabular_features = [
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

    if sequential_features is None:
        sequential_features = ["t_air", "snow_depth", "day_of_year_norm"]

    # 1. Построение датасета с целевой переменной
    lagged_df = build_lagged_dataset(
        df,
        horizon_days=horizon_days,
        window_days=30,
        output_dir=output_dir,
    )

    # 2. Построение 3D-тензоров последовательностей длины seq_length
    X_seq_all, valid_indices = extract_sequential_windows(
        lagged_df,
        seq_length=seq_length,
        feature_cols=sequential_features,
    )

    # Синхронизация табличных признаков и меток с последовательностями
    aligned_df = lagged_df.iloc[valid_indices].reset_index(drop=True)
    aligned_df["date"] = pd.to_datetime(aligned_df["date"])

    # Истинное хронологическое разбиение по уникальным календарным датам
    # Корректно работает как для одиночных рядов, так и для мульти-постовой телеметрии
    unique_dates = np.sort(aligned_df["date"].unique())
    n_dates = len(unique_dates)

    # Для коротких рядов (<= 3 лет) жесткая отсечка 70% оставляет в 15% тесте лишь 4 зимних месяца.
    # Для обеспечения полного сезонного цикла в тестовой выборке используем адаптивный баланс.
    eff_train = train_ratio if n_dates > 1000 else 0.55
    eff_val = val_ratio if n_dates > 1000 else 0.15

    train_date_end = unique_dates[int(n_dates * eff_train)]
    val_date_end = unique_dates[int(n_dates * (eff_train + eff_val))]

    train_mask = (aligned_df["date"] < train_date_end).values
    val_mask = ((aligned_df["date"] >= train_date_end) & (aligned_df["date"] < val_date_end)).values
    test_mask = (aligned_df["date"] >= val_date_end).values

    X_tab_all = aligned_df[tabular_features].values.astype(np.float32)
    y_all = aligned_df["target_y"].values.astype(int)

    # Строго хронологические срезы по календарным датам
    X_train_tab = X_tab_all[train_mask]
    X_train_seq = X_seq_all[train_mask]
    y_train = y_all[train_mask]

    X_val_tab = X_tab_all[val_mask]
    X_val_seq = X_seq_all[val_mask]
    y_val = y_all[val_mask]

    X_test_tab = X_tab_all[test_mask]
    X_test_seq = X_seq_all[test_mask]
    y_test = y_all[test_mask]

    train_dates = (pd.Timestamp(unique_dates[0]), pd.Timestamp(train_date_end))
    val_dates = (pd.Timestamp(train_date_end), pd.Timestamp(val_date_end))
    test_dates = (pd.Timestamp(val_date_end), pd.Timestamp(unique_dates[-1]))

    print(
        f"[DatasetBuilder] Выборка для горизонта {horizon_days} дней:\n"
        f"  Всего сэмплов: {len(aligned_df)} (дат: {n_dates})\n"
        f"  Обучение (70%): {len(y_train)} с {train_dates[0].strftime('%Y-%m-%d')} по {train_dates[1].strftime('%Y-%m-%d')}\n"
        f"  Валидация (15%): {len(y_val)} с {val_dates[0].strftime('%Y-%m-%d')} по {val_dates[1].strftime('%Y-%m-%d')}\n"
        f"  Тест (15%): {len(y_test)} с {test_dates[0].strftime('%Y-%m-%d')} по {test_dates[1].strftime('%Y-%m-%d')}"
    )

    return DatasetSplits(
        horizon_days=horizon_days,
        train_dates=train_dates,
        val_dates=val_dates,
        test_dates=test_dates,
        feature_names=tabular_features,
        X_train_tab=X_train_tab,
        y_train=y_train,
        X_val_tab=X_val_tab,
        y_val=y_val,
        X_test_tab=X_test_tab,
        y_test=y_test,
        seq_feature_names=sequential_features,
        X_train_seq=X_train_seq,
        X_val_seq=X_val_seq,
        X_test_seq=X_test_seq,
        df_test=aligned_df.loc[test_mask].reset_index(drop=True),
    )


def create_all_horizons(
    df: pd.DataFrame,
    horizons: tuple[int, ...] = (30, 60, 90),
    output_dir: str | Path = "data/processed",
) -> dict[int, DatasetSplits]:
    """Генерирует датасеты для всех трех горизонтов прогноза (1, 2, 3 месяца)."""
    splits_dict = {}
    for h in horizons:
        splits = prepare_chronological_splits(
            df=df,
            horizon_days=h,
            output_dir=output_dir,
        )
        splits_dict[h] = splits
    return splits_dict
