"""
Модуль построения датасетов с временными лагами и строго хронологическим разбиением.

Реализует:
1. Инженерию временных лагов (сдвиг целевой переменной на горизонты tau in {30, 60, 90} дней).
2. Расчет признаков в разрезе постов мониторинга и сезонов без утечки данных через летние перерывы.
3. Сезонно-хронологическое разбиение выборки:
   - Обучающая выборка (Train): начальные сезоны + первая половина предтестового сезона
   - Валидационная выборка (Val): вторая половина предтестового сезона
   - Тестовая выборка (Test): полный завершающий сезон (Out-Of-Time оценка без подглядывания)
4. Исключение признака 'is_heatpipe' из обучающих матриц.
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

    # Временные последовательности для CNN+BiLSTM+Attention (Batch, Seq_len, Features)
    seq_feature_names: list[str]
    X_train_seq: np.ndarray
    X_val_seq: np.ndarray
    X_test_seq: np.ndarray

    # Метаданные (исходные строки тестовой выборки для графиков динамики)
    df_test: pd.DataFrame


def build_lagged_dataset(
    df: pd.DataFrame,
    horizon_days: int = 30,
    window_days: int = 30,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Формирует датасет для заданного горизонта прогноза tau."""
    group_keys = []
    if "site_id" in df.columns:
        group_keys.append("site_id")
    if "season" in df.columns:
        group_keys.append("season")

    if group_keys:
        site_dfs = []
        for _, group in df.groupby(group_keys):
            group_sorted = group.sort_values("date").reset_index(drop=True)
            group_feat = compute_physical_features(group_sorted, window_days=window_days)
            group_feat["target_y"] = group_feat["defect_class"].shift(-horizon_days)
            site_dfs.append(group_feat)
        feat_df = pd.concat(site_dfs, ignore_index=True)
    else:
        feat_df = compute_physical_features(df, window_days=window_days)
        feat_df["target_y"] = feat_df["defect_class"].shift(-horizon_days)

    # Исключаем дни, для которых нет будущего горизонта
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
    horizon_days: int = 30,
    seq_length: int = 45,
    tabular_features: list[str] | None = None,
    sequential_features: list[str] | None = None,
    output_dir: str | Path | None = "data/processed",
) -> DatasetSplits:
    """Создает строго хронологическое разбиение данных с защитой от утечки."""
    if tabular_features is None:
        tabular_features = [
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

    if sequential_features is None:
        sequential_features = [
            "t_air_norm",
            "daily_cycle_flag",
            "zero_cross_proximity",
            "cum_cycles_norm",
            "t_g02_norm",
            "snow_norm",
            "moisture_norm",
        ]

    # 1. Построение датасета с лаговой целевой переменной
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

    aligned_df = lagged_df.iloc[valid_indices].reset_index(drop=True)
    aligned_df["date"] = pd.to_datetime(aligned_df["date"])

    # 3. Сезонно-хронологическое разбиение
    has_seasons = "season" in aligned_df.columns and aligned_df["season"].nunique() >= 3

    if has_seasons:
        unique_seasons = sorted(aligned_df["season"].unique())
        test_season = unique_seasons[-1]
        mid_season = unique_seasons[-2]
        prior_seasons = unique_seasons[:-2]

        mid_dates = aligned_df[aligned_df["season"] == mid_season]["date"].sort_values().unique()
        split_date_mid = mid_dates[int(len(mid_dates) * 0.60)]

        train_mask = (
            aligned_df["season"].isin(prior_seasons) |
            ((aligned_df["season"] == mid_season) & (aligned_df["date"] <= split_date_mid))
        ).values

        val_mask = (
            (aligned_df["season"] == mid_season) & (aligned_df["date"] > split_date_mid)
        ).values

        test_mask = (aligned_df["season"] == test_season).values

    else:
        unique_dates = np.sort(aligned_df["date"].unique())
        n_dates = len(unique_dates)
        train_date_end = unique_dates[int(n_dates * 0.65)]
        val_date_end = unique_dates[int(n_dates * 0.80)]

        train_mask = (aligned_df["date"] < train_date_end).values
        val_mask = ((aligned_df["date"] >= train_date_end) & (aligned_df["date"] < val_date_end)).values
        test_mask = (aligned_df["date"] >= val_date_end).values

    # Табличные признаки и целевая переменная
    X_tab_all = aligned_df[tabular_features].values.astype(np.float32)
    y_all = aligned_df["target_y"].values.astype(int)

    X_train_tab = X_tab_all[train_mask]
    X_train_seq = X_seq_all[train_mask]
    y_train = y_all[train_mask]

    X_val_tab = X_tab_all[val_mask]
    X_val_seq = X_seq_all[val_mask]
    y_val = y_all[val_mask]

    X_test_tab = X_tab_all[test_mask]
    X_test_seq = X_seq_all[test_mask]
    y_test = y_all[test_mask]

    dates_train = aligned_df.loc[train_mask, "date"]
    dates_val = aligned_df.loc[val_mask, "date"]
    dates_test = aligned_df.loc[test_mask, "date"]

    train_dates = (dates_train.min(), dates_train.max())
    val_dates = (dates_val.min(), dates_val.max())
    test_dates = (dates_test.min(), dates_test.max())

    print(
        f"[DatasetBuilder] Выборка для горизонта {horizon_days} дней (окно={seq_length}д):\n"
        f"  Обучение: {len(y_train)} с {train_dates[0].strftime('%Y-%m-%d')} по {train_dates[1].strftime('%Y-%m-%d')}\n"
        f"  Валидация: {len(y_val)} с {val_dates[0].strftime('%Y-%m-%d')} по {val_dates[1].strftime('%Y-%m-%d')}\n"
        f"  Тест: {len(y_test)} с {test_dates[0].strftime('%Y-%m-%d')} по {test_dates[1].strftime('%Y-%m-%d')}\n"
        f"  Распределение классов в тесте: Норма={int((y_test==0).sum())}, Преддефект={int((y_test==1).sum())}, Авария={int((y_test==2).sum())}"
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
    seq_length: int = 45,
    output_dir: str | Path = "data/processed",
) -> dict[int, DatasetSplits]:
    """Генерирует датасеты для всех заданных горизонтов прогноза."""
    splits_dict = {}
    for h in horizons:
        splits = prepare_chronological_splits(
            df=df,
            horizon_days=h,
            seq_length=seq_length,
            output_dir=output_dir,
        )
        splits_dict[h] = splits
    return splits_dict
