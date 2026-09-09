"""
Модуль загрузки, валидации и первичной подготовки данных.
Поддерживает загрузку как пользовательских сырых CSV, так и генерацию синтетики.
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
from src.data_generator import generate_weather_and_heave_data


def load_dataset(
    data_path: str | Path | None = None,
    raw_dir: str | Path = "data/raw",
    synthetic_dir: str | Path = "data/synthetic",
    auto_generate: bool = True,
) -> pd.DataFrame:
    """Загружает временной ряд метеоданных и дефектов дорожного полотна.

    Приоритет поиска:
    1. Явно переданный data_path (если существует).
    2. Файл в raw_dir (например, data/raw/weather_heave.csv или любой *.csv).
    3. Файл в synthetic_dir.
    4. Если ничего нет и auto_generate=True, генерирует синтетический ряд.
    """
    # 1. Проверка явно заданного пути
    if data_path is not None:
        p = Path(data_path)
        if p.exists():
            print(f"[DataLoader] Загрузка данных из: {p.resolve()}")
            return _read_and_validate(p)

    # 2. Поиск в raw_dir
    raw_path = Path(raw_dir)
    if raw_path.exists():
        raw_files = list(raw_path.glob("*.csv"))
        if raw_files:
            target = raw_files[0]
            print(f"[DataLoader] Обнаружен сырой файл данных: {target.resolve()}")
            return _read_and_validate(target)

    # 3. Поиск в synthetic_dir
    synth_path = Path(synthetic_dir)
    if synth_path.exists():
        synth_files = list(synth_path.glob("*.csv"))
        if synth_files:
            target = synth_files[0]
            print(f"[DataLoader] Обнаружен синтетический датасет: {target.resolve()}")
            return _read_and_validate(target)

    # 4. Автогенерация
    if auto_generate:
        print("[DataLoader] Данные не обнаружены. Запуск физической генерации временного ряда...")
        df = generate_weather_and_heave_data(output_dir=synth_path)
        return df

    raise FileNotFoundError(f"Данные не найдены в {raw_dir} или {synthetic_dir}, и auto_generate=False.")


def _read_and_validate(file_path: Path) -> pd.DataFrame:
    """Читает CSV, проверяет обязательные колонки и сортирует по дате."""
    df = pd.read_csv(file_path)

    # Приведение даты к datetime
    date_col = None
    for c in ["date", "Date", "DATE", "datetime", "timestamp"]:
        if c in df.columns:
            date_col = c
            break

    if date_col is None:
        raise ValueError(f"В файле {file_path} не найдена колонка с датой ('date').")

    df["date"] = pd.to_datetime(df[date_col])
    df = df.sort_values("date").reset_index(drop=True)

    # Проверка обязательных полей
    required_cols = ["t_air"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"В данных отсутствует обязательная колонка '{col}'. Доступные: {list(df.columns)}")

    # Заполнение пропусков при наличии
    if df.isna().sum().sum() > 0:
        print(f"[DataLoader] Предупреждение: обнаружено {df.isna().sum().sum()} пропусков, интерполяция...")
        df = df.ffill().bfill()

    return df
