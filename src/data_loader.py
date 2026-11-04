"""
Модуль загрузки, валидации и первичной подготовки данных.
Поддерживает загрузку пользовательских сырых и размеченных CSV-файлов телеметрии.
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
from src.data_generator import generate_weather_and_heave_data
from src.label_dataset import label_telemetry_dataset


def load_dataset(
    data_path: str | Path | None = None,
    raw_dir: str | Path = "data/raw",
    processed_dir: str | Path = "data/processed",
    synthetic_dir: str | Path = "data/synthetic",
    auto_generate: bool = True,
) -> pd.DataFrame:
    """Загружает датасет телеметрии и дефектов дорожного полотна.

    Приоритет поиска:
    1. Явно переданный data_path (если существует).
    2. Файл в processed_dir (например, dataset_defect_1_labelled.csv).
    3. Файл в raw_dir (например, dataset_defect_1.csv) с автоматической нормативной разметкой.
    4. Файл в synthetic_dir.
    5. Если ничего нет и auto_generate=True, генерирует физический ряд.
    """
    # 1. Проверка явно заданного пути
    if data_path is not None:
        p = Path(data_path)
        if p.exists():
            print(f"[DataLoader] Загрузка данных из: {p.resolve()}")
            df = _read_and_validate(p)
            if "defect_class" not in df.columns:
                print(f"[DataLoader] В файле {p.name} отсутствуют метки 'defect_class'. Выполняется разметка...")
                df = label_telemetry_dataset(input_csv=p)
            return df

    # 2. Поиск в processed_dir
    proc_path = Path(processed_dir)
    if proc_path.exists():
        proc_files = list(proc_path.glob("*defect*.csv")) + list(proc_path.glob("*labelled*.csv"))
        for target in proc_files:
            try:
                cols = pd.read_csv(target, nrows=2).columns
                if "defect_class" in cols:
                    print(f"[DataLoader] Обнаружен размеченный датасет: {target.resolve()}")
                    return _read_and_validate(target)
            except Exception:
                continue

    # 3. Поиск в raw_dir
    raw_path = Path(raw_dir)
    if raw_path.exists():
        raw_files = list(raw_path.glob("*defect*.csv")) + list(raw_path.glob("*.csv"))
        for target in raw_files:
            try:
                cols = pd.read_csv(target, nrows=2).columns
                if "defect_class" in cols:
                    print(f"[DataLoader] Обнаружен сырой файл с готовой разметкой: {target.resolve()}")
                    return _read_and_validate(target)
                elif "daily_cycle_flag" in cols or "t_air" in cols:
                    print(f"[DataLoader] Обнаружен сырой файл телеметрии {target.name}. Запуск нормативной разметки...")
                    labeled_df = label_telemetry_dataset(
                        input_csv=target,
                        output_csv=proc_path / f"{target.stem}_labelled.csv",
                    )
                    return _read_and_validate(proc_path / f"{target.stem}_labelled.csv")
            except Exception:
                continue

    # 4. Поиск в synthetic_dir
    synth_path = Path(synthetic_dir)
    if synth_path.exists():
        synth_files = list(synth_path.glob("*.csv"))
        if synth_files:
            target = synth_files[0]
            print(f"[DataLoader] Обнаружен синтетический датасет: {target.resolve()}")
            return _read_and_validate(target)

    # 5. Автогенерация
    if auto_generate:
        print("[DataLoader] Данные не обнаружены. Запуск физической генерации временного ряда...")
        df = generate_weather_and_heave_data(output_dir=synth_path)
        return df

    raise FileNotFoundError(f"Данные не найдены в {raw_dir} или {processed_dir}, и auto_generate=False.")


def _read_and_validate(file_path: Path) -> pd.DataFrame:
    """Читает CSV, проверяет обязательные колонки и сортирует по дате."""
    df = pd.read_csv(file_path)

    date_col = None
    for c in ["date", "Date", "DATE", "datetime", "timestamp"]:
        if c in df.columns:
            date_col = c
            break

    if date_col is None:
        raise ValueError(f"В файле {file_path} не найдена колонка с датой ('date').")

    df["date"] = pd.to_datetime(df[date_col])
    if "site_id" in df.columns:
        df = df.sort_values(["site_id", "date"]).reset_index(drop=True)
    else:
        df = df.sort_values("date").reset_index(drop=True)

    required_cols = ["t_air"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"В данных отсутствует обязательная колонка '{col}'. Доступные: {list(df.columns)}")

    if df.isna().sum().sum() > 0:
        print(f"[DataLoader] Предупреждение: обнаружено {df.isna().sum().sum()} пропусков, интерполяция...")
        df = df.ffill().bfill()

    return df
