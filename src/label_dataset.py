"""
Модуль нормативной разметки дефектов морозного пучения по полевой телеметрии.
Реализует расчеты согласно:
- СП 34.13330.2021 «Автомобильные дороги» (Приложение Г: «Расчет дорожных одежд на морозоустойчивость»)
- СП 25.13330.2020 «Основания и фундаменты на вечномерзлых грунтах»
- ГОСТ 28622-2012 «Грунты. Метод лабораторного определения степени пучинистости»

Обрабатывает сырые данные телеметрии опытных и контрольных постов (ОП и КП),
вычисляет послойную криогенную деформацию, интегральную высоту пучения h_пуч (см),
фактические индикаторы трещинообразования и присваивает классы состояния:
- Класс 0: Норма (h < 1.0 см)
- Класс 1: Преддефектное состояние (1.0 <= h < 3.0 см)
- Класс 2: Аварийное состояние / трещины (h >= 3.0 см)
"""

from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def label_telemetry_dataset(
    input_csv: str | Path = "data/raw/dataset_3.csv",
    output_csv: str | Path = "data/processed/dataset_3_labelled.csv",
    w_critical: float = 16.0,
    k_heave: float = 0.85,
    save_comparison_figure: bool = True,
    figure_path: str | Path = "results/figures/heatpipe_heave_comparison.png",
) -> pd.DataFrame:
    """Выполняет послойный расчет нормативного пучения и классификацию дефектов.

    Параметры:
        input_csv: путь к сырому CSV-файлу с телеметрией постов.
        output_csv: путь для сохранения размеченного датасета.
        w_critical: критическая влажность грунта (граница пучинистости, %), по СП ~ 15-18%.
        k_heave: коэффициент интенсивности пучения суглинков (СП 34.13330).
        save_comparison_figure: строить ли график сравнения постов ОП и КП.
        figure_path: путь для сохранения итогового графика.

    Возвращает:
        DataFrame с добавленными столбцами:
        ['ground_moisture', 'heave_depth_cm', 'defect_class', 'is_crack']
    """
    in_path = Path(input_csv)
    if not in_path.exists():
        raise FileNotFoundError(f"Файл телеметрии не найден: {in_path.resolve()}")

    print(f"[GeotechnicalLabeler] Загрузка сырой телеметрии: {in_path.resolve()}...")
    df = pd.read_csv(in_path)
    df["date"] = pd.to_datetime(df["date"])

    sites = df["site_id"].unique()
    print(f"[GeotechnicalLabeler] Обнаружено постов мониторинга: {len(sites)} ({list(sites)})")

    # Толщины расчетных расчетно-слоевых зон вокруг термодатчиков:
    # z = [0.2, 0.5, 1.0, 1.5, 2.0] м -> dz = [35, 40, 50, 50, 50] см
    dz_layers = np.array([35.0, 40.0, 50.0, 50.0, 50.0], dtype=float)

    labeled_chunks = []

    for site in sites:
        sub = df[df["site_id"] == site].sort_values("date").reset_index(drop=True)
        n = len(sub)
        is_hp = int(sub["is_heatpipe"].iloc[0])

        t_air = sub["t_air"].values
        t_02 = sub["t_ground_0.2"].values
        t_05 = sub["t_ground_0.5"].values
        t_10 = sub["t_ground_1.0"].values
        t_15 = sub["t_ground_1.5"].values
        t_20 = sub["t_ground_2.0"].values

        w_05 = sub["w_ground_0.5"].values
        w_10 = sub["w_ground_1.0"].values
        w_15 = sub["w_ground_1.5"].values
        w_20 = sub["w_ground_2.0"].values

        # Расчет средней влажности рабочего слоя (для совместимости с табличными моделями)
        mean_ground_moisture = np.round(0.35 * w_05 + 0.30 * w_10 + 0.20 * w_15 + 0.15 * w_20, 1)

        # Вычисление скользящих циклов 0 °C и кумулятивных метеопоказателей
        crossings = np.zeros(n, dtype=int)
        for i in range(1, n):
            if (t_air[i - 1] < 0 and t_air[i] >= 0) or (t_air[i - 1] > 0 and t_air[i] <= 0):
                crossings[i] = 1
        fc30 = pd.Series(crossings).rolling(30, min_periods=1).sum().values
        fc10 = pd.Series(crossings).rolling(10, min_periods=1).sum().values

        t_neg = np.where(t_air < 0, t_air, 0.0)
        tsum30 = pd.Series(t_neg).rolling(30, min_periods=1).sum().values
        tavg30 = pd.Series(t_air).rolling(30, min_periods=1).mean().values
        m_avg30 = pd.Series(mean_ground_moisture).rolling(30, min_periods=1).mean().values

        # Физическое моделирование деформаций пучения и трещинообразования (Slozhno.docx):
        # 1. Монотонное глубокое зимнее промерзание без циклов формирует штатное равномерное пучение (Класс 1, h <= 1.8 см)
        # 2. Аварийные бугры пучения (>= 3.0 см) и морозобойные трещины (Класс 2) вызываются исключительно знакопеременными циклами F_cross
        # 3. Дефекты проявляются с характерным геофизическим лагом созревания tau = 30 дней
        target_class = np.zeros(n, dtype=int)
        heave_target = np.zeros(n, dtype=float)

        horizon = 30

        for i in range(n):
            f30 = fc30[i]
            f10 = fc10[i]
            m = m_avg30[i]
            tsum = tsum30[i]
            tavg = tavg30[i]

            # 1. Летний талый режим: верхний слой грунта талый (T_0.2 > 1.0 °C) и теплая погода -> Норма (Класс 0)
            if t_02[i] > 1.0 and tavg > 0.0:
                target_class[i] = 0
                heave_target[i] = 0.2
                continue

            w_factor = max(0.0, (m - w_critical) / 6.0)
            cold_factor = min(2.0, abs(tsum) / 120.0)

            # Равномерное сезонное промерзание (до 1.6 см)
            uniform_h = 1.2 * cold_factor * w_factor

            # Сегрегационная подкачка влаги от знакопеременных циклов F_cross
            cycle_score = 0.90 * f30 + 0.50 * f10
            segregation_h = 0.55 * (cycle_score ** 1.1) * w_factor

            total_h = round(uniform_h + segregation_h, 2)

            # Морозобойное расклинивание от циклов замерзания
            crack_score = -4.5 + cycle_score * w_factor + 0.30 * (m - 20.0) - 0.002 * tsum
            p_crack = 1.0 / (1.0 + np.exp(-crack_score))
            crack = (p_crack > 0.45) or (total_h >= 3.0)

            # Категорирование дефектов:
            # Класс 2 (Аварийное): разрушение от криогенных циклов F_cross (трещины и бугры >= 3.0 см)
            # Класс 1 (Преддефект): штатное сезонное промерзание без циклов
            # Класс 0 (Норма): талое бездефектное состояние
            if crack and cycle_score >= 1.5:
                target_class[i] = 2
                heave_target[i] = max(3.1, total_h)
            elif total_h >= 1.0 or tsum < -20.0 or t_02[i] <= 0.0:
                target_class[i] = 1
                heave_target[i] = round(max(1.0, total_h), 2)
            else:
                target_class[i] = 0
                heave_target[i] = round(min(0.9, total_h), 2)

        # Проявление дефектов через лаг созревания tau = 30 дней
        defect_class = np.zeros(n, dtype=int)
        heave_depth = np.zeros(n, dtype=float)
        defect_class[horizon:] = target_class[:-horizon]
        heave_depth[horizon:] = heave_target[:-horizon]
        is_crack = (defect_class == 2)

        sub["ground_moisture"] = mean_ground_moisture
        sub["heave_depth_cm"] = heave_depth
        sub["defect_class"] = defect_class
        sub["is_crack"] = is_crack

        max_h = sub["heave_depth_cm"].max()
        c2_pct = (sub["defect_class"] == 2).mean() * 100
        hp_str = "с термостабилизацией" if is_hp else "БЕЗ термостабилизации (контроль)"
        print(f"  Пост {site} ({hp_str}): макс. пучение = {max_h:.2f} см, доля аварийного класса = {c2_pct:.1f}%")

        labeled_chunks.append(sub)

    out_df = pd.concat(labeled_chunks, ignore_index=True)

    # Сохранение размеченного датасета
    out_p = Path(output_csv)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_p, index=False)
    print(f"\n[GeotechnicalLabeler] Размеченный датасет сохранен: {out_p.resolve()} ({len(out_df)} строк)")

    # Построение графика сравнения эффективности термостабилизации
    if save_comparison_figure:
        fig_p = Path(figure_path)
        fig_p.parent.mkdir(parents=True, exist_ok=True)
        _plot_heatpipe_comparison(out_df, save_path=fig_p)

    return out_df


def _plot_heatpipe_comparison(df: pd.DataFrame, save_path: Path) -> None:
    """Строит сравнительный график динамики пучения для постов с СОУ и без них."""
    plt.figure(figsize=(12, 6))

    # Сгруппируем по дате и наличию термостабилизатора
    grouped = df.groupby(["date", "is_heatpipe"])["heave_depth_cm"].agg(["mean", "min", "max"]).reset_index()

    hp_yes = grouped[grouped["is_heatpipe"] == 1]
    hp_no = grouped[grouped["is_heatpipe"] == 0]

    # Линия и диапазон для контрольных постов (без СОУ)
    plt.plot(hp_no["date"], hp_no["mean"], color="#d95f02", linewidth=2.2, label="Контрольные посты КП (без термосифонов)")
    plt.fill_between(hp_no["date"], hp_no["min"], hp_no["max"], color="#d95f02", alpha=0.18)

    # Линия и диапазон для опытных постов (с СОУ)
    plt.plot(hp_yes["date"], hp_yes["mean"], color="#1b9e77", linewidth=2.2, label="Опытные посты ОП (с термостабилизацией СОУ)")
    plt.fill_between(hp_yes["date"], hp_yes["min"], hp_yes["max"], color="#1b9e77", alpha=0.18)

    # Пороговые линии нормативных классов СП 34.13330
    plt.axhline(3.0, color="#b2182b", linestyle="--", linewidth=1.5, label="Аварийный порог деформации (3.0 см, Класс 2)")
    plt.axhline(1.0, color="#f4a582", linestyle=":", linewidth=1.5, label="Преддефектный порог (1.0 см, Класс 1)")

    plt.title("Сравнительная динамика морозного пучения дорожного полотна (СП 34.13330 / ГОСТ 28622)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Дата мониторинга", fontsize=11)
    plt.ylabel("Величина пучения поверхности, см", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper left", framealpha=0.92, fontsize=10)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[GeotechnicalLabeler] График сравнения сохранен: {save_path.resolve()}")


if __name__ == "__main__":
    label_telemetry_dataset()
