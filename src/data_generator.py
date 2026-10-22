"""
Физико-математический генератор синтетических данных для моделирования пучинообразования.

Моделирует:
- Многолетние суточные метеоданные: температура воздуха (с ярко выраженными циклами
  замерзания-оттаивания весной и осенью), снежный покров, влажность.
- Физику накопления криогенных деформаций грунта (пучин) на основе теории сегрегационного
  льдообразования при многократных переходах через 0 °C.
- Журнал мониторинга дефектов дорожного полотна с замерами глубины пучин.
"""

from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd


def generate_weather_and_heave_data(
    start_date: str = "2015-01-01",
    end_date: str = "2024-12-31",
    random_seed: int = 42,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Генерирует согласованный временной ряд погоды и состояния дорожного полотна.

    Параметры:
        start_date: начальная дата (ГГГГ-ММ-ДД).
        end_date: конечная дата (ГГГГ-ММ-ДД).
        random_seed: зерно генератора случайных чисел для воспроизводимости.
        output_dir: директория для сохранения CSV (если указана).

    Возвращает:
        DataFrame со столбцами:
        ['date', 't_air', 'snow_depth', 'precipitation', 'ground_moisture',
         'heave_depth_cm', 'defect_class', 'is_crack']
    """
    rng = np.random.default_rng(random_seed)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    n_days = len(dates)

    day_of_year = dates.dayofyear.values

    # 1. Моделирование температуры воздуха T_air
    # Сезонная гармоника: зима (январь) ~ -10 °C, лето (июль) ~ +20 °C
    # Пик холода ~ 20 января (day 20), пик тепла ~ 20 июля (day 200)
    season_angle = 2 * np.pi * (day_of_year - 200) / 365.25
    t_season = 5.0 + 15.0 * np.cos(season_angle)

    # Синоптические волны тепла и холода (периоды 7-21 день, приносящие регулярные оттепели и заморозки)
    synoptic_waves = (
        5.0 * np.sin(2 * np.pi * np.arange(n_days) / 8.5 + rng.uniform(0, 2 * np.pi))
        + 4.0 * np.cos(2 * np.pi * np.arange(n_days) / 14.2 + rng.uniform(0, 2 * np.pi))
        + 2.5 * np.sin(2 * np.pi * np.arange(n_days) / 21.0 + rng.uniform(0, 2 * np.pi))
    )

    # Высокочастотный шум погоды
    weather_noise = rng.normal(0, 2.0, size=n_days)

    t_air = t_season + synoptic_waves + weather_noise
    t_air = np.round(t_air, 1)

    # 2. Моделирование осадков и снежного покрова
    # Осадки чаще в циклонические дни (когда шум отрицательный или положительный)
    precip_prob = 0.35 + 0.1 * np.sin(season_angle)
    precip_event = rng.uniform(size=n_days) < precip_prob
    precip_amount = np.where(
        precip_event,
        rng.exponential(scale=3.5, size=n_days),
        0.0,
    )

    # Снежный покров накапливается при T < 0 и тает при T > 0
    snow_depth = np.zeros(n_days, dtype=float)
    ground_moisture = np.zeros(n_days, dtype=float)
    curr_snow = 0.0
    curr_moist = 20.0  # процент влажности грунта (~15-35%)

    for i in range(n_days):
        t = t_air[i]
        p = precip_amount[i]

        if t <= 0:
            # Снегопад: 1 мм осадков ~ 1 см свежего снега (с уплотнением)
            curr_snow += p * 0.8
            # Постепенное оседание снега
            curr_snow = max(0.0, curr_snow * 0.99)
            # Влажность в мерзлом состоянии стабилизируется
            curr_moist = max(15.0, curr_moist * 0.999)
        else:
            # Таяние снега: интенсивность зависит от температуры
            melt_rate = 0.7 * t
            melted = min(curr_snow, melt_rate)
            curr_snow = max(0.0, curr_snow - melted)
            # Талая вода и дожди насыщают грунт влагой
            water_inflow = melted * 0.5 + p * 0.4
            curr_moist += water_inflow * 0.3
            # Дренаж и испарение
            curr_moist = 18.0 + (curr_moist - 18.0) * 0.96

        curr_moist = float(np.clip(curr_moist, 12.0, 38.0))
        snow_depth[i] = np.round(curr_snow, 1)
        ground_moisture[i] = np.round(curr_moist, 1)

    # 3. Физика пучинообразования и растрескивания дорожного полотна
    # Фундаментальный геотехнический закон (Slozhno.docx):
    # Доминирующий фактор разрушения дорожного полотна — циклы замерзания-оттаивания (F_cross).
    # Во время оттепелей талая вода проникает в поры асфальтобетона и подстилающего грунта,
    # а при последующем замерзании кристаллизационное давление льда вызывает микроразрывы
    # связей (усталостное криогенное расклинивание) и мощную сегрегационную накачку влаги.
    # В результате кумулятивное воздействие циклов через характерный лаг созревания (tau = 30 дней)
    # приводит к аварийным пучинам (> 3 см) и сквозным трещинам (Класс 2).
    #
    # Равномерное промерзание без циклов вызывает лишь плавное сезонное пучение (< 1-1.5 см),
    # а в теплый (летний) период деформации пучения полностью отсутствуют (Класс 0).

    # Вычисляем скользящие кумулятивные индикаторы по временному ряду
    crossings = np.zeros(n_days, dtype=int)
    for i in range(1, n_days):
        if (t_air[i - 1] < 0 and t_air[i] >= 0) or (t_air[i - 1] > 0 and t_air[i] <= 0):
            crossings[i] = 1

    f_cross_30 = pd.Series(crossings).rolling(30, min_periods=1).sum().values
    f_cross_10 = pd.Series(crossings).rolling(10, min_periods=1).sum().values
    t_neg = np.where(t_air < 0, t_air, 0.0)
    t_sum_30 = pd.Series(t_neg).rolling(30, min_periods=1).sum().values
    t_avg_30 = pd.Series(t_air).rolling(30, min_periods=1).mean().values
    moist_avg_30 = pd.Series(ground_moisture).rolling(30, min_periods=1).mean().values

    horizon = 30
    target_class = np.zeros(n_days, dtype=int)
    heave_target = np.zeros(n_days, dtype=float)

    for i in range(n_days):
        fc30 = f_cross_30[i]
        fc10 = f_cross_10[i]
        m = moist_avg_30[i]
        tsum = t_sum_30[i]
        tavg = t_avg_30[i]

        # Летний режим: отсутствие отрицательных температур и циклов
        if tavg > 5.0 and fc30 == 0:
            target_class[i] = 0
            heave_target[i] = 0.1
            continue

        # Физика криогенного разрушения полотна
        cycle_score = 0.85 * fc30 + 0.55 * fc10
        moist_mult = float(np.clip((m - 16.0) / 7.0, 0.0, 2.5))
        cold_mult = float(np.clip(abs(tsum) / 160.0, 0.0, 2.5))

        # Накопление величины морозного пучения
        heave_val = 0.15 + 0.50 * cycle_score * moist_mult * (0.3 + 0.7 * cold_mult) + 0.40 * cold_mult
        heave_val += rng.normal(0, 0.12)
        heave_val = max(0.0, heave_val)

        # Вероятность трещин и аварийного состояния (Класс 2)
        # Циклы F_cross являются доминирующим разрушающим фактором
        z_c2 = -4.5 + cycle_score * moist_mult + 0.30 * (m - 20.0) - 0.003 * tsum
        p_c2 = 1.0 / (1.0 + np.exp(-z_c2))
        is_c2 = (rng.uniform() < p_c2) or (heave_val >= 3.0)

        if is_c2 and tavg < 12.0:
            target_class[i] = 2
            heave_target[i] = max(3.1, round(heave_val, 2))
        elif heave_val >= 1.0 and tavg < 15.0:
            target_class[i] = 1
            heave_target[i] = round(heave_val, 2)
        else:
            target_class[i] = 0
            heave_target[i] = round(min(0.9, heave_val), 2)

    # Задаем хронологическое проявление дефектов через лаг tau = 30 дней
    defect_class = np.zeros(n_days, dtype=int)
    heave_depth = np.zeros(n_days, dtype=float)
    defect_class[horizon:] = target_class[:-horizon]
    heave_depth[horizon:] = heave_target[:-horizon]
    is_crack = (defect_class == 2)

    df = pd.DataFrame(
        {
            "date": dates,
            "t_air": t_air,
            "snow_depth": snow_depth,
            "precipitation": np.round(precip_amount, 1),
            "ground_moisture": ground_moisture,
            "heave_depth_cm": heave_depth,
            "defect_class": defect_class,
            "is_crack": is_crack,
        }
    )

    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        csv_file = out_path / "simulated_weather_heave_timeseries.csv"
        df.to_csv(csv_file, index=False)
        print(f"[DataGenerator] Датасет сохранен: {csv_file} ({len(df)} записей)")

    return df


if __name__ == "__main__":
    df = generate_weather_and_heave_data(
        start_date="2015-01-01",
        end_date="2024-12-31",
        output_dir="a:/PythonProjects/PuchinaAI/data/synthetic",
    )
    print(df.head(10))
    print("\nРаспределение классов состояния:")
    print(df["defect_class"].value_counts(normalize=True))
