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
    # Сезонная гармоника: зима (январь) ~ -15 °C, лето (июль) ~ +20 °C
    # Пик холода ~ 20 января (day 20), пик тепла ~ 20 июля (day 200)
    season_angle = 2 * np.pi * (day_of_year - 200) / 365.25
    t_season = 5.0 + 17.0 * np.cos(season_angle)

    # Синоптические волны тепла и холода (периоды 5-15 дней)
    synoptic_waves = (
        4.0 * np.sin(2 * np.pi * np.arange(n_days) / 7.3 + rng.uniform(0, 2 * np.pi))
        + 3.0 * np.sin(2 * np.pi * np.arange(n_days) / 13.7 + rng.uniform(0, 2 * np.pi))
    )

    # Высокочастотный шум погоды
    weather_noise = rng.normal(0, 2.5, size=n_days)

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

    # 3. Физика пучинообразования (криогенная деформация дорожного полотна)
    # Пучина возникает при сочетании:
    #   а) Повторных переходов через 0 °C (миграция влаги к фронту промерзания)
    #   б) Накопленного холода (T < 0)
    #   в) Достаточной влажности грунта (> 20%)
    #   г) Отсутствия мощного теплоизолирующего снежного покрова в начале зимы
    heave_depth = np.zeros(n_days, dtype=float)
    accumulated_ice = 0.0

    # Вычислим скользящие индикаторы
    is_freeze_thaw = np.zeros(n_days, dtype=bool)
    for i in range(1, n_days):
        if (t_air[i - 1] < 0 <= t_air[i]) or (t_air[i - 1] > 0 >= t_air[i]):
            is_freeze_thaw[i] = True

    for i in range(n_days):
        t = t_air[i]
        moist = ground_moisture[i]
        snow = snow_depth[i]

        # Если зима/весна: активное промерзание с миграцией влаги
        if t < 0:
            # Интенсивность миграционной накачки льда при цикле оттаивания/замерзания
            cycle_factor = 2.8 if is_freeze_thaw[i] else 0.5
            moist_factor = max(0.1, (moist - 16.0) / 12.0)
            # Теплоизоляция снега: снижает градиент промерзания
            insulation = np.exp(-snow / 28.0)

            daily_heave_growth = 0.085 * (abs(t) ** 0.5) * cycle_factor * moist_factor * insulation
            accumulated_ice += daily_heave_growth
        elif t > 0:
            # Весенне-летняя осадка пучины и консолидация грунта
            thaw_rate = 0.10 * (t ** 0.8)
            accumulated_ice = max(0.0, accumulated_ice - thaw_rate)

        heave_depth[i] = np.round(accumulated_ice, 2)

    # 4. Классификация дефектов дорожного покрытия
    # Класс 0: Норма (< 1 см)
    # Класс 1: Преддефектное (1–3 см)
    # Класс 2: Аварийное (> 3 см или трещина)
    defect_class = np.zeros(n_days, dtype=int)
    is_crack = np.zeros(n_days, dtype=bool)

    for i in range(n_days):
        h = heave_depth[i]
        # Вероятность образования трещины при сильном пучении
        crack_prob = 1.0 / (1.0 + np.exp(-3.5 * (h - 2.5))) if h >= 2.0 else 0.0
        crack = rng.uniform() < crack_prob

        if h >= 3.0 or crack:
            defect_class[i] = 2
            is_crack[i] = crack
        elif h >= 1.0:
            defect_class[i] = 1
        else:
            defect_class[i] = 0

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
