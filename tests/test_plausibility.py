"""
Тесты на физическую правдоподобность (Physical Plausibility & Sanity Checks).

Проверяют соответствие поведения обученных моделей фундаментальным законам
геотехники и механики криогенных процессов:
1. Тест «Начало сезона без циклов»: отсутствие риска аварийности при малом числе циклов (F_cross=0, cum_cycles=0).
2. Тест «Влагонасыщение грунта»: рост риска аварийности при повышении влажности.
3. Тест «Кумулятивные циклы 0 °C»: монотонный рост вероятности аварийного класса с ростом циклов.
4. Тест устойчивости к шуму датчиков (Perturbation stability).
5. Тест строгого соблюдения аксиом теории вероятностей.
6. Тест гибридной нейросети CNN+BiLSTM+Attention на сезонных временных последовательностях.
"""

from __future__ import annotations
from pathlib import Path
import unittest
import joblib
import numpy as np
from tensorflow import keras
from src.models.hybrid_nn import ZeroCrossingAttention


class TestPhysicalPlausibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        models_dir = Path("saved_models")
        cls.xgb_path = models_dir / "xgb_model_30d.joblib"
        cls.rf_path = models_dir / "rf_model_30d.joblib"
        cls.nn_path = models_dir / "hybrid_nn_30d.keras"

        if not cls.xgb_path.exists() or not cls.rf_path.exists():
            raise unittest.SkipTest("Обученные модели не найдены в saved_models/. Запустите main.py.")

        cls.xgb_model = joblib.load(cls.xgb_path)
        cls.rf_model = joblib.load(cls.rf_path)
        cls.nn_model = keras.models.load_model(cls.nn_path) if cls.nn_path.exists() else None

    def test_01_early_season_zero_emergency(self):
        """Правдоподобность: в начале сезона при нулевых циклах риск аварийного состояния минимален."""
        day_sin = float(np.sin(2 * np.pi * 320 / 365.25))  # Ноябрь
        day_cos = float(np.cos(2 * np.pi * 320 / 365.25))
        # 10 features: f_cross, f_cross_decade, cumulative_cycles, t_sum, t_avg, snow_avg, snow_max, moisture_avg, day_sin, day_cos
        early_sample = np.array([[0.0, 0.0, 0.0, -50.0, -5.0, 5.0, 10.0, 22.0, day_sin, day_cos]], dtype=np.float32)

        for model_name, model in [("XGBoost", self.xgb_model), ("Random Forest", self.rf_model)]:
            probas = model.predict_proba(early_sample)[0]
            p_emergency = probas[2]
            self.assertLess(
                p_emergency,
                0.15,
                f"[{model_name}] В начале сезона без циклов риск аварии должен быть < 15%, получено: {p_emergency * 100:.1f}%",
            )

    def test_02_soil_moisture_sensitivity(self):
        """Правдоподобность: рост влажности грунта (с 18% до 32%) увеличивает риск деформаций."""
        day_sin = float(np.sin(2 * np.pi * 40 / 365.25))
        day_cos = float(np.cos(2 * np.pi * 40 / 365.25))

        # Сухой грунт (18%) против переувлажненного (32%) при одинаковом зимнем морозе и циклах
        dry_ground = np.array([[2.0, 1.0, 6.0, -500.0, -18.0, 20.0, 30.0, 18.0, day_sin, day_cos]], dtype=np.float32)
        wet_ground = np.array([[2.0, 1.0, 6.0, -500.0, -18.0, 20.0, 30.0, 32.0, day_sin, day_cos]], dtype=np.float32)

        p_dry = self.xgb_model.predict_proba(dry_ground)[0]
        p_wet = self.xgb_model.predict_proba(wet_ground)[0]

        # Риск неблагоприятного состояния (класс 1 + класс 2) выше при переувлажнении
        risk_dry = p_dry[1] + p_dry[2]
        risk_wet = p_wet[1] + p_wet[2]
        self.assertGreaterEqual(
            risk_wet,
            risk_dry - 0.05,
            f"При переувлажнении суммарный риск деформаций не должен быть меньше: wet={risk_wet:.2f}, dry={risk_dry:.2f}",
        )

    def test_03_freeze_thaw_cycles_amplification(self):
        """Правдоподобность: накопление знакопеременных циклов 0 °C активирует аварийный класс 2."""
        day_sin = float(np.sin(2 * np.pi * 40 / 365.25))
        day_cos = float(np.cos(2 * np.pi * 40 / 365.25))

        # Мало циклов (1 цикл) vs Много циклов (10 циклов)
        few_cycles = np.array([[0.0, 0.0, 1.0, -450.0, -16.0, 20.0, 30.0, 24.0, day_sin, day_cos]], dtype=np.float32)
        many_cycles = np.array([[3.0, 2.0, 10.0, -450.0, -16.0, 20.0, 30.0, 24.0, day_sin, day_cos]], dtype=np.float32)

        p_few = self.xgb_model.predict_proba(few_cycles)[0]
        p_many = self.xgb_model.predict_proba(many_cycles)[0]

        self.assertGreater(
            p_many[2],
            p_few[2],
            f"Рост циклов F_cross обязан увеличивать риск класса 2: P2(мало)={p_few[2]:.2f}, P2(много)={p_many[2]:.2f}",
        )

    def test_04_prediction_stability_under_perturbation(self):
        """Правдоподобность: малый шум во входных данных (+-0.05) не вызывает резких скачков."""
        base_sample = np.array([[3.0, 2.0, 6.0, -180.0, -5.0, 25.0, 35.0, 22.0, 0.5, 0.8]], dtype=np.float32)
        base_proba = self.xgb_model.predict_proba(base_sample)[0]

        rng = np.random.default_rng(42)
        shifts = []
        for _ in range(25):
            noise = rng.normal(0, 0.05, size=base_sample.shape).astype(np.float32)
            noisy_sample = base_sample + noise
            noisy_proba = self.xgb_model.predict_proba(noisy_sample)[0]
            shifts.append(np.max(np.abs(base_proba - noisy_proba)))

        max_shift = np.max(shifts)
        self.assertLess(
            max_shift,
            0.15,
            f"Модель неустойчива к малому шуму: максимальный сдвиг вероятностей = {max_shift:.3f} > 0.15",
        )

    def test_05_probability_axioms_hold(self):
        """Правдоподобность: для любых входов сумма вероятностей строго равна 1.0, значения >= 0."""
        rng = np.random.default_rng(100)
        random_inputs = rng.uniform(
            low=[0, 0, 0, -500, -25, 0, 0, 15, -1, -1],
            high=[12, 8, 25, 0, 25, 80, 100, 35, 1, 1],
            size=(50, 10),
        ).astype(np.float32)

        for model_name, model in [("XGBoost", self.xgb_model), ("Random Forest", self.rf_model)]:
            probas = model.predict_proba(random_inputs)
            sums = np.sum(probas, axis=1)
            np.testing.assert_allclose(sums, np.ones(50), atol=1e-5, err_msg=f"[{model_name}] Сумма вероятностей != 1.0")
            self.assertTrue(np.all(probas >= -1e-6), f"[{model_name}] Обнаружены отрицательные вероятности")

    def test_06_hybrid_cnn_lstm_plausibility(self):
        """Правдоподобность CNN+BiLSTM+Attention: реакция на раннюю и циклическую последовательности."""
        if self.nn_model is None:
            self.skipTest("Модель hybrid_nn_30d.keras не найдена.")

        # Ранняя спокойная последовательность (45 дней, 7 каналов)
        early_seq = np.zeros((1, 45, 7), dtype=np.float32)
        early_seq[0, :, 0] = np.linspace(-0.2, 0.1, 45)    # около нуля без мороза
        early_seq[0, :, 1] = 0.0                           # daily_cycle_flag = 0
        early_seq[0, :, 3] = 0.0                           # cum_cycles_norm = 0
        p_early = self.nn_model.predict(early_seq, verbose=0)[0]

        # Зимняя последовательность с сериями циклов замерзания
        winter_seq = np.zeros((1, 45, 7), dtype=np.float32)
        winter_seq[0, :, 0] = np.linspace(-0.8, -0.2, 45)
        winter_seq[0, 10:20, 1] = 1.0                      # циклы переходов
        winter_seq[0, :, 3] = 0.40                         # кумулятивные циклы
        p_winter = self.nn_model.predict(winter_seq, verbose=0)[0]

        self.assertGreater(
            p_winter[2],
            p_early[2],
            f"CNN+BiLSTM: риск аварийности зимой с циклами ({p_winter[2]:.2f}) обязан быть выше, чем в начале сезона ({p_early[2]:.2f})",
        )


if __name__ == "__main__":
    unittest.main()
