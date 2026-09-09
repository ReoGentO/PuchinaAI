"""
Тесты на физическую правдоподобность (Physical Plausibility & Sanity Checks).

Проверяют соответствие поведения обученных моделей фундаментальным законам
геотехники и механики криогенных процессов:
1. Тест «Летний сценарий»: отсутствие риска пучения при положительных температурах.
2. Тест «Влагонасыщение грунта»: резкий рост риска аварийности при переувлажнении (W > 25%).
3. Тест «Сумма отрицательных температур Tsum»: монотонный рост риска с ростом суровости промерзания.
4. Тест устойчивости к шуму датчиков (Perturbation stability).
5. Тест строгого соблюдения аксиом теории вероятностей.
6. Тест гибридной нейросети CNN+LSTM на синтетических сезонных временных рядах.
"""

from __future__ import annotations
from pathlib import Path
import unittest
import joblib
import numpy as np
from tensorflow import keras


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

    def test_01_summer_scenario_zero_risk(self):
        """Правдоподобность: летом (T=+22.5 °C, снега нет, Tsum=0) риск пучения обязан быть нулевым."""
        day_sin = float(np.sin(2 * np.pi * 200 / 365.25))
        day_cos = float(np.cos(2 * np.pi * 200 / 365.25))
        summer_sample = np.array([[0.0, 0.0, 0.0, 22.5, 0.0, 0.0, 18.0, day_sin, day_cos]], dtype=np.float32)

        for model_name, model in [("XGBoost", self.xgb_model), ("Random Forest", self.rf_model)]:
            probas = model.predict_proba(summer_sample)[0]
            p_normal, p_emergency = probas[0], probas[2]

            self.assertGreater(
                p_normal,
                0.90,
                f"[{model_name}] Летом вероятность нормы должна быть > 90%, получено: {p_normal * 100:.1f}%",
            )
            self.assertLess(
                p_emergency,
                0.01,
                f"[{model_name}] Летом риск аварийной пучины должен быть < 1%, получено: {p_emergency * 100:.1f}%",
            )

    def test_02_soil_moisture_sensitivity(self):
        """Правдоподобность: рост влажности грунта (с 15% до 30%) резко увеличивает риск аварийности."""
        day_sin = float(np.sin(2 * np.pi * 40 / 365.25))
        day_cos = float(np.cos(2 * np.pi * 40 / 365.25))

        # Сухой грунт (15%) против переувлажненного (30%) при одинаковом зимнем морозе
        dry_ground = np.array([[2.0, 1.0, -150.0, -4.0, 20.0, 30.0, 15.0, day_sin, day_cos]], dtype=np.float32)
        wet_ground = np.array([[2.0, 1.0, -150.0, -4.0, 20.0, 30.0, 30.0, day_sin, day_cos]], dtype=np.float32)

        p_dry = self.xgb_model.predict_proba(dry_ground)[0]
        p_wet = self.xgb_model.predict_proba(wet_ground)[0]

        # При переувлажнении риск аварийного класса (P2) должен значительно вырасти
        self.assertGreater(
            p_wet[2],
            p_dry[2] + 0.40,
            f"Рост влажности грунта обязан активировать аварийный класс: P2(сухой)={p_dry[2]:.2f}, P2(влажный)={p_wet[2]:.2f}",
        )
        self.assertLess(
            p_wet[0],
            p_dry[0],
            f"При переувлажнении доля нормы обязана снизиться: P0(сухой)={p_dry[0]:.2f}, P0(влажный)={p_wet[0]:.2f}",
        )

    def test_03_freezing_degree_days_accumulation(self):
        """Правдоподобность: рост накопленного холода (T_sum) усиливает риск аварийного состояния."""
        day_sin = float(np.sin(2 * np.pi * 40 / 365.25))
        day_cos = float(np.cos(2 * np.pi * 40 / 365.25))

        # Слабый холод (Tsum=-50) vs Глубокое промерзание (Tsum=-350)
        mild_frost = np.array([[2.0, 1.0, -50.0, -1.6, 20.0, 30.0, 24.0, day_sin, day_cos]], dtype=np.float32)
        deep_frost = np.array([[2.0, 1.0, -350.0, -11.6, 20.0, 30.0, 24.0, day_sin, day_cos]], dtype=np.float32)

        p_mild = self.xgb_model.predict_proba(mild_frost)[0]
        p_deep = self.xgb_model.predict_proba(deep_frost)[0]

        self.assertLess(
            p_deep[0],
            p_mild[0],
            f"При глубоком промерзании норма должна быть ниже: P0(mild)={p_mild[0]:.3f}, P0(deep)={p_deep[0]:.3f}",
        )

    def test_04_prediction_stability_under_perturbation(self):
        """Правдоподобность: малый шум во входных данных (+-0.05) не вызывает стохастических скачков."""
        base_sample = np.array([[3.0, 2.0, -180.0, -5.0, 25.0, 35.0, 22.0, 0.5, 0.8]], dtype=np.float32)
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
            0.08,
            f"Модель неустойчива к малому шуму: максимальный сдвиг вероятностей = {max_shift:.3f} > 0.08",
        )

    def test_05_probability_axioms_hold(self):
        """Правдоподобность: для любых входов сумма вероятностей строго равна 1.0, значения >= 0."""
        rng = np.random.default_rng(100)
        random_inputs = rng.uniform(
            low=[0, 0, -500, -25, 0, 0, 15, -1, -1],
            high=[12, 8, 0, 25, 80, 100, 35, 1, 1],
            size=(50, 9),
        ).astype(np.float32)

        for model_name, model in [("XGBoost", self.xgb_model), ("Random Forest", self.rf_model)]:
            probas = model.predict_proba(random_inputs)
            sums = np.sum(probas, axis=1)
            np.testing.assert_allclose(sums, np.ones(50), atol=1e-5, err_msg=f"[{model_name}] Сумма вероятностей != 1.0")
            self.assertTrue(np.all(probas >= -1e-6), f"[{model_name}] Обнаружены отрицательные вероятности")

    def test_06_hybrid_cnn_lstm_plausibility(self):
        """Правдоподобность CNN+LSTM: реакция на летнюю и зимнюю последовательности."""
        if self.nn_model is None:
            self.skipTest("Модель hybrid_nn_30d.keras не найдена.")

        summer_seq = np.zeros((1, 60, 3), dtype=np.float32)
        summer_seq[0, :, 0] = np.linspace(18.0, 24.0, 60)
        summer_seq[0, :, 2] = np.linspace(0.45, 0.60, 60)
        p_summer = self.nn_model.predict(summer_seq, verbose=0)[0]

        winter_seq = np.zeros((1, 60, 3), dtype=np.float32)
        winter_seq[0, :, 0] = -5.0 + 7.0 * np.sin(np.linspace(0, 8 * np.pi, 60))
        winter_seq[0, :, 1] = 40.0
        winter_seq[0, :, 2] = np.linspace(0.05, 0.20, 60)
        p_winter = self.nn_model.predict(winter_seq, verbose=0)[0]

        self.assertGreater(
            p_summer[0],
            p_winter[0],
            f"CNN+LSTM: вероятность нормы летом ({p_summer[0]:.2f}) должна быть выше, чем зимой ({p_winter[0]:.2f})",
        )
        self.assertGreater(
            p_summer[0],
            0.85,
            f"CNN+LSTM: летом норма должна быть > 85%, получено: {p_summer[0]*100:.1f}%",
        )


if __name__ == "__main__":
    unittest.main()
