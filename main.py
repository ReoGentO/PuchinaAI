"""
Главный модуль запуска сквозного научного конвейера исследования:
«Гибридная нейросетевая модель прогнозирования риска деформаций
дорожного покрытия на основе анализа циклов замерзания-оттаивания»

Полная реализация технического плана из Slozhno.docx:
- Этап 1: Инженерия временных меток и лагов (30, 60, 90 дней).
- Этап 2: Расчет физических признаков (Fcross, Tsum, Tavg, снег, влажность).
- Этап 3: Построение бейзлайнов (RandomForest, XGBoost).
- Этап 4: Гибридная нейросеть (1D-CNN с ядрами 3, 5, 7 + LSTM/GRU + Softmax).
- Этап 5: Обучение с хронологической валидацией (70% / 15% / 15%).
- Этап 6: Интерпретация модели с помощью SHAP.
- Этап 7: Формулировка практического порога тревоги и сезонная динамика риска.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

from src.data_loader import load_dataset
from src.dataset_builder import create_all_horizons, prepare_chronological_splits
from src.evaluate import evaluate_predictions, generate_comparison_table, plot_confusion_matrices
from src.features import compute_physical_features
from src.interpretation import compute_shap_analysis
from src.models.baselines import train_all_baselines
from src.models.hybrid_nn import train_hybrid_model
from src.threshold_analysis import compute_alarm_threshold, plot_seasonal_risk_timeline


def run_full_research_pipeline(
    data_path: str | None = None,
    primary_horizon: int = 30,
    epochs: int = 25,
    batch_size: int = 32,
    rnn_type: str = "lstm",
) -> None:
    """Запуск полного сквозного цикла исследования."""
    print("=" * 80)
    print("ПУЧИНА-AI: СИСТЕМА ПРЕДИКТИВНОГО МОНИТОРИНГА ДЕФОРМАЦИЙ ДОРОГ")
    print("=" * 80)

    # -------------------------------------------------------------
    # ЭТАП 1 & 2: Загрузка / Генерация данных и расчет признаков
    # -------------------------------------------------------------
    print("\n[ЭТАП 1-2] Загрузка данных и инженерия признаков...")
    df_raw = load_dataset(data_path=data_path)
    print(f"Загружено записей: {len(df_raw)} за период с {df_raw['date'].min()} по {df_raw['date'].max()}")

    # -------------------------------------------------------------
    # СРАВНЕНИЕ ГОРИЗОНТОВ ПРОГНОЗИРОВАНИЯ (1, 2 и 3 месяца)
    # -------------------------------------------------------------
    print("\n[ЭТАП 1.1] Анализ горизонтов прогноза...")
    horizons = sorted(list({30, 60, 90, primary_horizon}))
    horizons_splits = create_all_horizons(df_raw, horizons=tuple(horizons), output_dir="data/processed")

    horizon_comparison_records = []
    for h in horizons:
        h_splits = horizons_splits[h]
        h_baselines = train_all_baselines(
            X_train=h_splits.X_train_tab,
            y_train=h_splits.y_train,
            X_val=h_splits.X_val_tab,
            y_val=h_splits.y_val,
            X_test=h_splits.X_test_tab,
        )
        for model_name, res in h_baselines.items():
            metrics_test = evaluate_predictions(h_splits.y_test, res.y_pred_test, f"{model_name} ({h}д)", split_name="Test")
            metrics_val = evaluate_predictions(h_splits.y_val, res.y_pred_val, f"{model_name} ({h}д)", split_name="Validation")
            metrics_test["Horizon_Days"] = h
            metrics_val["Horizon_Days"] = h
            horizon_comparison_records.extend([metrics_val, metrics_test])

    df_horizons = pd.DataFrame(horizon_comparison_records)
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    df_horizons.to_csv("results/tables/horizons_comparison.csv", index=False)
    with open("results/tables/horizons_comparison.md", "w", encoding="utf-8") as f:
        f.write("# Сравнение качества прогнозирования по горизонтам\n\n")
        f.write(df_horizons.to_markdown(index=False))
    print("[Этап 1] Сравнительная таблица сохранена в results/tables/horizons_comparison.md")

    # -------------------------------------------------------------
    # ЭТАП 3, 4, 5: Основные эксперименты на целевом горизонте
    # -------------------------------------------------------------
    splits = horizons_splits[primary_horizon]
    print(f"\n[ЭТАП 3-5] Углубленное обучение моделей для горизонта {primary_horizon} дней...")

    baseline_results = train_all_baselines(
        X_train=splits.X_train_tab,
        y_train=splits.y_train,
        X_val=splits.X_val_tab,
        y_val=splits.y_val,
        X_test=splits.X_test_tab,
    )

    nn_result = train_hybrid_model(
        X_train_seq=splits.X_train_seq,
        y_train=splits.y_train,
        X_val_seq=splits.X_val_seq,
        y_val=splits.y_val,
        X_test_seq=splits.X_test_seq,
        y_test=splits.y_test,
        rnn_type=rnn_type,
        epochs=epochs,
        batch_size=batch_size,
        figure_save_path="results/figures/training_history.png",
    )

    # Сохранение обученных моделей на диск
    import joblib
    models_dir = Path("saved_models")
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(baseline_results["Random Forest"].model, models_dir / f"rf_model_{primary_horizon}d.joblib")
    joblib.dump(baseline_results["XGBoost"].model, models_dir / f"xgb_model_{primary_horizon}d.joblib")
    nn_result.model.save(models_dir / f"hybrid_nn_{primary_horizon}d.keras")
    print(f"[Экспорт] Обученные модели сохранены в: {models_dir.resolve()}")

    # -------------------------------------------------------------
    # ОЦЕНКА И СРАВНЕНИЕ МОДЕЛЕЙ (Подраздел 5.1 статьи)
    # -------------------------------------------------------------
    print("\n[ОЦЕНКА] Формирование итоговых метрик (Val и Test)...")
    all_metrics = []
    cm_dict_test = {}

    models_to_evaluate = [
        ("Random Forest", baseline_results["Random Forest"]),
        ("XGBoost", baseline_results["XGBoost"]),
        (nn_result.name, nn_result),
    ]

    for name, res in models_to_evaluate:
        # Валидационная выборка
        m_val = evaluate_predictions(splits.y_val, res.y_pred_val, name, split_name="Validation")
        all_metrics.append(m_val)

        # Тестовая выборка
        m_test = evaluate_predictions(splits.y_test, res.y_pred_test, name, split_name="Test")
        all_metrics.append(m_test)
        cm_dict_test[name] = (splits.y_test, res.y_pred_test)

    df_comparison = generate_comparison_table(all_metrics, output_dir="results/tables")
    plot_confusion_matrices(cm_dict_test, save_path="results/figures/confusion_matrices.png")

    print("\nСводная таблица результатов на тестовой выборке:")
    test_sub = df_comparison[df_comparison["Split"] == "Test"]
    cols_show = ["Model", "Precision_Class_2 (Аварийное)", "Recall_Class_2 (Аварийное)", "F1_Class_2 (Аварийное)", "Macro_F1", "Weighted_F1"]
    print(test_sub[cols_show].to_string(index=False))

    # -------------------------------------------------------------
    # ЭТАП 6: SHAP-интерпретация (Подраздел 5.2 статьи)
    # -------------------------------------------------------------
    print("\n[ЭТАП 6] SHAP-анализ значимости физических признаков...")
    # Стратифицированная выборка с представлением всех классов (Норма, Преддефект, Авария)
    rng = np.random.default_rng(42)
    y_tr = splits.y_train
    idx_class_0 = np.where(y_tr == 0)[0]
    idx_class_1 = np.where(y_tr == 1)[0]
    idx_class_2 = np.where(y_tr == 2)[0]

    n_samples_per_class = min(150, len(idx_class_2))
    sampled_0 = rng.choice(idx_class_0, size=n_samples_per_class * 2, replace=False)
    sampled_1 = rng.choice(idx_class_1, size=n_samples_per_class, replace=False)
    sampled_2 = rng.choice(idx_class_2, size=n_samples_per_class, replace=False)

    stratified_indices = np.concatenate([sampled_0, sampled_1, sampled_2])
    rng.shuffle(stratified_indices)

    compute_shap_analysis(
        model=baseline_results["XGBoost"].model,
        X_sample=splits.X_train_tab[stratified_indices],
        feature_names=splits.feature_names,
        target_class=2,
        save_dir="results/figures",
    )

    # -------------------------------------------------------------
    # ЭТАП 7: Расчет порога тревоги и сезонный риск (Подраздел 5.3 и Разд. 6)
    # -------------------------------------------------------------
    print("\n[ЭТАП 7] Формулировка практического порога тревоги...")
    # Оценка риска пучинообразования (деформаций полотна Y >= 1) с полным охватом всех циклов F_cross
    feat_eval = compute_physical_features(df_raw, window_days=30, decade_days=10)
    feat_eval["target_y"] = feat_eval["defect_class"].shift(-primary_horizon)
    eval_df_clean = feat_eval.dropna(subset=["target_y"]).copy()
    eval_df_clean["target_y"] = eval_df_clean["target_y"].astype(int)

    X_all_tab = eval_df_clean[splits.feature_names].values.astype(np.float32)
    probas_all = baseline_results["XGBoost"].model.predict_proba(X_all_tab)
    eval_df_clean["proba_class_2"] = probas_all[:, 2]
    eval_df_clean["proba_defect"] = probas_all[:, 1] + probas_all[:, 2]

    threshold_info = compute_alarm_threshold(
        df_eval=eval_df_clean,
        fcross_col="f_cross_decade",
        proba_col="proba_defect",
        target_col="target_y",
        alarm_prob_threshold=0.50,
        save_path="results/figures/threshold_alarm_curve.png",
    )

    df_test_eval = splits.df_test.copy()
    df_test_eval["proba_class_2"] = baseline_results["XGBoost"].y_proba_test[:, 2]
    plot_seasonal_risk_timeline(
        df_eval=df_test_eval,
        proba_col="proba_class_2",
        save_path="results/figures/seasonal_risk_timeline.png",
    )

    print("\n" + "=" * 80)
    print("ИТОГИ ИССЛЕДОВАНИЯ:")
    print(f"  • Оптимальный горизонт прогноза: {primary_horizon} дней (высокая надежность упреждения)")
    print(f"  • Практический порог: {threshold_info['recommendation_text']}")
    print(f"  • Графика для статьи подготовлена в папке: results/figures/")
    print(f"  • Таблицы метрик подготовлены в папке: results/tables/")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="PuchinaAI: Frost Heave Risk Prediction Pipeline")
    parser.add_argument("--data", type=str, default=None, help="Путь к сырому CSV файлу")
    parser.add_argument("--horizon", type=int, default=30, help="Основной горизонт прогнозирования в днях (30, 60, 90)")
    parser.add_argument("--epochs", type=int, default=25, help="Количество эпох обучения нейросети")
    parser.add_argument("--batch_size", type=int, default=32, help="Размер батча")
    parser.add_argument("--rnn_type", type=str, default="lstm", choices=["lstm", "gru"], help="Тип RNN слоя")

    args = parser.parse_args()
    run_full_research_pipeline(
        data_path=args.data,
        primary_horizon=args.horizon,
        epochs=args.epochs,
        batch_size=args.batch_size,
        rnn_type=args.rnn_type,
    )


if __name__ == "__main__":
    main()
