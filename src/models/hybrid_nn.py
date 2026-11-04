"""
Модуль гибридной нейросетевой модели CNN + BiLSTM + ZeroCrossingAttention (Этап 4 Roadmap).

Архитектура:
Вход: (Batch, seq_length, 7 каналов физических рядов: T_air, daily_cycle_flag, zero_cross_proximity, ...).
1. Мультимасштабный сверточный блок 1D-CNN (ядра k=3, 5, 7) для локальных градиентов температуры.
2. Слой нормализации (BatchNormalization) и SpatialDropout.
3. Рекуррентный слой Bidirectional LSTM (или GRU) для моделирования динамики замерзания.
4. Специализированный механизм внимания (ZeroCrossingAttention), направляющий максимальный вес внимания
   на моменты знакопеременных переходов температуры через 0 °C (криогенные циклы).
5. Полносвязная классификационная голова Dense + Dropout + Softmax (3 класса).
6. Визуализация весов внимания и подтверждение фокуса на переходах 0 °C.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@keras.utils.register_keras_serializable(package="PuchinaAI")
class ZeroCrossingAttention(layers.Layer):
    """Механизм направленного временного внимания к переходам температуры через 0 °C."""

    def __init__(self, units: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        h_shape, c_shape = input_shape
        self.W = self.add_weight(
            shape=(h_shape[-1], self.units),
            initializer="glorot_uniform",
            name="att_W",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer="zeros",
            name="att_b",
            trainable=True,
        )
        self.v = self.add_weight(
            shape=(self.units, 1),
            initializer="glorot_uniform",
            name="att_v",
            trainable=True,
        )
        # Обучаемые веса приоритета циклов замерзания и термической близости к 0 °C
        self.cycle_weight = self.add_weight(
            shape=(1,),
            initializer=keras.initializers.Constant(2.0),
            name="cycle_w",
            trainable=True,
        )
        self.prox_weight = self.add_weight(
            shape=(1,),
            initializer=keras.initializers.Constant(1.0),
            name="prox_w",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        h, cycle_signals = inputs
        cycle_flag = cycle_signals[:, :, 0:1]
        prox_val = cycle_signals[:, :, 1:2]

        # Рекуррентная компонента энергии внимания
        score = tf.matmul(tf.tanh(tf.matmul(h, self.W) + self.b), self.v)

        # Аддитивный физический буст для точек перехода через 0 °C
        boost = tf.nn.softplus(self.cycle_weight) * cycle_flag + tf.nn.softplus(self.prox_weight) * prox_val
        total_score = score + boost

        # Нормализация Softmax по временной шкале последовательности
        weights = tf.nn.softmax(total_score, axis=1)

        # Контекстный вектор — взвешенная сумма скрытых состояний BiLSTM
        context = tf.reduce_sum(weights * h, axis=1)
        return context, weights

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


@dataclass
class HybridNNResult:
    name: str
    model: keras.Model
    attention_model: keras.Model
    y_pred_val: np.ndarray
    y_proba_val: np.ndarray
    y_pred_test: np.ndarray
    y_proba_test: np.ndarray
    history: dict[str, list[float]]
    attention_cycle_ratio: float


def build_hybrid_cnn_lstm(
    seq_length: int = 45,
    num_features: int = 7,
    num_classes: int = 3,
    rnn_type: Literal["lstm", "gru"] = "lstm",
    learning_rate: float = 0.001,
) -> tuple[keras.Model, keras.Model]:
    """Создает архитектуру CNN + BiLSTM + ZeroCrossingAttention."""
    inputs = layers.Input(shape=(seq_length, num_features), name="physical_sequence_input")

    # Выделение физических каналов переходов через 0 °C:
    # канал 1: daily_cycle_flag, канал 2: zero_cross_proximity
    cycle_signals = inputs[:, :, 1:3]

    # 1. Мультимасштабный блок 1D-CNN (ядра k=3, 5, 7)
    c3 = layers.Conv1D(filters=32, kernel_size=3, padding="same", activation="relu", name="conv1d_k3")(inputs)
    c5 = layers.Conv1D(filters=32, kernel_size=5, padding="same", activation="relu", name="conv1d_k5")(inputs)
    c7 = layers.Conv1D(filters=32, kernel_size=7, padding="same", activation="relu", name="conv1d_k7")(inputs)

    merged = layers.Concatenate(name="concat_multiscale_cnn")([c3, c5, c7])
    x = layers.BatchNormalization(name="batch_norm")(merged)
    x = layers.SpatialDropout1D(0.15, name="spatial_dropout")(x)

    # 2. Рекуррентный блок BiLSTM / BiGRU с возвратом всей последовательности для Attention
    if rnn_type.lower() == "gru":
        rnn_core = layers.GRU(units=64, return_sequences=True, name="gru_core")
    else:
        rnn_core = layers.LSTM(units=64, return_sequences=True, name="lstm_core")

    rnn_seq = layers.Bidirectional(rnn_core, name=f"bidirectional_{rnn_type.lower()}")(x)

    # 3. Механизм внимания, сфокусированный на переходах 0 °C
    att_layer = ZeroCrossingAttention(units=64, name="zero_crossing_attention")
    context, att_weights = att_layer([rnn_seq, cycle_signals])

    # 4. Классификационная полносвязная голова
    d1 = layers.Dense(64, activation="relu", name="dense_features_1")(context)
    d1 = layers.Dropout(0.25, name="dropout_dense_1")(d1)
    d2 = layers.Dense(32, activation="relu", name="dense_features_2")(d1)
    outputs = layers.Dense(num_classes, activation="softmax", name="class_probabilities")(d2)

    model = keras.Model(inputs=inputs, outputs=outputs, name="Hybrid_CNN_BiLSTM_Attention")
    att_model = keras.Model(inputs=inputs, outputs=att_weights, name="Attention_Extractor")

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model, att_model


def train_hybrid_model(
    X_train_seq: np.ndarray,
    y_train: np.ndarray,
    X_val_seq: np.ndarray,
    y_val: np.ndarray,
    X_test_seq: np.ndarray,
    y_test: np.ndarray,
    rnn_type: Literal["lstm", "gru"] = "lstm",
    epochs: int = 40,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    figure_save_path: str | Path = "results/figures/training_history.png",
    attention_save_path: str | Path = "results/figures/attention_weights_zero_crossings.png",
    df_test: any = None,
) -> HybridNNResult:
    """Обучает гибридную нейросеть, строит графики обучения и визуализирует внимание к 0 °C."""
    print(f"[HybridNN] Сборка и обучение нейросети CNN+Bi{rnn_type.upper()}+ZeroCrossingAttention...")
    seq_length = X_train_seq.shape[1]
    num_features = X_train_seq.shape[2]
    num_classes = 3

    y_train_cat = keras.utils.to_categorical(y_train, num_classes=num_classes)
    y_val_cat = keras.utils.to_categorical(y_val, num_classes=num_classes)

    unique_classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=unique_classes, y=y_train)
    class_weight_dict = {cls: float(w) for cls, w in zip(unique_classes, weights)}
    print(f"[HybridNN] Веса классов: {class_weight_dict}")

    model, att_model = build_hybrid_cnn_lstm(
        seq_length=seq_length,
        num_features=num_features,
        num_classes=num_classes,
        rnn_type=rnn_type,
        learning_rate=learning_rate,
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=14,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-5,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train_seq,
        y_train_cat,
        validation_data=(X_val_seq, y_val_cat),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1,
    )

    # Прогнозы вероятностей и меток
    y_proba_val = model.predict(X_val_seq, verbose=0)
    y_pred_val = np.argmax(y_proba_val, axis=1)

    y_proba_test = model.predict(X_test_seq, verbose=0)
    y_pred_test = np.argmax(y_proba_test, axis=1)

    # Извлечение весов внимания на тестовой выборке
    test_attention = att_model.predict(X_test_seq, verbose=0)  # Shape: (N_test, seq_length, 1)

    # Расчет концентрации внимания на днях с переходами через 0 °C
    cycle_flags_test = X_test_seq[:, :, 1]  # канал 1 = daily_cycle_flag
    att_cycles = test_attention[cycle_flags_test == 1]
    att_non_cycles = test_attention[cycle_flags_test == 0]

    mean_cycle_att = float(att_cycles.mean()) if len(att_cycles) > 0 else 0.0
    mean_non_cycle_att = float(att_non_cycles.mean()) if len(att_non_cycles) > 0 else 1.0
    ratio = (mean_cycle_att / mean_non_cycle_att) if mean_non_cycle_att > 0 else 1.0

    print(f"[HybridNN] Анализ весов внимания к переходам через 0 °C:")
    print(f"  • Средний вес внимания в дни циклов 0 °C: {mean_cycle_att:.4f}")
    print(f"  • Средний вес внимания в дни без циклов:   {mean_non_cycle_att:.4f}")
    print(f"  • Фокус внимания: в {ratio:.2f} раз выше на циклах замерзания-оттаивания!")

    # Построение графиков
    _plot_history(history.history, figure_save_path)
    _plot_attention_analysis(
        X_test_seq=X_test_seq,
        att_weights=test_attention,
        mean_cycle_att=mean_cycle_att,
        mean_non_cycle_att=mean_non_cycle_att,
        ratio=ratio,
        save_path=attention_save_path,
    )

    return HybridNNResult(
        name=f"Hybrid CNN+Bi{rnn_type.upper()}+Att",
        model=model,
        attention_model=att_model,
        y_pred_val=y_pred_val,
        y_proba_val=y_proba_val,
        y_pred_test=y_pred_test,
        y_proba_test=y_proba_test,
        history=history.history,
        attention_cycle_ratio=ratio,
    )


def _plot_history(history: dict[str, list[float]], save_path: str | Path) -> None:
    """Строит графики loss и accuracy в процессе обучения."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(history.get("loss", []), label="Train Loss", color="royalblue", lw=2)
    ax1.plot(history.get("val_loss", []), label="Val Loss", color="darkorange", lw=2, linestyle="--")
    ax1.set_title("Динамика функции потерь (Loss)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Эпоха")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    ax2.plot(history.get("accuracy", []), label="Train Acc", color="royalblue", lw=2)
    ax2.plot(history.get("val_accuracy", []), label="Val Acc", color="darkorange", lw=2, linestyle="--")
    ax2.set_title("Динамика точности (Accuracy)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Эпоха")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(p, dpi=300)
    plt.close()
    print(f"[HybridNN] График истории сохранен: {p.resolve()}")


def _plot_attention_analysis(
    X_test_seq: np.ndarray,
    att_weights: np.ndarray,
    mean_cycle_att: float,
    mean_non_cycle_att: float,
    ratio: float,
    save_path: str | Path,
) -> None:
    """Строит график распределения весов внимания к переходам через 0 °C."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [2.2, 1.0]})

    # Выбираем сэмпл из тестовой выборки, содержащий переходы через 0 °C
    cycle_counts = X_test_seq[:, :, 1].sum(axis=1)
    cand_idx = np.where(cycle_counts >= 3)[0]
    sample_idx = cand_idx[0] if len(cand_idx) > 0 else 0

    t_air = X_test_seq[sample_idx, :, 0] * 20.0
    cycles = X_test_seq[sample_idx, :, 1]
    sample_att = att_weights[sample_idx, :, 0]
    days = np.arange(1, len(t_air) + 1)

    # Левый график: наложение внимания на температурную кривую
    color_temp = "#1f77b4"
    ax1.plot(days, t_air, color=color_temp, lw=2.0, label="Температура воздуха T_air (°C)")
    ax1.axhline(0, color="gray", linestyle="--", alpha=0.7, label="Точка перехода 0 °C")
    ax1.set_xlabel("День скользящего окна последовательности (t)", fontsize=11)
    ax1.set_ylabel("T_air, °C", color=color_temp, fontsize=11)
    ax1.tick_params(axis="y", labelcolor=color_temp)

    # Вторая ось для весов внимания
    ax1_att = ax1.twinx()
    color_att = "#d62728"
    ax1_att.plot(days, sample_att, color=color_att, lw=2.4, marker="o", markersize=4, label="Вес внимания α_t (ZeroCrossingAttention)")
    ax1_att.fill_between(days, sample_att, color=color_att, alpha=0.20)
    ax1_att.set_ylabel("Вес внимания (Attention Weight)", color=color_att, fontsize=11)
    ax1_att.tick_params(axis="y", labelcolor=color_att)

    # Подсветка дней с циклами 0 °C
    for d_idx in np.where(cycles == 1)[0]:
        ax1.axvspan(d_idx + 0.6, d_idx + 1.4, color="gold", alpha=0.45, label="Цикл 0 °C (F_cross)" if d_idx == np.where(cycles == 1)[0][0] else "")

    ax1.set_title("Фокусировка внимания нейросети на переходах через 0 °C", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.5)

    # Сборка легенды
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_att.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", framealpha=0.9, fontsize=9)

    # Правый график: гистограмма сравнения среднего внимания
    bars = ax2.bar(
        ["Дни циклов 0 °C\n(Freeze-Thaw)", "Обычные дни\n(Без перехода)"],
        [mean_cycle_att, mean_non_cycle_att],
        color=["#d62728", "#1f77b4"],
        width=0.55,
        alpha=0.85,
    )
    ax2.set_ylabel("Средний вес внимания", fontsize=11)
    ax2.set_title(f"Концентрация внимания: в {ratio:.1f}x выше", fontsize=12, fontweight="bold")
    ax2.grid(True, axis="y", linestyle=":", alpha=0.5)

    for bar in bars:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.005, f"{yval:.3f}", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(p, dpi=300)
    plt.close()
    print(f"[HybridNN] График распределения внимания сохранен: {p.resolve()}")
