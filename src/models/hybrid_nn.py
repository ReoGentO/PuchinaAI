"""
Модуль гибридной нейросетевой модели CNN + LSTM (Этап 4 Roadmap).

Архитектура:
Вход: (Batch, 60 дней, 3 признака: T_air, снег, день_года).
1. Мультимасштабный сверточный блок 1D-CNN с ядрами 3, 5, 7 для захвата
   локальных перепадов температуры и циклов замерзания/оттаивания.
2. Слой объединения признаков (Concatenate) + BatchNormalization + MaxPool1D.
3. Рекуррентный слой LSTM (или GRU) для моделирования накопленной деформации во времени.
4. Полносвязная классификационная голова Dense + Softmax (3 класса).
5. Оптимизатор Adam (lr=0.001), функция потерь CategoricalCrossentropy.
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


@dataclass
class HybridNNResult:
    name: str
    model: keras.Model
    y_pred_val: np.ndarray
    y_proba_val: np.ndarray
    y_pred_test: np.ndarray
    y_proba_test: np.ndarray
    history: dict[str, list[float]]


def build_hybrid_cnn_lstm(
    seq_length: int = 60,
    num_features: int = 3,
    num_classes: int = 3,
    rnn_type: Literal["lstm", "gru"] = "lstm",
    learning_rate: float = 0.001,
) -> keras.Model:
    """Создает архитектуру гибридной нейросети CNN + LSTM/GRU."""
    inputs = layers.Input(shape=(seq_length, num_features), name="weather_sequence_input")

    # Мультимасштабный блок 1D-CNN (ядра 3, 5, 7)
    conv_k3 = layers.Conv1D(filters=32, kernel_size=3, padding="same", activation="relu", name="conv1d_k3")(inputs)
    conv_k5 = layers.Conv1D(filters=32, kernel_size=5, padding="same", activation="relu", name="conv1d_k5")(inputs)
    conv_k7 = layers.Conv1D(filters=32, kernel_size=7, padding="same", activation="relu", name="conv1d_k7")(inputs)

    merged_conv = layers.Concatenate(name="concat_conv_features")([conv_k3, conv_k5, conv_k7])
    x = layers.BatchNormalization(name="batch_norm")(merged_conv)
    x = layers.MaxPooling1D(pool_size=2, name="max_pooling_1d")(x)
    x = layers.Dropout(0.2, name="dropout_cnn")(x)

    # Рекуррентный блок (LSTM или GRU)
    if rnn_type.lower() == "gru":
        rnn_layer = layers.GRU(units=64, return_sequences=False, name="gru_recurrent")
    else:
        rnn_layer = layers.LSTM(units=64, return_sequences=False, name="lstm_recurrent")

    x = rnn_layer(x)
    x = layers.Dropout(0.3, name="dropout_rnn")(x)

    # Классификационная голова
    x = layers.Dense(32, activation="relu", name="dense_features")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="class_probabilities")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="Hybrid_CNN_LSTM_Puchina")

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def train_hybrid_model(
    X_train_seq: np.ndarray,
    y_train: np.ndarray,
    X_val_seq: np.ndarray,
    y_val: np.ndarray,
    X_test_seq: np.ndarray,
    y_test: np.ndarray,
    rnn_type: Literal["lstm", "gru"] = "lstm",
    epochs: int = 60,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    figure_save_path: str | Path = "results/figures/training_history.png",
) -> HybridNNResult:
    """Обучает гибридную нейросеть с временным контролем и весами классов."""
    print(f"[HybridNN] Сборка и обучение нейросети CNN+{rnn_type.upper()}...")
    seq_length = X_train_seq.shape[1]
    num_features = X_train_seq.shape[2]
    num_classes = 3

    # One-hot кодирование меток
    y_train_cat = keras.utils.to_categorical(y_train, num_classes=num_classes)
    y_val_cat = keras.utils.to_categorical(y_val, num_classes=num_classes)

    # Расчет сбалансированных весов классов для предотвращения доминирования класса 0
    unique_classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=unique_classes, y=y_train)
    class_weight_dict = {cls: float(w) for cls, w in zip(unique_classes, weights)}
    print(f"[HybridNN] Веса классов: {class_weight_dict}")

    model = build_hybrid_cnn_lstm(
        seq_length=seq_length,
        num_features=num_features,
        num_classes=num_classes,
        rnn_type=rnn_type,
        learning_rate=learning_rate,
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=6,
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

    # Прогнозы вероятностей и классов
    y_proba_val = model.predict(X_val_seq, verbose=0)
    y_pred_val = np.argmax(y_proba_val, axis=1)

    y_proba_test = model.predict(X_test_seq, verbose=0)
    y_pred_test = np.argmax(y_proba_test, axis=1)

    # Построение графика обучения
    _plot_history(history.history, figure_save_path)

    return HybridNNResult(
        name=f"Hybrid CNN+{rnn_type.upper()}",
        model=model,
        y_pred_val=y_pred_val,
        y_proba_val=y_proba_val,
        y_pred_test=y_pred_test,
        y_proba_test=y_proba_test,
        history=history.history,
    )


def _plot_history(history: dict[str, list[float]], save_path: str | Path) -> None:
    """Строит графики loss и accuracy в процессе обучения."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Loss
    ax1.plot(history.get("loss", []), label="Train Loss", color="royalblue", lw=2)
    ax1.plot(history.get("val_loss", []), label="Val Loss", color="darkorange", lw=2, linestyle="--")
    ax1.set_title("Динамика функции потерь (Loss)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Эпоха")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    # Accuracy
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
    print(f"[HybridNN] График истории обучения сохранен: {p.resolve()}")
