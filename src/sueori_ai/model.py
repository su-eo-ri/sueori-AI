"""v0 모델 구조와 학습·온도 보정.

모델은 Keras 2(`tf_keras`)로 정의한다(tensorflowjs 변환기 호환, README 툴체인 표).
구조는 변환 스모크에서 TF.js·TFLite 양쪽 통과를 확인한 1D-CNN이다.
"""

from __future__ import annotations

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from dataclasses import dataclass  # noqa: E402

import numpy as np  # noqa: E402
import tf_keras as keras  # noqa: E402

from . import spec  # noqa: E402
from .dataset import Examples, mirror_x  # noqa: E402


def build_cnn(num_classes: int, *, width: int = 64, dropout: float = 0.2) -> keras.Model:
    inp = keras.Input(shape=(spec.SEQUENCE_LENGTH, spec.FEATURES), name="landmarks")
    x = keras.layers.Conv1D(width, 5, padding="same", activation="relu")(inp)
    x = keras.layers.Conv1D(width, 5, padding="same", activation="relu")(x)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.Dense(width, activation="relu")(x)
    out = keras.layers.Dense(num_classes, name="logits")(x)
    return keras.Model(inp, out, name="sueori_cnn_v0")


@dataclass
class TrainConfig:
    epochs: int = 60
    batch_size: int = 32
    learning_rate: float = 1e-3
    mirror_augment: bool = True
    patience: int = 8
    val_fraction: float = 0.15
    seed: int = 0


def _class_weights(y: np.ndarray, num_classes: int) -> dict[int, float]:
    counts = np.bincount(y, minlength=num_classes).astype(float)
    counts[counts == 0] = 1.0
    w = len(y) / (num_classes * counts)
    return {i: float(v) for i, v in enumerate(w)}


def split_validation(ex: Examples, fraction: float, seed: int) -> tuple[Examples, Examples]:
    """조기 종료·온도 보정용 검증 세트.

    촬영자가 2명 이상이면 **촬영자 단위**로 1명을 뺀다(스펙 §6 "학습에 쓰지 않은 사람" 기준, CTO 2026-10-16).
    1명뿐이면 sample_id 단위로 나눈다(같은 클립의 여러 프레임이 양쪽에 나뉘지 않게).
    """
    rng = np.random.default_rng(seed)
    people = np.unique(ex.participant)
    if len(people) >= 2:
        held = rng.choice(people)
        mask = ex.participant == held
        return ex.subset(~mask), ex.subset(mask)
    ids = np.unique(ex.sample_id)
    val_ids = set(rng.choice(ids, size=max(1, int(len(ids) * fraction)), replace=False).tolist())
    mask = np.array([s in val_ids for s in ex.sample_id])
    return ex.subset(~mask), ex.subset(mask)


def train(ex: Examples, num_classes: int, cfg: TrainConfig = TrainConfig()) -> tuple[keras.Model, float, dict]:
    """학습 → (모델, 보정된 온도 τ, 기록)."""
    keras.utils.set_random_seed(cfg.seed)
    tr, va = split_validation(ex, cfg.val_fraction, cfg.seed)
    x, y = tr.x, tr.y
    if cfg.mirror_augment:
        x = np.concatenate([x, mirror_x(x)])
        y = np.concatenate([y, y])
    model = build_cnn(num_classes)
    model.compile(
        optimizer=keras.optimizers.Adam(cfg.learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    hist = model.fit(
        x, y,
        validation_data=(va.x, va.y),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        class_weight=_class_weights(y, num_classes),
        callbacks=[keras.callbacks.EarlyStopping(patience=cfg.patience, restore_best_weights=True)],
        verbose=0,
    )
    tau = calibrate_temperature(model.predict(va.x, verbose=0), va.y)
    return model, tau, {"epochs_run": len(hist.history["loss"]), "val_loss": float(min(hist.history["val_loss"])), "tau": tau}


def softmax(logits: np.ndarray, tau: float = 1.0) -> np.ndarray:
    z = logits / tau
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def calibrate_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """검증 세트 음의 로그 우도가 가장 작은 τ(스펙 §6). 0.25~8 범위 격자 탐색."""
    best_tau, best_nll = 1.0, np.inf
    for tau in np.exp(np.linspace(np.log(0.25), np.log(8.0), 61)):
        p = softmax(logits, tau)[np.arange(len(y)), y]
        nll = float(-np.log(np.clip(p, 1e-12, 1.0)).mean())
        if nll < best_nll:
            best_tau, best_nll = float(tau), nll
    return best_tau


def scores(logits: np.ndarray, targets: np.ndarray, tau: float) -> np.ndarray:
    """스펙 §6 점수 = 100·softmax(logits/τ)[target]."""
    return 100.0 * softmax(logits, tau)[np.arange(len(targets)), targets]
