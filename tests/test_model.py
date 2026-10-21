"""split_validation·train — 촬영자 1명 + 클래스당 예제가 적은 경우의 검증 분할.

버그 재현(2026-10-21, KCISA 47단어 실데이터): 촬영자가 1명이고 클래스당 예제가
1~3개뿐이면, sample_id를 무작위로 검증셋에 보내다가 어떤 클래스를 학습 데이터에서
통째로 비울 수 있었다. 그러면 그 클래스의 val_loss가 구조적으로 개선 불가능해서
EarlyStopping이 사실상 학습 안 된 상태(1~2에폭)에서 멈춰버린다(47클래스 중
학습셋 정확도 3.9% — 거의 무작위 수준). class_weight·검증분할을 빼고 직접 돌리면
200에폭에 학습셋 정확도 94.5%까지 올라가는 걸로 보아 데이터·모델 자체는 정상이었다.
"""

import numpy as np

from sueori_ai import spec
from sueori_ai.dataset import Examples
from sueori_ai.model import TrainConfig, split_validation, train


def _examples(class_clips: list[list[int]]) -> Examples:
    """클래스별 클립 구성으로 Examples를 만든다. 촬영자 1명.

    class_clips[c]는 클래스 c의 클립(sample_id)마다 예제(프레임) 수를 담은 리스트다.
    예: [3] → 클립 1개에 프레임 3장(정적 단어 한 클립, §3.2). [1, 1] → 클립 2개에 프레임 1장씩
    (같은 단어를 두 번 촬영). 같은 클립의 여러 프레임은 같은 sample_id를 공유한다.
    """
    xs, ys, parts, sids, labels, words = [], [], [], [], [], []
    for c, clips in enumerate(class_clips):
        for clip_idx, n in enumerate(clips):
            sid = f"word{c}-clip{clip_idx}"
            for _ in range(n):
                xs.append(np.zeros((spec.SEQUENCE_LENGTH, spec.FEATURES), dtype=np.float32))
                ys.append(c)
                parts.append("P1")
                sids.append(sid)
                labels.append("correct")
                words.append(f"word{c}")
    return Examples(np.stack(xs), np.array(ys), np.array(parts), np.array(sids), np.array(labels), np.array(words))


def test_split_validation_never_empties_a_class_from_train():
    # 절반은 클립 1개뿐(동적 단어처럼 분할 불가능), 절반은 클립 2개(분할 가능)
    ex = _examples([[1], [3], [1], [3], [1], [3], [1], [3], [1, 3], [1, 1]])
    tr, va = split_validation(ex, fraction=0.5, seed=0)
    assert set(np.unique(ex.y).tolist()) == set(np.unique(tr.y).tolist())
    assert len(tr) + len(va) == len(ex)


def test_split_validation_empty_when_every_class_is_a_singleton():
    ex = _examples([[1], [1], [1], [1], [1]])
    tr, va = split_validation(ex, fraction=0.5, seed=0)
    assert len(va) == 0
    assert len(tr) == len(ex)


def test_split_validation_still_splits_when_classes_have_room():
    # 클래스마다 클립이 2개씩(=같은 단어를 두 번 촬영)이면 한쪽을 검증셋으로 보낼 수 있다
    ex = _examples([[2, 2]] * 8)
    tr, va = split_validation(ex, fraction=0.3, seed=0)
    assert len(va) > 0
    assert set(np.unique(ex.y).tolist()) == set(np.unique(tr.y).tolist())


def test_train_handles_empty_validation_without_crashing():
    # 전부 싱글톤 클래스라 검증셋이 비는 극단 상황에서도 train()이 죽지 않아야 한다
    ex = _examples([[1]] * 6)
    cfg = TrainConfig(epochs=3, batch_size=2, mirror_augment=False)
    model, tau, info = train(ex, num_classes=6, cfg=cfg)
    assert info["val_loss"] is None
    assert info["epochs_run"] == 3
    assert 0.25 <= tau <= 8.0
    logits = model.predict(ex.x, verbose=0)
    assert logits.shape == (6, 6)
