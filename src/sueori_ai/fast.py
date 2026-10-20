"""numpy 빠른 경로 — 학습 데이터 처리·평가용.

`reference.py`(정본)와 **비트 단위로 같은 결과**를 내도록 연산 순서를 맞춘다.
- 제곱합은 `dx*dx + dy*dy + dz*dz`를 원소별로 왼쪽부터 계산한다(np.sum이나 norm 대신)
- 21개 점 거리 평균은 점 0→20 순서로 하나씩 더한 뒤 21로 나눈다(pairwise 합산을 피함)
- DTW 누적은 Python float로 같은 순서로 계산한다
tests/test_fast.py가 공통 테스트 벡터로 일치를 확인한다.
"""

from __future__ import annotations

import math

import numpy as np

from . import spec
from .reference import DtwResult, InvalidInput

_PALM_PAIRS = [
    (spec.PALM_INDICES[i], spec.PALM_INDICES[j])
    for i in range(len(spec.PALM_INDICES))
    for j in range(i + 1, len(spec.PALM_INDICES))
]


def _as_frames(frames) -> np.ndarray:
    arr = np.asarray(frames, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.shape[1:] != (spec.NUM_LANDMARKS, spec.DIMS):
        raise InvalidInput(f"프레임 모양은 [N, 21, 3]이어야 함: {arr.shape}")
    if not np.isfinite(arr).all():
        raise InvalidInput("NaN·Infinity 좌표가 있음")
    return arr


def normalize_frames(frames) -> np.ndarray:
    """[N, 21, 3] → [N, 63]. reference.normalize_frame과 같은 결과."""
    arr = _as_frames(frames)
    scale = np.zeros(arr.shape[0])
    for a, b in _PALM_PAIRS:
        d = arr[:, a, :] - arr[:, b, :]
        dist = np.sqrt(d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1] + d[:, 2] * d[:, 2])
        scale = np.where(dist > scale, dist, scale)
    scale = np.where(scale < spec.SCALE_EPSILON, 1.0, scale)
    out = (arr - arr[:, :1, :]) / scale[:, None, None]
    return out.reshape(arr.shape[0], spec.FEATURES)


def resample(vectors: np.ndarray, length: int = spec.SEQUENCE_LENGTH) -> np.ndarray:
    """[N, 63] → [length, 63]. reference.resample과 같은 위치식·보간식."""
    vectors = np.asarray(vectors, dtype=np.float64)
    n = vectors.shape[0]
    if n == 0:
        raise InvalidInput("리샘플할 프레임이 없음")
    if n == 1:
        return np.repeat(vectors, length, axis=0)
    k = np.arange(length)
    s = (k * (n - 1)) / (length - 1)
    lo = np.floor(s).astype(int)
    hi = np.ceil(s).astype(int)
    w = (s - lo)[:, None]
    a, b = vectors[lo], vectors[hi]
    return a + (b - a) * w


def static_tensor(frame) -> np.ndarray:
    v = normalize_frames(frame)
    return np.repeat(v, spec.SEQUENCE_LENGTH, axis=0)


def dynamic_tensor(frames) -> np.ndarray:
    if len(frames) < spec.MIN_DYNAMIC_FRAMES:
        raise InvalidInput(f"유효 프레임 {len(frames)}개 < {spec.MIN_DYNAMIC_FRAMES}")
    return resample(normalize_frames(frames))


def _pairwise_cost(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """정규화 벡터 a[n,63], b[m,63] → 비용 행렬 [n, m] (랜드마크당 평균 거리)."""
    pa = a.reshape(-1, 1, spec.NUM_LANDMARKS, 3)
    pb = b.reshape(1, -1, spec.NUM_LANDMARKS, 3)
    total = np.zeros((a.shape[0], b.shape[0]))
    for i in range(spec.NUM_LANDMARKS):
        d = pa[:, :, i, :] - pb[:, :, i, :]
        total = total + np.sqrt(d[..., 0] * d[..., 0] + d[..., 1] * d[..., 1] + d[..., 2] * d[..., 2])
    return total / spec.NUM_LANDMARKS


def landmark_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(_pairwise_cost(np.asarray(a)[None, :], np.asarray(b)[None, :])[0, 0])


def mirror_vectors(v: np.ndarray) -> np.ndarray:
    """v1 반전: 정규화 벡터의 x 성분 부호를 바꾼다(reference.mirror_vector와 같음)."""
    out = np.array(v, dtype=np.float64, copy=True)
    out[..., 0::3] = -out[..., 0::3]
    return out


def static_score(reference_frame, candidate_frame, *, spec_version: str = spec.SPEC_VERSION) -> float:
    spec.check_version(spec_version)
    r, c = normalize_frames(reference_frame)[0], normalize_frames(candidate_frame)[0]
    d = landmark_distance(r, c)
    if spec.MIRROR_FALLBACK[spec_version]:
        dm = landmark_distance(r, mirror_vectors(c))
        d = dm if dm < d else d  # 동률이면 원본
    x = 100.0 * math.exp(-spec.STATIC_DECAY * d)
    return 0.0 if x < 0.0 else (100.0 if x > 100.0 else x)


def dtw_score(reference_frames, candidate_frames, *, spec_version: str = spec.SPEC_VERSION) -> DtwResult:
    """reference.dtw_score와 같은 결과(비용 행렬만 numpy, 누적·되짚기는 같은 순서의 Python float)."""
    spec.check_version(spec_version)
    if len(reference_frames) == 0 or len(candidate_frames) == 0:
        return DtwResult(score=0.0, avg_cost=math.inf, path_length=0)
    a, b = normalize_frames(reference_frames), normalize_frames(candidate_frames)
    avg, length = _dtw_core(_pairwise_cost(a, b).tolist())
    mirrored = False
    if spec.MIRROR_FALLBACK[spec_version]:
        am, lm = _dtw_core(_pairwise_cost(a, mirror_vectors(b)).tolist())
        if am < avg:  # 동률이면 원본
            avg, length, mirrored = am, lm, True
    x = 100.0 * math.exp(-spec.DTW_DECAY * avg)
    return DtwResult(score=0.0 if x < 0.0 else (100.0 if x > 100.0 else x), avg_cost=avg, path_length=length, mirrored=mirrored)


def _dtw_core(cost: list[list[float]]) -> tuple[float, int]:
    n, m = len(cost), len(cost[0])
    inf = math.inf
    d = [[inf] * (m + 1) for _ in range(n + 1)]
    d[0][0] = 0.0
    for i in range(1, n + 1):
        row, prev, ci = d[i], d[i - 1], cost[i - 1]
        for j in range(1, m + 1):
            row[j] = ci[j - 1] + min(prev[j], row[j - 1], prev[j - 1])
    i, j, length = n, m, 0
    while i > 0 or j > 0:
        length += 1
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            diag, up, left = d[i - 1][j - 1], d[i - 1][j], d[i][j - 1]
            best = min(diag, up, left)
            if best == diag:
                i, j = i - 1, j - 1
            elif best == up:
                i -= 1
            else:
                j -= 1
    return d[n][m] / length, length
