"""ai-io-v0 / ai-io-v1 Python 기준 구현 (정본).

v1의 차이는 §10 폴백뿐이다: 입력을 정규화한 뒤 x를 반전(x → −x)한 버전도 채점해 비용이 더 작은 쪽을 쓴다.
동률이면 원본을 쓴다. 폴백 함수는 `spec_version` 인자로 버전을 고르고, 기본값은 현재 버전(spec.SPEC_VERSION)이다.

TS(sueori-FE)·Dart(sueori-AOS)·Swift(sueori-IOS) 구현은 이 모듈이 만든 공통 테스트
벡터 **vectors/ai-io-v1-vectors.json**(현재 버전)을 통과해야 한다. ai-io-v0-vectors.json은 기록용이다.

비트 단위 재현 규칙(스펙 §10.0)을 지키기 위해 numpy를 쓰지 않는다.
- Python float는 IEEE 754 Float64다.
- 합산은 항상 인덱스 순서대로 한 번에 하나씩 더한다(numpy의 pairwise 합산 금지).
- `dx*dx + dy*dy + dz*dz`는 왼쪽부터 계산되고, CPython은 FMA로 합치지 않는다.
- sqrt는 IEEE에서 정확히 반올림된다. exp는 플랫폼마다 마지막 비트가 다를 수 있다(§8 허용 오차).

학습 파이프라인에서 numpy로 빠르게 계산하는 버전을 따로 두더라도, 그 결과는 이 모듈과
비교해 검증한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

from . import spec

Point = Sequence[float]  # (x, y, z)
Frame = Sequence[Point]  # 21개 점
Vector = list[float]  # 정규화된 63차원


class InvalidInput(ValueError):
    """채점할 수 없는 입력(NaN·Infinity 좌표, 프레임 부족 등)."""


# ── §10.0 유효성 ──────────────────────────────────────────────────────────


def is_finite_frame(frame: Frame) -> bool:
    for p in frame:
        for c in p:
            if not math.isfinite(c):
                return False
    return True


def _check_frame(frame: Frame) -> None:
    if len(frame) != spec.NUM_LANDMARKS:
        raise InvalidInput(f"랜드마크는 {spec.NUM_LANDMARKS}개여야 함: {len(frame)}")
    for p in frame:
        if len(p) != spec.DIMS:
            raise InvalidInput(f"점은 (x, y, z) 3개 값이어야 함: {len(p)}")
    if not is_finite_frame(frame):
        raise InvalidInput("NaN·Infinity 좌표가 있음")


# ── §2 프레임 정규화 ──────────────────────────────────────────────────────


def _dist3(a: Point, b: Point) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def palm_scale(frame: Frame) -> float:
    """손바닥 5점 10개 쌍의 3D 거리 중 최댓값. `SCALE_EPSILON` 미만이면 1.0."""
    scale = 0.0
    idx = spec.PALM_INDICES
    for i in range(len(idx)):
        for j in range(i + 1, len(idx)):
            d = _dist3(frame[idx[i]], frame[idx[j]])
            if d > scale:
                scale = d
    return 1.0 if scale < spec.SCALE_EPSILON else scale


def normalize_frame(frame: Frame) -> Vector:
    """손목 원점 이동 + 손바닥 스케일로 나눔. 회전은 정규화하지 않는다.

    출력: [x0, y0, z0, x1, y1, z1, …, x20, y20, z20]
    """
    _check_frame(frame)
    wrist = frame[0]
    scale = palm_scale(frame)
    out: Vector = []
    for p in frame:
        out.append((p[0] - wrist[0]) / scale)
        out.append((p[1] - wrist[1]) / scale)
        out.append((p[2] - wrist[2]) / scale)
    return out


# ── §4 시간축 ─────────────────────────────────────────────────────────────


def resample(vectors: Sequence[Vector], length: int = spec.SEQUENCE_LENGTH) -> list[Vector]:
    """정규화된 벡터 N개를 프레임 인덱스 기준 선형 보간으로 `length`개로 만든다.

    k번째 출력의 원래 위치는 s = (k·(N−1)) / (T−1) 이다(정수 곱을 먼저 하고 나눈다).
    lo = ⌊s⌋, hi = ⌈s⌉, w = s − lo 일 때 값은 a + (b − a)·w 다(a = v[lo], b = v[hi]).
    N = 1이면 그 벡터를 반복한다.
    """
    n = len(vectors)
    if n == 0:
        raise InvalidInput("리샘플할 프레임이 없음")
    if length < 2:
        raise ValueError("length는 2 이상이어야 함")
    if n == 1:
        return [list(vectors[0]) for _ in range(length)]
    out: list[Vector] = []
    for k in range(length):
        s = (k * (n - 1)) / (length - 1)
        lo = math.floor(s)
        hi = math.ceil(s)
        w = s - lo
        a = vectors[lo]
        b = vectors[hi]
        out.append([a[i] + (b[i] - a[i]) * w for i in range(len(a))])
    return out


def static_tensor(frame: Frame) -> list[Vector]:
    """정적(지문자): 프레임 1장을 정규화해서 T번 반복 → [T, 63]."""
    v = normalize_frame(frame)
    return [list(v) for _ in range(spec.SEQUENCE_LENGTH)]


def dynamic_tensor(frames: Sequence[Frame]) -> list[Vector]:
    """동적(단어): 프레임별 정규화 → T로 리샘플 → [T, 63]. 유효 프레임 8개 미만이면 채점 불가."""
    if len(frames) < spec.MIN_DYNAMIC_FRAMES:
        raise InvalidInput(f"유효 프레임 {len(frames)}개 < {spec.MIN_DYNAMIC_FRAMES}")
    return resample([normalize_frame(f) for f in frames])


# ── §10 폴백 채점 ─────────────────────────────────────────────────────────


def landmark_distance(a: Vector, b: Vector) -> float:
    """랜드마크당 평균 3D 거리. 점 0→20 순서로 순차 합산한 뒤 21로 나눈다."""
    if len(a) != spec.FEATURES or len(b) != spec.FEATURES:
        raise ValueError("63차원 벡터여야 함")
    total = 0.0
    for i in range(spec.NUM_LANDMARKS):
        dx = a[i * 3] - b[i * 3]
        dy = a[i * 3 + 1] - b[i * 3 + 1]
        dz = a[i * 3 + 2] - b[i * 3 + 2]
        total += math.sqrt(dx * dx + dy * dy + dz * dz)
    return total / spec.NUM_LANDMARKS


def _clamp_score(x: float) -> float:
    return 0.0 if x < 0.0 else (100.0 if x > 100.0 else x)


def mirror_vector(v: Vector) -> Vector:
    """v1 반전: 정규화된 63차원 벡터의 x 성분 부호를 바꾼다(스펙 §3 증강과 같은 정의).

    이미지 좌표에서 1 − x로 뒤집는 것과 수학적으로 같지만 부동소수점 마지막 비트가 달라질 수 있어 쓰지 않는다.
    """
    return [-c if i % 3 == 0 else c for i, c in enumerate(v)]


@dataclass(frozen=True)
class StaticResult:
    score: float
    distance: float
    mirrored: bool = False  # v1: 반전 쪽을 썼는지


def static_score(reference: Frame, candidate: Frame, *, spec_version: str = spec.SPEC_VERSION) -> StaticResult:
    """§10.1 정적 거리 점수: clamp(100·exp(−2.0·d), 0, 100). v1은 반전 후보와 비교해 d가 더 작은 쪽."""
    spec.check_version(spec_version)
    r, c = normalize_frame(reference), normalize_frame(candidate)
    d, mirrored = landmark_distance(r, c), False
    if spec.MIRROR_FALLBACK[spec_version]:
        dm = landmark_distance(r, mirror_vector(c))
        if dm < d:  # 동률이면 원본
            d, mirrored = dm, True
    return StaticResult(score=_clamp_score(100.0 * math.exp(-spec.STATIC_DECAY * d)), distance=d, mirrored=mirrored)


@dataclass(frozen=True)
class DtwResult:
    score: float
    avg_cost: float
    path_length: int
    mirrored: bool = False  # v1: 반전 쪽을 썼는지


def dtw_score(reference: Sequence[Frame], candidate: Sequence[Frame], *, spec_version: str = spec.SPEC_VERSION) -> DtwResult:
    """§10.2 DTW 점수. 리샘플하지 않고, 윈도우 제약도 없다. v1은 반전 후보와 비교해 평균 비용이 더 작은 쪽.

    동적 입력의 N<8 규칙은 호출하는 쪽(`score_dynamic`)에서 적용한다. 이 함수는 MVP
    DtwScorer와 같은 동작(빈 시퀀스면 0점)을 유지해 교차 확인에 쓸 수 있게 한다.
    """
    spec.check_version(spec_version)
    a = [normalize_frame(f) for f in reference]
    b = [normalize_frame(f) for f in candidate]
    if len(a) == 0 or len(b) == 0:
        return DtwResult(score=0.0, avg_cost=math.inf, path_length=0)
    avg_cost, path_length = _dtw_core(a, b)
    mirrored = False
    if spec.MIRROR_FALLBACK[spec_version]:
        am, lm = _dtw_core(a, [mirror_vector(v) for v in b])
        if am < avg_cost:  # 동률이면 원본
            avg_cost, path_length, mirrored = am, lm, True
    score = _clamp_score(100.0 * math.exp(-spec.DTW_DECAY * avg_cost))
    return DtwResult(score=score, avg_cost=avg_cost, path_length=path_length, mirrored=mirrored)


def _dtw_core(a: list[Vector], b: list[Vector]) -> tuple[float, int]:
    """정규화 벡터 시퀀스 → (평균 정렬 비용, 경로 길이)."""
    n, m = len(a), len(b)
    inf = math.inf
    d = [[inf] * (m + 1) for _ in range(n + 1)]
    d[0][0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = landmark_distance(a[i - 1], b[j - 1])
            d[i][j] = cost + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])

    path_length = _backtrack(d, n, m)
    return d[n][m] / path_length, path_length


def _backtrack(d: list[list[float]], n: int, m: int) -> int:
    """(n, m)에서 (0, 0)까지 되짚는다. 동률이면 diag → up → left 순서."""
    i, j, length = n, m, 0
    while i > 0 or j > 0:
        length += 1
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            diag = d[i - 1][j - 1]
            up = d[i - 1][j]
            left = d[i][j - 1]
            best = min(diag, up, left)
            if best == diag:
                i -= 1
                j -= 1
            elif best == up:
                i -= 1
            else:
                j -= 1
    return length


def score_dynamic(reference: Sequence[Frame], candidate: Sequence[Frame], *, spec_version: str = spec.SPEC_VERSION) -> DtwResult:
    """동적 트랙 하나의 폴백 채점(유효성 규칙 포함). 채점 불가면 InvalidInput."""
    if len(candidate) < spec.MIN_DYNAMIC_FRAMES:
        raise InvalidInput(f"유효 프레임 {len(candidate)}개 < {spec.MIN_DYNAMIC_FRAMES}")
    return dtw_score(reference, candidate, spec_version=spec_version)


def best_track_score(
    reference: Sequence[Frame], tracks: dict[str, Sequence[Frame]], *, dynamic: bool, spec_version: str = spec.SPEC_VERSION
) -> Optional[tuple[str, float]]:
    """§3 handedness 트랙별로 채점하고 최고점 트랙을 고른다. 채점 가능한 트랙이 없으면 None.

    동점이면 먼저 나온 트랙(dict 삽입 순서 = 캡처에서 처음 나타난 순서)을 유지한다.
    """
    best: Optional[tuple[str, float]] = None
    for label, frames in tracks.items():
        try:
            if dynamic:
                s = score_dynamic(reference, frames, spec_version=spec_version).score
            else:
                if not frames:
                    continue
                s = static_score(reference[0], frames[0], spec_version=spec_version).score
        except InvalidInput:
            continue
        if best is None or s > best[1]:
            best = (label, s)
    return best


def recording_duration_ms(first_t_ms: Optional[float], last_t_ms: Optional[float]) -> int:
    """§10.3 녹화 길이. 반올림은 0에서 먼 쪽으로(half away from zero, Dart `round()`와 같음)."""
    if first_t_ms is None or last_t_ms is None or last_t_ms <= first_t_ms:
        return spec.DEFAULT_RECORDING_MS
    raw = (last_t_ms - first_t_ms) * spec.RECORDING_FACTOR
    # floor(raw + 0.5)는 0.49999999999999994 같은 값에서 덧셈 반올림으로 틀리므로,
    # 소수부를 따로 비교한다(raw > 0이라 raw − floor(raw)는 정확히 계산된다).
    whole = math.floor(raw)
    rounded = int(whole) + (1 if raw - whole >= 0.5 else 0)
    return max(spec.MIN_RECORDING_MS, min(spec.MAX_RECORDING_MS, rounded))
