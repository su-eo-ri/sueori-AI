"""합성 손 랜드마크 생성기 (테스트 벡터·스모크용).

이 레포는 public이므로 벡터에는 실제 촬영자 데이터나 국립국어원 기준 데이터(CC BY-SA 2.0)를
쓰지 않고, 여기서 계산으로 만든 손 모양만 쓴다. 해부학적으로 정확할 필요는 없고, MediaPipe
출력과 비슷한 좌표 범위(이미지 좌표 0~1, z는 손목 기준 작은 값)와 21점 구조만 맞춘다.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace

# 손 좌표계: 원점 = 손목, +x = 오른쪽, −y = 손가락 방향(이미지 위쪽), −z = 카메라 쪽
_FINGER_BASES = {
    "index": (-0.030, -0.090),
    "middle": (0.000, -0.095),
    "ring": (0.025, -0.088),
    "pinky": (0.048, -0.075),
}
_SEGMENTS = (0.040, 0.025, 0.020)
_THUMB_BASE = (-0.025, -0.025)
_THUMB_SEGMENTS = (0.035, 0.030, 0.025, 0.020)


@dataclass(frozen=True)
class HandPose:
    """손 자세 파라미터. curl은 0(편 손가락) ~ 1(완전히 굽힘)."""

    thumb: float = 0.0
    index: float = 0.0
    middle: float = 0.0
    ring: float = 0.0
    pinky: float = 0.0
    roll: float = 0.0  # 화면 평면 회전(rad)
    yaw: float = 0.0  # 손바닥이 옆으로 도는 회전(rad)
    scale: float = 1.0  # 카메라 거리(손 크기)
    x: float = 0.5  # 손목 위치(이미지 좌표)
    y: float = 0.75


def _finger(base: tuple[float, float], curl: float) -> list[tuple[float, float, float]]:
    pts = [(base[0], base[1], 0.0)]
    x, y, z = pts[0]
    angle = 0.0
    for seg in _SEGMENTS:
        angle += curl * 1.2
        y -= seg * math.cos(angle)
        z -= seg * math.sin(angle)
        pts.append((x, y, z))
    return pts


def _thumb(curl: float) -> list[tuple[float, float, float]]:
    x, y, z = _THUMB_BASE[0], _THUMB_BASE[1], 0.0
    pts = [(x, y, z)]
    angle = 0.0
    for k, seg in enumerate(_THUMB_SEGMENTS[1:]):
        angle += curl * 0.9
        dx = -math.cos(0.8 + angle)
        dy = -math.sin(0.8 + angle)
        x += seg * dx
        y += seg * dy
        z -= seg * 0.3 * curl * (k + 1)
        pts.append((x, y, z))
    return pts


def hand_frame(pose: HandPose) -> list[list[float]]:
    """자세 → MediaPipe 순서(0 손목, 1~4 엄지, 5~8 검지, 9~12 중지, 13~16 약지, 17~20 소지)의 21점."""
    local: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)]
    local += _thumb(pose.thumb)
    for name in ("index", "middle", "ring", "pinky"):
        local += _finger(_FINGER_BASES[name], getattr(pose, name))
    assert len(local) == 21

    cy, sy = math.cos(pose.yaw), math.sin(pose.yaw)
    cr, sr = math.cos(pose.roll), math.sin(pose.roll)
    out: list[list[float]] = []
    for lx, ly, lz in local:
        # yaw: y축 회전
        x1 = lx * cy + lz * sy
        z1 = -lx * sy + lz * cy
        # roll: z축(화면 평면) 회전
        x2 = x1 * cr - ly * sr
        y2 = x1 * sr + ly * cr
        out.append([pose.x + x2 * pose.scale, pose.y + y2 * pose.scale, z1 * pose.scale])
    return out


def round_frame(frame: list[list[float]], digits: int = 6) -> list[list[float]]:
    """벡터 입력값을 짧게 만든다. 반올림된 값이 곧 입력이므로 정확성에는 영향이 없다."""
    return [[round(c, digits) for c in p] for p in frame]


def jitter(frame: list[list[float]], rng: random.Random, sigma: float) -> list[list[float]]:
    return [[c + rng.gauss(0.0, sigma) for c in p] for p in frame]


def _lerp_pose(a: HandPose, b: HandPose, t: float) -> HandPose:
    fields = ("thumb", "index", "middle", "ring", "pinky", "roll", "yaw", "scale", "x", "y")
    return replace(a, **{f: getattr(a, f) + (getattr(b, f) - getattr(a, f)) * t for f in fields})


def motion(keyposes: list[HandPose], frames: int, *, warp: float = 1.0) -> list[list[list[float]]]:
    """키 자세 사이를 보간한 동작. warp != 1이면 시간축을 비선형으로 늘린다(빠르게 시작 → 느리게 등)."""
    out = []
    segs = len(keyposes) - 1
    for k in range(frames):
        t = (k / (frames - 1)) ** warp
        pos = t * segs
        i = min(int(pos), segs - 1)
        out.append(hand_frame(_lerp_pose(keyposes[i], keyposes[i + 1], pos - i)))
    return out
