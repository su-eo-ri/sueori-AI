"""합성 클립 데이터셋 — 학습 파이프라인을 실제 데이터 없이 끝까지 돌려 보기 위한 것.

데이터 수집 프로토콜 protocol-v0 §6 레코드 형식으로 만든다. 촬영자 가명은 드라이런 규칙대로
테스트 가명(`T001`…)이고, 실제 가명 체계(`P…`)와 섞이지 않는다.
⚠️ 합성 데이터로 나온 성능 수치는 의미가 없다. 파이프라인이 동작하는지만 확인한다.

단어는 실제 수어가 아니라 계산으로 만든 가상의 동작이다(`syn/d0`…, `syn/s0`…).
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, replace

from . import spec
from .reference import recording_duration_ms
from .synthetic import HandPose, hand_frame

FRAME_MS = 33


@dataclass(frozen=True)
class Participant:
    pid: str
    hand_scale: float  # 손 크기·카메라 거리
    speed: float  # 1보다 크면 느리게
    offset: tuple[float, float]  # 화면 위치
    noise: float
    left_handed: bool
    frame: tuple[int, int]  # (width, height)


@dataclass(frozen=True)
class SynthWord:
    key: str
    word_type: str  # static | dynamic
    keyposes: tuple[HandPose, ...]  # 정적은 1개
    duration_ms: int


def make_vocabulary(n_dynamic: int, n_static: int, seed: int) -> list[SynthWord]:
    rng = random.Random(seed)
    words = []
    for i in range(n_dynamic):
        def pose():
            return HandPose(
                thumb=rng.random(), index=rng.random(), middle=rng.random(),
                ring=rng.random(), pinky=rng.random(),
                roll=rng.uniform(-0.6, 0.6), yaw=rng.uniform(-0.8, 0.8),
                x=rng.uniform(0.35, 0.65), y=rng.uniform(0.5, 0.8),
            )
        k = rng.choice([2, 3])
        words.append(SynthWord(f"syn/d{i}", "dynamic", tuple(pose() for _ in range(k)), rng.randrange(1400, 2600, 100)))
    for i in range(n_static):
        p = HandPose(
            thumb=rng.random(), index=rng.choice([0.0, 1.0]), middle=rng.choice([0.0, 1.0]),
            ring=rng.choice([0.0, 1.0]), pinky=rng.choice([0.0, 1.0]),
            roll=rng.uniform(-0.4, 0.4), yaw=rng.uniform(-0.6, 0.6),
        )
        words.append(SynthWord(f"syn/s{i}", "static", (p,), 0))
    return words


def make_participants(n: int, seed: int) -> list[Participant]:
    rng = random.Random(seed + 1)
    frames = [(1280, 720), (720, 1280), (640, 480)]
    out = []
    for i in range(n):
        out.append(
            Participant(
                pid=f"T{i + 1:03d}",
                hand_scale=rng.uniform(0.8, 1.25),
                speed=rng.uniform(0.8, 1.3),
                offset=(rng.uniform(-0.08, 0.08), rng.uniform(-0.06, 0.06)),
                noise=rng.uniform(0.001, 0.004),
                left_handed=(i == n - 1 and n >= 3),  # 마지막 촬영자는 왼손잡이(스펙 §3)
                frame=frames[i % len(frames)],
            )
        )
    return out


def _lerp(a: HandPose, b: HandPose, t: float) -> HandPose:
    f = ("thumb", "index", "middle", "ring", "pinky", "roll", "yaw", "scale", "x", "y")
    return replace(a, **{k: getattr(a, k) + (getattr(b, k) - getattr(a, k)) * t for k in f})


def _pose_at(keys: tuple[HandPose, ...], t: float) -> HandPose:
    if len(keys) == 1:
        return keys[0]
    pos = min(max(t, 0.0), 1.0) * (len(keys) - 1)
    i = min(int(pos), len(keys) - 2)
    return _lerp(keys[i], keys[i + 1], pos - i)


def _apply_participant(pose: HandPose, p: Participant) -> HandPose:
    return replace(pose, scale=pose.scale * p.hand_scale, x=pose.x + p.offset[0], y=pose.y + p.offset[1])


def _render(pose: HandPose, p: Participant, rng: random.Random) -> list[list[float]]:
    pts = hand_frame(_apply_participant(pose, p))
    if p.left_handed:  # 왼손 = 오른손 동작의 좌우 거울상
        pts = [[1.0 - x, y, z] for x, y, z in pts]
    return [[x + rng.gauss(0, p.noise), y + rng.gauss(0, p.noise), z + rng.gauss(0, p.noise * 0.5)] for x, y, z in pts]


def _frames_from(poses: list[HandPose], p: Participant, rng: random.Random, dropout: float) -> list[dict]:
    label = "Left" if p.left_handed else "Right"
    out = []
    for k, pose in enumerate(poses):
        hands = [] if rng.random() < dropout else [{"handedness": label, "score": 0.95, "landmarks": _render(pose, p, rng)}]
        out.append({"t_ms": k * FRAME_MS, "hands": hands})
    return out


def _n_frames(duration_ms: float) -> int:
    return max(2, int(round(duration_ms / FRAME_MS)))


def _near_miss(word: SynthWord, rng: random.Random) -> tuple[str, tuple[HandPose, ...], float]:
    """근접 오답 하나: (error_type, 바뀐 키 자세, 동작 진행 비율)."""
    keys = word.keyposes
    if word.word_type == "static":
        kind = rng.choice(["confusable", "orientation"])
        p = keys[0]
        if kind == "confusable":  # 손가락 하나를 반대로
            f = rng.choice(["index", "middle", "ring", "pinky"])
            p = replace(p, **{f: 1.0 - getattr(p, f)})
        else:
            p = replace(p, yaw=p.yaw + rng.choice([-1, 1]) * 0.9)
        return kind, (p,), 1.0
    kind = rng.choice(["handshape", "orientation", "movement", "incomplete"])
    if kind == "handshape":
        f = rng.choice(["index", "middle", "ring", "pinky"])
        return kind, tuple(replace(k, **{f: 1.0 - getattr(k, f)}) for k in keys), 1.0
    if kind == "orientation":
        return kind, tuple(replace(k, yaw=k.yaw + 0.9) for k in keys), 1.0
    if kind == "movement":
        return kind, tuple(reversed(keys)), 1.0
    return kind, keys, 0.5  # incomplete: 절반에서 멈춤


def generate(
    *,
    n_participants: int = 5,
    n_dynamic: int = 7,
    n_static: int = 4,
    correct: int = 6,
    near_miss: int = 2,
    still: int = 1,
    unrelated: int = 6,
    static_hold_ms: int = 1500,
    dropout: float = 0.04,
    seed: int = 7,
) -> tuple[list[dict], dict[str, list[list[list[float]]]]]:
    """(클립 레코드 목록, 기준 시퀀스 {word_key: frames}) 반환.

    기준 시퀀스는 촬영자 특성이 없는 표준 동작으로, 국립국어원 기준 데이터 역할을 한다(DTW 폴백 비교용).
    """
    vocab = make_vocabulary(n_dynamic, n_static, seed)
    people = make_participants(n_participants, seed)
    rng = random.Random(seed + 2)
    neutral = Participant("REF", 1.0, 1.0, (0.0, 0.0), 0.0, False, (1280, 720))

    references = {}
    for w in vocab:
        if w.word_type == "static":
            references[w.key] = [hand_frame(w.keyposes[0])]
        else:
            n = _n_frames(w.duration_ms)
            references[w.key] = [hand_frame(_pose_at(w.keyposes, k / (n - 1))) for k in range(n)]

    records = []

    def emit(p: Participant, word_key: str, word_type: str, label: str, frames: list[dict], error_type=None):
        records.append(
            {
                "sample_id": str(uuid.UUID(int=rng.getrandbits(128))),
                "participant_id": p.pid,
                "data_source": "synthetic",
                "spec_version": spec.SPEC_VERSION,
                "protocol_version": "protocol-v0",
                "word_key": word_key,
                "word_id": None,
                "word_type": word_type,
                "label": label,
                "error_type": error_type,
                "dominant_hand": "left" if p.left_handed else "right",
                "hand_used": "left" if p.left_handed else "right",
                "frame": {"width": p.frame[0], "height": p.frame[1], "rotation": 0},
                "input_mirrored": False,
                "timing_source": "synthetic",
                "measured_fps": 1000 / FRAME_MS,
                "review": {"participant_kept": True, "qc_status": "pending"},
                "frames": frames,
            }
        )

    for p in people:
        for w in vocab:
            if w.word_type == "static":
                hold = [w.keyposes[0]] * _n_frames(static_hold_ms)
                for _ in range(correct):
                    emit(p, w.key, "static", "correct", _frames_from(hold, p, rng, dropout))
                for _ in range(near_miss):
                    kind, keys, _ = _near_miss(w, rng)
                    emit(p, w.key, "static", "near_miss", _frames_from([keys[0]] * len(hold), p, rng, dropout), kind)
                continue
            ref_ms = recording_duration_ms(0, w.duration_ms)  # 녹화 길이(스펙 §10.3)
            for _ in range(correct):
                n = _n_frames(w.duration_ms * p.speed * rng.uniform(0.9, 1.1))
                warp = rng.uniform(0.8, 1.25)
                poses = [_pose_at(w.keyposes, (k / (n - 1)) ** warp) for k in range(n)]
                lead = [w.keyposes[0]] * rng.randrange(0, 6)  # 녹화 시작 직후 잠깐 멈춤
                emit(p, w.key, "dynamic", "correct", _frames_from(lead + poses, p, rng, dropout))
            for _ in range(near_miss):
                kind, keys, progress = _near_miss(w, rng)
                n = _n_frames(w.duration_ms * p.speed)
                poses = [_pose_at(keys, progress * k / (n - 1)) for k in range(n)]
                emit(p, w.key, "dynamic", "near_miss", _frames_from(poses, p, rng, dropout), kind)
            for _ in range(still):
                t = rng.choice([0.0, 0.5])  # 시작 자세 / 중간 자세
                hold = [_pose_at(w.keyposes, t)] * _n_frames(ref_ms * 0.8)
                emit(p, w.key, "dynamic", "still", _frames_from(hold, p, rng, dropout))
        for _ in range(unrelated):
            a = HandPose(thumb=rng.random(), index=rng.random(), middle=rng.random(), ring=rng.random(),
                         pinky=rng.random(), roll=rng.uniform(-1, 1), x=rng.uniform(0.3, 0.7), y=rng.uniform(0.5, 0.9))
            b = replace(a, y=a.y - rng.uniform(0.1, 0.3), roll=a.roll + rng.uniform(-0.5, 0.5))
            n = _n_frames(rng.uniform(1500, 3000))
            emit(p, "unrelated", "dynamic", "unrelated",
                 _frames_from([_lerp(a, b, k / (n - 1)) for k in range(n)], p, rng, dropout))
    return records, references
