"""클립 → 학습 예제([32, 63] 텐서 + 클래스), 라벨 맵, 촬영자 단위 분할.

- 클래스: 학습 대상 단어(정렬된 word_key) + 마지막에 `other`(스펙 §5)
- `correct` 클립 → 그 단어 클래스. `near_miss`·`still`·`unrelated` → `other`(스펙 §9.1, 프로토콜 §2)
- 학습에는 클립의 대표 트랙(프레임이 가장 많은 손)을 쓴다. 채점·평가는 트랙별 최고점(스펙 §3, evaluate.py)
- 정적 클립(1.5초 유지)은 유지 구간 가운데에서 프레임 몇 장을 뽑아 각각 예제로 쓴다(프로토콜 §3.2)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Sequence

import numpy as np

from . import fast, spec
from .clips import Clip, primary_track
from .reference import InvalidInput

OTHER = "other"


@dataclass(frozen=True)
class LabelMap:
    words: tuple[str, ...]  # 정렬된 word_key, other 제외

    @classmethod
    def from_clips(cls, clips: Sequence[Clip]) -> "LabelMap":
        return cls(tuple(sorted({c.word_key for c in clips if c.label == "correct"})))

    @property
    def classes(self) -> tuple[str, ...]:
        return self.words + (OTHER,)

    @property
    def other_index(self) -> int:
        return len(self.words)

    def index(self, word_key: str) -> int:
        return self.words.index(word_key)

    def to_labels_json(self, *, model_version: str, temperature: float, word_ids: Optional[dict] = None) -> dict:
        """스펙 §5 labels.json. word_id는 BE `words.id`가 정해지면 채운다(지금은 word_key만)."""
        word_ids = word_ids or {}
        labels = [{"index": i, "word_key": w, "word_id": word_ids.get(w)} for i, w in enumerate(self.words)]
        labels.append({"index": self.other_index, "word_key": None, "word_id": None, "name": OTHER})
        return {"spec_version": spec.SPEC_VERSION, "model_version": model_version, "temperature": temperature, "labels": labels}

    @classmethod
    def from_labels_json(cls, data: dict) -> "LabelMap":
        if data["spec_version"] != spec.SPEC_VERSION:
            raise ValueError(f"spec_version 불일치: {data['spec_version']}")
        words = [l["word_key"] for l in sorted(data["labels"], key=lambda l: l["index"]) if l.get("name") != OTHER]
        return cls(tuple(words))


@dataclass
class Examples:
    x: np.ndarray  # [N, 32, 63] float32
    y: np.ndarray  # [N] int
    participant: np.ndarray  # [N] str
    sample_id: np.ndarray  # [N] str
    label: np.ndarray  # [N] str (correct / near_miss / still / unrelated)
    word_key: np.ndarray  # [N] str

    def __len__(self) -> int:
        return len(self.y)

    def subset(self, mask: np.ndarray) -> "Examples":
        return Examples(*(getattr(self, f)[mask] for f in ("x", "y", "participant", "sample_id", "label", "word_key")))


def _static_frames(track: list, k: int) -> list:
    """유지 구간 가운데 60%에서 k장을 고르게 뽑는다(시작·끝의 손이 움직이는 구간을 피함)."""
    n = len(track)
    lo, hi = int(n * 0.2), max(int(n * 0.8), int(n * 0.2) + 1)
    idx = np.linspace(lo, hi - 1, num=min(k, hi - lo)).round().astype(int)
    return [track[i] for i in sorted(set(idx.tolist()))]


def clip_tensors(clip: Clip, *, static_frames_per_clip: int = 3) -> list[np.ndarray]:
    """클립 → [32, 63] 텐서 목록. 채점할 수 없는 클립(트랙 없음, 동적 N<8 등)은 빈 리스트."""
    track = primary_track(clip)
    if not track:
        return []
    try:
        if clip.is_static:
            return [fast.static_tensor(f) for f in _static_frames(track, static_frames_per_clip)]
        return [fast.dynamic_tensor(track)]
    except InvalidInput:
        return []


def build_examples(clips: Sequence[Clip], label_map: LabelMap, *, static_frames_per_clip: int = 3) -> Examples:
    xs, ys, ps, ss, ls, ws = [], [], [], [], [], []
    for c in clips:
        if c.label == "correct" and c.word_key not in label_map.words:
            continue  # 라벨 맵에 없는 단어(예: 나중에 추가된 콘텐츠)는 학습 대상이 아님
        y = label_map.index(c.word_key) if c.label == "correct" else label_map.other_index
        for t in clip_tensors(c, static_frames_per_clip=static_frames_per_clip):
            xs.append(t.astype(np.float32))
            ys.append(y)
            ps.append(c.participant_id)
            ss.append(c.sample_id)
            ls.append(c.label)
            ws.append(c.word_key)
    if not xs:
        raise ValueError("학습 예제가 없음")
    return Examples(np.stack(xs), np.asarray(ys), np.asarray(ps), np.asarray(ss), np.asarray(ls), np.asarray(ws))


def single_participant_words(clips: Sequence[Clip]) -> list[str]:
    """정답(correct) 클립의 촬영자가 1명뿐인 word_key 목록(정렬).

    촬영자 단위 교차 검증에서 그 사람이 평가 촬영자가 되면 fold 모델은 그 단어를 학습하지 못한다.
    수집 프로토콜 필수 조건 "단어마다 촬영자 2명 이상"의 검사용(CTO 2026-10-21).
    """
    people: dict[str, set[str]] = {}
    for c in clips:
        if c.label == "correct":
            people.setdefault(c.word_key, set()).add(c.participant_id)
    return sorted(w for w, ps in people.items() if len(ps) < 2)


def participant_folds(participants: Sequence[str]) -> Iterator[tuple[list[str], str]]:
    """촬영자 단위 교차 검증(leave-one-participant-out, 스펙 §9.3·프로토콜 §8): (학습 촬영자들, 평가 촬영자)."""
    ps = sorted(set(participants))
    if len(ps) < 2:
        raise ValueError("촬영자가 2명 이상이어야 함")
    for held in ps:
        yield [p for p in ps if p != held], held


def mirror_x(x: np.ndarray) -> np.ndarray:
    """좌우 반전 증강(스펙 §3): 정규화된 좌표의 x 부호를 바꾼다."""
    out = x.copy()
    out[..., 0::3] = -out[..., 0::3]
    return out


def save_references(references: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(references, separators=(",", ":")), encoding="utf-8")


def load_references(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
