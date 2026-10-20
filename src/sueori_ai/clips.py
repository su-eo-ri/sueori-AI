"""수집 클립 레코드(데이터 수집 프로토콜 protocol-v0 §6) 읽기·검증·트랙 분리.

클립 1개 = JSON 객체 1개. 파일은 JSON Lines(`*.jsonl`, 한 줄에 클립 하나)로 읽는다.
수집 DB(행 1개 = 클립 1개, frames는 jsonb)에서 내보낸 스냅샷도 같은 형식으로 둔다.
⚠️ 실제 수집 데이터 스냅샷은 이 레포에 두지 않는다(.gitignore의 data/, 동의서 운영 메모).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional

from . import spec

LABELS = ("correct", "near_miss", "still", "unrelated")
WORD_TYPES = ("static", "dynamic")
_REQUIRED = ("sample_id", "participant_id", "word_key", "word_type", "label", "frames")


@dataclass
class Clip:
    sample_id: str
    participant_id: str
    word_key: str
    word_type: str
    label: str
    frames: list[dict]
    error_type: Optional[str] = None
    dominant_hand: Optional[str] = None
    qc_status: Optional[str] = None
    spec_version: Optional[str] = None
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def is_static(self) -> bool:
        return self.word_type == "static"


def validate(record: dict) -> list[str]:
    """레코드 문제 목록(빈 리스트면 정상). 좌표 유효성(NaN 등)은 트랙 단계에서 거른다."""
    errors = [f"필드 없음: {k}" for k in _REQUIRED if k not in record]
    if errors:
        return errors
    if record["label"] not in LABELS:
        errors.append(f"label 값 이상: {record['label']}")
    if record["word_type"] not in WORD_TYPES:
        errors.append(f"word_type 값 이상: {record['word_type']}")
    if record["label"] == "near_miss" and not record.get("error_type"):
        errors.append("near_miss에는 error_type이 필요함")
    # 수집 클립은 원시 좌표라 spec 버전과 무관하게 다시 계산할 수 있다 → 지원 버전이면 받는다
    sv = record.get("spec_version")
    if sv is not None and sv not in spec.SUPPORTED_VERSIONS:
        errors.append(f"spec_version 미지원: {sv} (지원: {spec.SUPPORTED_VERSIONS})")
    if not isinstance(record["frames"], list):
        errors.append("frames는 리스트여야 함")
    return errors


def from_record(record: dict) -> Clip:
    errors = validate(record)
    if errors:
        raise ValueError(f"{record.get('sample_id', '?')}: " + "; ".join(errors))
    review = record.get("review") or {}
    return Clip(
        sample_id=record["sample_id"],
        participant_id=record["participant_id"],
        word_key=record["word_key"],
        word_type=record["word_type"],
        label=record["label"],
        frames=record["frames"],
        error_type=record.get("error_type"),
        dominant_hand=record.get("dominant_hand"),
        qc_status=review.get("qc_status"),
        spec_version=record.get("spec_version"),
        raw=record,
    )


def iter_records(path: Path) -> Iterator[dict]:
    files = sorted(path.rglob("*.jsonl")) if path.is_dir() else [path]
    for f in files:
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)


def load_clips(path: str | Path, *, skip_rejected: bool = True) -> list[Clip]:
    """`qc_status == "reject"` 클립은 학습에서 뺀다(프로토콜 §7.2). 삭제가 아니라 제외다."""
    clips = [from_record(r) for r in iter_records(Path(path))]
    return [c for c in clips if not (skip_rejected and c.qc_status == "reject")]


def write_clips(records: Iterable[dict], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
            n += 1
    return n


def tracks(clip: Clip) -> dict[str, list[list[list[float]]]]:
    """스펙 §3: handedness 라벨별 트랙. 처음 나타난 순서를 유지하고, 손이 없는 프레임은 버린다.

    좌표에 NaN·Infinity가 있는 프레임이 하나라도 있으면 그 트랙 전체를 채점 불가로 보고 뺀다
    (기준 구현 reference.best_track_score와 같은 동작).
    """
    out: dict[str, list] = {}
    bad: set[str] = set()
    for fr in clip.frames:
        for hand in fr.get("hands", []):
            label = hand.get("handedness") or ""
            lm = hand["landmarks"]
            out.setdefault(label, []).append(lm)
            if any(not math.isfinite(c) for p in lm for c in p):
                bad.add(label)
    return {k: v for k, v in out.items() if k not in bad}


def primary_track(clip: Clip) -> Optional[list[list[list[float]]]]:
    """학습용 대표 트랙: 프레임이 가장 많은 트랙(동률이면 먼저 나타난 트랙)."""
    best = None
    for frames in tracks(clip).values():
        if best is None or len(frames) > len(best):
            best = frames
    return best
