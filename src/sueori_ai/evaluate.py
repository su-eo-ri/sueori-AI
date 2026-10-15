"""평가: 같은 (클립, 타깃 단어) 쌍을 모델과 DTW 폴백으로 각각 채점해서 스펙 §6 보정 목표 표를 만든다.

채점은 클라이언트와 같은 방식으로 한다.
- 트랙별로 채점하고 최고점(스펙 §3). 채점할 수 없는 트랙은 건너뛴다
- 입력 모양은 **타깃 단어의 유형**을 따른다. 정적 단어면 트랙 가운데 프레임 1장(채점 순간의 현재 프레임),
  동적 단어면 트랙 전체(스펙 §4)
- 폴백은 정적 = 거리 점수, 동적 = DTW(스펙 §10), 기준 시퀀스와 비교

평가 범주(스펙 §6)
- correct: 정답 클립 vs 자기 단어
- other_word: 정답 클립 vs 다른 모든 단어
- still: 정지 자세 클립 vs 자기 단어
- unrelated: 무관 동작 클립 vs 모든 단어
- near_miss: 근접 오답 클립 vs 자기 단어
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

from . import fast, spec
from .clips import Clip, tracks
from .dataset import LabelMap
from .model import scores as model_scores
from .reference import InvalidInput

CATEGORIES = ("correct", "other_word", "still", "unrelated", "near_miss")

# 스펙 §6 보정 목표
TARGETS = {
    "correct": {"pass_rate_min": 0.95, "median_range": (85.0, 95.0)},
    "other_word": {"pass_rate_max": 0.01, "max_score": 60.0},
    "still": {"pass_rate_max": 0.01, "max_score": 70.0},
    "unrelated": {"pass_rate_max": 0.01, "max_score": 70.0},
    "near_miss": {"median_range": (40.0, 75.0)},
}


@dataclass
class Pair:
    clip: Clip
    target: str
    category: str


def make_pairs(clips: Sequence[Clip], label_map: LabelMap) -> list[Pair]:
    words = label_map.words
    pairs = []
    for c in clips:
        if c.label == "correct" and c.word_key in words:
            pairs.append(Pair(c, c.word_key, "correct"))
            pairs += [Pair(c, w, "other_word") for w in words if w != c.word_key]
        elif c.label in ("still", "near_miss") and c.word_key in words:
            pairs.append(Pair(c, c.word_key, c.label))
        elif c.label == "unrelated":
            pairs += [Pair(c, w, "unrelated") for w in words]
    return pairs


def _candidate_inputs(clip: Clip, target_static: bool) -> list:
    """트랙별 입력(정적 타깃이면 가운데 프레임 1장, 동적이면 전체). 유효하지 않은 트랙은 뺀다."""
    out = []
    for frames in tracks(clip).values():
        if target_static:
            out.append([frames[len(frames) // 2]])
        elif len(frames) >= spec.MIN_DYNAMIC_FRAMES:
            out.append(frames)
    return out


def score_pairs_dtw(pairs: Sequence[Pair], references: dict, word_types: dict) -> np.ndarray:
    """폴백 점수. 채점할 수 없으면 NaN."""
    out = np.full(len(pairs), np.nan)
    for i, p in enumerate(pairs):
        static = word_types[p.target] == "static"
        best = None
        for cand in _candidate_inputs(p.clip, static):
            try:
                s = fast.static_score(references[p.target][0], cand[0]) if static else fast.dtw_score(references[p.target], cand).score
            except InvalidInput:
                continue
            best = s if best is None or s > best else best
        if best is not None:
            out[i] = best
    return out


def score_pairs_model(
    pairs: Sequence[Pair], predict: Callable[[np.ndarray], np.ndarray], label_map: LabelMap, word_types: dict, tau: float
) -> np.ndarray:
    """모델 점수. 입력을 모아 한 번에 추론한 뒤 트랙별 최고점을 고른다. 채점할 수 없으면 NaN."""
    xs, owner, targets = [], [], []
    for i, p in enumerate(pairs):
        static = word_types[p.target] == "static"
        for cand in _candidate_inputs(p.clip, static):
            try:
                xs.append(fast.static_tensor(cand[0]) if static else fast.dynamic_tensor(cand))
            except InvalidInput:
                continue
            owner.append(i)
            targets.append(label_map.index(p.target))
    out = np.full(len(pairs), np.nan)
    if not xs:
        return out
    s = model_scores(predict(np.stack(xs).astype(np.float32)), np.asarray(targets), tau)
    for i, v in zip(owner, s):
        out[i] = v if np.isnan(out[i]) else max(out[i], v)
    return out


def summarize(pairs: Sequence[Pair], values: np.ndarray) -> dict:
    cats = np.array([p.category for p in pairs])
    res = {}
    for c in CATEGORIES:
        v = values[cats == c]
        n_total = int(len(v))
        v = v[~np.isnan(v)]
        if n_total == 0:
            continue
        res[c] = {
            "n": n_total,
            "unscorable": n_total - int(len(v)),
            "pass_rate": float((v >= spec.PASS_THRESHOLD).mean()) if len(v) else None,
            "median": float(np.median(v)) if len(v) else None,
            "max": float(v.max()) if len(v) else None,
        }
    return res


def check_targets(summary: dict) -> dict:
    """스펙 §6 목표 충족 여부(범주별)."""
    out = {}
    for c, t in TARGETS.items():
        s = summary.get(c)
        if not s or s["pass_rate"] is None:
            out[c] = None
            continue
        ok = True
        if "pass_rate_min" in t:
            ok &= s["pass_rate"] >= t["pass_rate_min"]
        if "pass_rate_max" in t:
            ok &= s["pass_rate"] <= t["pass_rate_max"]
        if "max_score" in t:
            ok &= s["max"] <= t["max_score"]
        if "median_range" in t:
            lo, hi = t["median_range"]
            ok &= lo <= s["median"] <= hi
        out[c] = bool(ok)
    return out


# Go/No-Go 판정 — ⚠️ 초안. 임계값은 PM·사용자 합의가 필요하다("모델이 DTW보다 확실히 나은가").
GO_DRAFT = {
    "correct_pass_gain_min": 0.10,  # 정답 통과율이 DTW보다 10%p 이상 높음
    "false_pass_ratio_max": 0.5,  # 또는 오답(다른 단어·정지·무관) 통과율이 DTW의 절반 이하
    "regression_tolerance": 0.02,  # 어느 범주에서도 DTW보다 2%p 넘게 나빠지지 않음
}


def go_decision(model: dict, dtw: dict) -> dict:
    neg = ("other_word", "still", "unrelated")

    def rate(s, c):
        return s.get(c, {}).get("pass_rate")

    reasons, ok_gain = [], False
    mc, dc = rate(model, "correct"), rate(dtw, "correct")
    if mc is not None and dc is not None and mc - dc >= GO_DRAFT["correct_pass_gain_min"]:
        ok_gain = True
        reasons.append(f"정답 통과율 +{(mc - dc) * 100:.1f}%p")
    m_neg = [rate(model, c) for c in neg if rate(model, c) is not None]
    d_neg = [rate(dtw, c) for c in neg if rate(dtw, c) is not None]
    if m_neg and d_neg and sum(d_neg) > 0 and sum(m_neg) <= GO_DRAFT["false_pass_ratio_max"] * sum(d_neg):
        ok_gain = True
        reasons.append("오답 통과율 DTW의 절반 이하")
    regressions = []
    if mc is not None and dc is not None and mc < dc - GO_DRAFT["regression_tolerance"]:
        regressions.append("correct")
    for c in neg:
        m, d = rate(model, c), rate(dtw, c)
        if m is not None and d is not None and m > d + GO_DRAFT["regression_tolerance"]:
            regressions.append(c)
    targets = check_targets(model)
    meets = all(v for v in targets.values() if v is not None)
    return {
        "go": bool(meets and ok_gain and not regressions),
        "meets_spec_targets": meets,
        "spec_targets": targets,
        "clearly_better": ok_gain,
        "better_reasons": reasons,
        "regressions": regressions,
        "criteria": GO_DRAFT,
        "note": "초안 기준 — PM·사용자 합의 필요",
    }


def top1_accuracy(pairs: Sequence[Pair], predict, label_map: LabelMap, word_types: dict) -> Optional[float]:
    """정답 클립을 자기 단어 유형 입력으로 넣었을 때 argmax가 자기 단어인 비율(참고용)."""
    xs, ys = [], []
    for p in pairs:
        if p.category != "correct":
            continue
        cands = _candidate_inputs(p.clip, word_types[p.target] == "static")
        if not cands:
            continue
        cand = max(cands, key=len)
        try:
            xs.append(fast.static_tensor(cand[0]) if word_types[p.target] == "static" else fast.dynamic_tensor(cand))
        except InvalidInput:
            continue
        ys.append(label_map.index(p.target))
    if not xs:
        return None
    pred = predict(np.stack(xs).astype(np.float32)).argmax(axis=-1)
    return float((pred == np.asarray(ys)).mean())
