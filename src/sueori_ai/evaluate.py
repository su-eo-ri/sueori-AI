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
    """모델 점수. 채점할 수 없으면 NaN.

    입력은 (클립, 타깃 유형)에만 달려 있고 타깃 단어와는 무관하므로, (sample_id, 정적 여부, 트랙)마다
    **한 번만** 추론하고 타깃 열만 골라 쓴다(CTO 2026-10-16). 트랙별 최고점을 고른다.
    """
    keys, xs = {}, []  # (sample_id, static) -> [행 번호…]
    for p in pairs:
        static = word_types[p.target] == "static"
        key = (p.clip.sample_id, static)
        if key in keys:
            continue
        rows = []
        for cand in _candidate_inputs(p.clip, static):
            try:
                xs.append(fast.static_tensor(cand[0]) if static else fast.dynamic_tensor(cand))
            except InvalidInput:
                continue
            rows.append(len(xs) - 1)
        keys[key] = rows
    out = np.full(len(pairs), np.nan)
    if not xs:
        return out
    logits = predict(np.stack(xs).astype(np.float32))
    for i, p in enumerate(pairs):
        rows = keys[(p.clip.sample_id, word_types[p.target] == "static")]
        if rows:
            t = np.full(len(rows), label_map.index(p.target))
            out[i] = float(model_scores(logits[rows], t, tau).max())
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
    "correct_pass_gain_min": 0.10,  # ② 정답 통과율이 DTW보다 10%p 이상 높음(신뢰구간 하한 기준)
    "false_pass_ratio_max": 0.5,  # ② 또는 오답 통과율이 DTW의 절반 이하(신뢰구간 상한 기준)
    "regression_tolerance": 0.02,  # ③ 어느 범주에서도 DTW보다 2%p 넘게 나빠지지 않음(신뢰구간 기준)
    "min_pairs": {"correct": 100, "other_word": 300, "still": 50, "unrelated": 100, "near_miss": 50},  # 범주별 최소 표본
    "word_floor": 0.80,  # 단어별 바닥선: 정답 통과율이 이보다 낮은 단어는 그 단어만 DTW 유지
    "left_gap_max": 0.10,  # 왼손 정답 통과율이 오른손보다 10%p 넘게 낮으면 No-Go
    "bootstrap": {"resamples": 1000, "ci": 0.95},  # 촬영자 단위 부트스트랩
}
NEGATIVES = ("other_word", "still", "unrelated")


def _counts(pairs: Sequence[Pair], values: np.ndarray) -> dict:
    """{participant: {category: (통과 수, 채점된 수)}}"""
    out: dict = {}
    for p, v in zip(pairs, values):
        if np.isnan(v):
            continue
        d = out.setdefault(p.clip.participant_id, {})
        k, n = d.get(p.category, (0, 0))
        d[p.category] = (k + int(v >= spec.PASS_THRESHOLD), n + 1)
    return out


def _rates(counts: dict, people: Sequence[str]) -> dict:
    agg: dict = {}
    for pid in people:
        for c, (k, n) in counts.get(pid, {}).items():
            a = agg.setdefault(c, [0, 0])
            a[0] += k
            a[1] += n
    rates = {c: (k / n if n else None) for c, (k, n) in agg.items()}
    nk = sum(agg.get(c, [0, 0])[0] for c in NEGATIVES)
    nn = sum(agg.get(c, [0, 0])[1] for c in NEGATIVES)
    rates["negatives"] = nk / nn if nn else None
    return rates


def bootstrap_ci(pairs: Sequence[Pair], m_vals: np.ndarray, d_vals: np.ndarray, *, resamples: int = 1000,
                 ci: float = 0.95, seed: int = 0) -> dict:
    """촬영자 단위 부트스트랩 신뢰구간(PM 2026-10-16): 촬영자를 복원 추출해 통과율 차이·비율을 다시 계산한다.

    반환: {지표: (하한, 상한)}. 지표 = correct_gain(모델−DTW), {범주}_diff(모델−DTW), negatives_ratio(모델/DTW)
    """
    mc, dc = _counts(pairs, m_vals), _counts(pairs, d_vals)
    people = sorted({p.clip.participant_id for p in pairs})
    rng = np.random.default_rng(seed)
    stats: dict[str, list] = {}
    for _ in range(resamples):
        sample = rng.choice(people, size=len(people), replace=True)
        mr, dr = _rates(mc, sample), _rates(dc, sample)
        for c in ("correct",) + NEGATIVES:
            if mr.get(c) is not None and dr.get(c) is not None:
                stats.setdefault("correct_gain" if c == "correct" else f"{c}_diff", []).append(mr[c] - dr[c])
        if mr.get("negatives") is not None and dr.get("negatives"):
            stats.setdefault("negatives_ratio", []).append(mr["negatives"] / dr["negatives"])
    a = (1 - ci) / 2
    return {k: (float(np.quantile(v, a)), float(np.quantile(v, 1 - a))) for k, v in stats.items() if v}


def go_decision(model: dict, dtw: dict, ci: Optional[dict] = None) -> dict:
    """②·③은 신뢰구간이 있으면 구간으로, 없으면 점추정으로 판정한다. ①은 점추정. 최소 표본도 확인한다."""
    tol = GO_DRAFT["regression_tolerance"]

    def rate(s, c):
        return s.get(c, {}).get("pass_rate")

    def bounds(key, point):
        return ci[key] if ci and key in ci else (point, point)

    reasons, ok_gain = [], False
    mc, dc = rate(model, "correct"), rate(dtw, "correct")
    if mc is not None and dc is not None:
        lo, _ = bounds("correct_gain", mc - dc)
        if lo >= GO_DRAFT["correct_pass_gain_min"]:
            ok_gain = True
            reasons.append(f"정답 통과율 +{(mc - dc) * 100:.1f}%p (하한 {lo * 100:+.1f}%p)")
    m_neg = [rate(model, c) for c in NEGATIVES if rate(model, c) is not None]
    d_neg = [rate(dtw, c) for c in NEGATIVES if rate(dtw, c) is not None]
    if m_neg and d_neg and sum(d_neg) > 0:
        _, hi = bounds("negatives_ratio", sum(m_neg) / sum(d_neg))
        if hi <= GO_DRAFT["false_pass_ratio_max"]:
            ok_gain = True
            reasons.append(f"오답 통과율 DTW 대비 ×{sum(m_neg) / sum(d_neg):.2f} (상한 ×{hi:.2f})")

    regressions = []
    if mc is not None and dc is not None and bounds("correct_gain", mc - dc)[0] < -tol:
        regressions.append("correct")
    for c in NEGATIVES:
        m, d = rate(model, c), rate(dtw, c)
        if m is not None and d is not None and bounds(f"{c}_diff", m - d)[1] > tol:
            regressions.append(c)

    short = [c for c, n in GO_DRAFT["min_pairs"].items() if model.get(c, {}).get("n", 0) < n]
    targets = check_targets(model)
    meets = all(v for v in targets.values() if v is not None)
    return {
        "go": bool(meets and ok_gain and not regressions and not short),
        "meets_spec_targets": meets,
        "spec_targets": targets,
        "clearly_better": ok_gain,
        "better_reasons": reasons,
        "regressions": regressions,
        "insufficient_samples": short,
        "uses_ci": bool(ci),
        "note": "초안 기준 — PM·사용자 합의 필요",
    }


def per_word(pairs: Sequence[Pair], m_vals: np.ndarray, d_vals: np.ndarray) -> dict:
    """단어별 정답 통과율(모델·DTW)과 바닥선 판정(PM 2026-10-16)."""
    out: dict = {}
    for p, m, d in zip(pairs, m_vals, d_vals):
        if p.category != "correct":
            continue
        w = out.setdefault(p.target, {"n": 0, "m": [], "d": []})
        w["n"] += 1
        if not np.isnan(m):
            w["m"].append(m)
        if not np.isnan(d):
            w["d"].append(d)
    res = {}
    for word, w in sorted(out.items()):
        mr = float(np.mean(np.asarray(w["m"]) >= spec.PASS_THRESHOLD)) if w["m"] else None
        dr = float(np.mean(np.asarray(w["d"]) >= spec.PASS_THRESHOLD)) if w["d"] else None
        res[word] = {"n": w["n"], "model_pass_rate": mr, "dtw_pass_rate": dr,
                     "below_floor": mr is None or mr < GO_DRAFT["word_floor"]}
    return res


def per_participant(pairs: Sequence[Pair], m_vals: np.ndarray, d_vals: np.ndarray) -> dict:
    """촬영자별 정답 통과율(모델·DTW). 특정 촬영자(왼손잡이, 떨림이 큰 사람 등)에게 몰린 실패를 찾는 용도."""
    out: dict = {}
    for p, m, d in zip(pairs, m_vals, d_vals):
        if p.category != "correct":
            continue
        r = out.setdefault(p.clip.participant_id, {"hand": p.clip.dominant_hand, "m": [], "d": []})
        if not np.isnan(m):
            r["m"].append(m >= spec.PASS_THRESHOLD)
        if not np.isnan(d):
            r["d"].append(d >= spec.PASS_THRESHOLD)
    return {
        pid: {"hand": r["hand"], "n": len(r["m"]),
              "model_pass_rate": float(np.mean(r["m"])) if r["m"] else None,
              "dtw_pass_rate": float(np.mean(r["d"])) if r["d"] else None}
        for pid, r in sorted(out.items())
    }


def left_hand_gap(pairs: Sequence[Pair], m_vals: np.ndarray) -> dict:
    """왼손 vs 오른손 정답 통과율(모델). 왼손 데이터가 없으면 판정하지 않고 경고만(스펙 §3, PM 2026-10-16)."""
    rates = {}
    for hand in ("left", "right"):
        v = np.asarray([m for p, m in zip(pairs, m_vals)
                        if p.category == "correct" and (p.clip.dominant_hand or "right") == hand and not np.isnan(m)])
        rates[hand] = {"n": int(len(v)), "pass_rate": float((v >= spec.PASS_THRESHOLD).mean()) if len(v) else None}
    if rates["left"]["pass_rate"] is None or rates["right"]["pass_rate"] is None:
        return {**rates, "gap": None, "ok": None, "status": "no_data"}
    gap = rates["right"]["pass_rate"] - rates["left"]["pass_rate"]
    return {**rates, "gap": gap, "ok": gap <= GO_DRAFT["left_gap_max"], "status": "measured"}


def decide(pairs: Sequence[Pair], m_vals: np.ndarray, d_vals: np.ndarray, word_types: dict, *, seed: int = 0) -> dict:
    """그룹별 Go/No-Go + 단어별 끄기 목록(PM 2026-10-16: 부분 Go).

    그룹 = 단어 유형(dynamic = 단계 A 단어, static = 단계 B 지문자). 그룹이 Go여도 바닥선 아래 단어는
    `disabled_words`로 DTW를 유지한다. 최종 모델 사용 단어 = Go 그룹의 단어 − disabled_words.
    """
    groups = {}
    bs = GO_DRAFT["bootstrap"]
    for g in sorted(set(word_types.values())):
        idx = [i for i, p in enumerate(pairs) if word_types.get(p.target) == g]
        gp = [pairs[i] for i in idx]
        gm, gd = m_vals[idx], d_vals[idx]
        ci = bootstrap_ci(gp, gm, gd, resamples=bs["resamples"], ci=bs["ci"], seed=seed)
        dec = go_decision(summarize(gp, gm), summarize(gp, gd), ci)
        lh = left_hand_gap(gp, gm)
        if lh["ok"] is False:
            dec["go"] = False
            dec["left_hand_gap_exceeded"] = True
        groups[g] = {"decision": dec, "ci": ci, "left_hand": lh, "model": summarize(gp, gm), "dtw": summarize(gp, gd)}
    words = per_word(pairs, m_vals, d_vals)
    disabled = sorted(w for w, r in words.items() if r["below_floor"])
    enabled = sorted(w for w in words if groups.get(word_types.get(w), {}).get("decision", {}).get("go") and w not in disabled)
    return {"groups": groups, "per_word": words, "per_participant": per_participant(pairs, m_vals, d_vals),
            "disabled_words": disabled, "model_enabled_words": enabled,
            "criteria": GO_DRAFT, "note": "초안 기준 — PM·사용자 합의 필요"}


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
