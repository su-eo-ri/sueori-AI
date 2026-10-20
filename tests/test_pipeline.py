"""클립 읽기·트랙, 데이터셋, 평가 판정, 합성 end-to-end."""

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from sueori_ai import evaluate as ev
from sueori_ai import spec, synthetic_clips
from sueori_ai.clips import from_record, load_clips, primary_track, tracks, validate, write_clips
from sueori_ai.dataset import LabelMap, build_examples, mirror_x, participant_folds
from sueori_ai.synthetic import HandPose, hand_frame

ROOT = Path(__file__).resolve().parent.parent
FRAME = hand_frame(HandPose())


def _rec(**kw):
    base = {
        "sample_id": "s1", "participant_id": "T001", "word_key": "syn/d0", "word_type": "dynamic",
        "label": "correct", "spec_version": spec.SPEC_VERSION,
        "frames": [{"t_ms": 33 * k, "hands": [{"handedness": "Right", "landmarks": FRAME}]} for k in range(10)],
    }
    base.update(kw)
    return base


@pytest.fixture(scope="module")
def small():
    records, refs = synthetic_clips.generate(n_participants=3, n_dynamic=2, n_static=2, correct=2, near_miss=1, still=1, unrelated=2)
    return [from_record(r) for r in records], refs


# ── 클립 ──


def test_validate_reports_problems():
    assert validate(_rec()) == []
    assert any("label" in e for e in validate(_rec(label="maybe")))
    assert any("error_type" in e for e in validate(_rec(label="near_miss")))
    assert any("spec_version" in e for e in validate(_rec(spec_version="ai-io-v9")))
    bad = _rec()
    del bad["participant_id"]
    assert validate(bad) == ["필드 없음: participant_id"]


def test_tracks_split_by_handedness_and_drop_nonfinite_track():
    nan_frame = [list(p) for p in FRAME]
    nan_frame[3][1] = math.nan
    frames = [
        {"t_ms": 0, "hands": [{"handedness": "Right", "landmarks": FRAME}, {"handedness": "Left", "landmarks": FRAME}]},
        {"t_ms": 33, "hands": []},
        {"t_ms": 66, "hands": [{"handedness": "Right", "landmarks": FRAME}, {"handedness": "Left", "landmarks": nan_frame}]},
    ]
    c = from_record(_rec(frames=frames))
    t = tracks(c)
    assert list(t) == ["Right"] and len(t["Right"]) == 2  # Left는 NaN 때문에 통째로 제외, 빈 프레임은 버림


def test_primary_track_is_longest_then_first():
    moved = hand_frame(HandPose(x=0.3))
    both = [{"t_ms": 0, "hands": [{"handedness": "A", "landmarks": FRAME}, {"handedness": "B", "landmarks": moved}]}]
    assert primary_track(from_record(_rec(frames=both)))[0] == FRAME  # 동률이면 먼저 나타난 트랙(A)
    longer = both + [{"t_ms": 33, "hands": [{"handedness": "B", "landmarks": moved}]}]
    t = primary_track(from_record(_rec(frames=longer)))
    assert len(t) == 2 and t[0] == moved  # 더 긴 트랙(B)


def test_load_skips_rejected(tmp_path):
    write_clips([_rec(sample_id="a"), _rec(sample_id="b", review={"qc_status": "reject"})], tmp_path / "x.jsonl")
    assert [c.sample_id for c in load_clips(tmp_path)] == ["a"]
    assert len(load_clips(tmp_path, skip_rejected=False)) == 2


# ── 데이터셋 ──


def test_label_map_and_labels_json_roundtrip(small):
    clips, _ = small
    lm = LabelMap.from_clips(clips)
    assert lm.classes[-1] == "other" and list(lm.words) == sorted(lm.words)
    data = lm.to_labels_json(model_version="v0.0.0-test", temperature=1.3)
    assert data["spec_version"] == spec.SPEC_VERSION and data["labels"][-1]["name"] == "other"
    assert LabelMap.from_labels_json(data) == lm


def test_examples_shapes_and_targets(small):
    clips, _ = small
    lm = LabelMap.from_clips(clips)
    ex = build_examples(clips, lm, static_frames_per_clip=3)
    assert ex.x.shape[1:] == (spec.SEQUENCE_LENGTH, spec.FEATURES) and ex.x.dtype == np.float32
    assert set(ex.y[ex.label != "correct"]) == {lm.other_index}
    for w in lm.words:
        assert (ex.y[(ex.label == "correct") & (ex.word_key == w)] == lm.index(w)).all()
    # 정적 정답 클립은 프레임 여러 장 → 같은 sample_id가 여러 예제
    static_ids = ex.sample_id[(ex.label == "correct") & np.char.startswith(ex.word_key.astype(str), "syn/s")]
    assert len(static_ids) > len(set(static_ids))


def test_participant_folds_never_leak():
    for train_ps, held in participant_folds(["T1", "T2", "T2", "T3"]):
        assert held not in train_ps and len(train_ps) == 2


def test_mirror_flips_only_x():
    x = np.arange(2 * 32 * 63, dtype=np.float32).reshape(2, 32, 63)
    m = mirror_x(x)
    assert (m[..., 0::3] == -x[..., 0::3]).all() and (m[..., 1::3] == x[..., 1::3]).all() and (m[..., 2::3] == x[..., 2::3]).all()


# ── 평가 ──


def test_pairs_categories(small):
    clips, _ = small
    lm = LabelMap.from_clips(clips)
    pairs = ev.make_pairs(clips, lm)
    cats = {p.category for p in pairs}
    assert cats == set(ev.CATEGORIES)
    n_correct = sum(c.label == "correct" for c in clips)
    assert sum(p.category == "other_word" for p in pairs) == n_correct * (len(lm.words) - 1)


def test_dtw_scores_match_fallback_semantics(small):
    clips, refs = small
    lm = LabelMap.from_clips(clips)
    wt = {c.word_key: c.word_type for c in clips if c.word_key in lm.words}
    pairs = [p for p in ev.make_pairs(clips, lm) if p.category == "correct"]
    vals = ev.score_pairs_dtw(pairs, refs, wt)
    assert np.nanmedian(vals) > 50  # 같은 단어끼리는 대체로 높게


def _summ(correct, other, still=0.0, unrelated=0.0, near=60.0, n=1000):
    return {
        "correct": {"n": n, "pass_rate": correct, "median": 90.0, "max": 100.0},
        "other_word": {"n": n, "pass_rate": other, "median": 5.0, "max": 50.0},
        "still": {"n": n, "pass_rate": still, "median": 10.0, "max": 60.0},
        "unrelated": {"n": n, "pass_rate": unrelated, "median": 5.0, "max": 60.0},
        "near_miss": {"n": n, "pass_rate": 0.1, "median": near, "max": 85.0},
    }


def test_go_decision_rules():
    assert ev.go_decision(_summ(0.98, 0.0), _summ(0.80, 0.0))["go"]  # +18%p, 목표 충족
    assert not ev.go_decision(_summ(0.98, 0.0), _summ(0.95, 0.0))["go"]  # 확실히 낫지 않음
    d = ev.go_decision(_summ(0.98, 0.0, still=0.05), _summ(0.80, 0.0, still=0.0))
    assert not d["go"] and "still" in d["regressions"]  # 정지 자세 통과율이 DTW보다 나빠짐
    assert not ev.go_decision(_summ(0.90, 0.0), _summ(0.70, 0.0))["meets_spec_targets"]  # 정답 95% 미달
    d = ev.go_decision(_summ(0.98, 0.0, n=10), _summ(0.80, 0.0, n=10))
    assert not d["go"] and "correct" in d["insufficient_samples"]  # 표본 부족


def test_go_decision_uses_confidence_interval():
    m, d = _summ(0.98, 0.0), _summ(0.80, 0.0)
    assert ev.go_decision(m, d, ci={"correct_gain": (0.12, 0.25)})["go"]
    # 점추정은 +18%p지만 하한이 +10%p 미만이면 "확실히 나음"이 아님
    assert not ev.go_decision(m, d, ci={"correct_gain": (0.03, 0.30)})["clearly_better"]
    # 점추정은 같아도 상한이 +2%p를 넘으면 퇴보로 본다
    assert "still" in ev.go_decision(m, d, ci={"correct_gain": (0.12, 0.25), "still_diff": (-0.01, 0.04)})["regressions"]


class _C:
    def __init__(self, pid, hand="right"):
        self.participant_id, self.dominant_hand, self.sample_id = pid, hand, pid


def _pairs(spec_):
    """[(참여자, 범주, 타깃, 모델 점수, DTW 점수, 손)] → (pairs, m, d)"""
    pairs = [ev.Pair(_C(p, h), t, c) for p, c, t, _, _, h in spec_]
    return pairs, np.array([m for *_, m, _, _ in spec_], float), np.array([d for *_, _, d, _ in spec_], float)


def test_bootstrap_ci_brackets_point_estimate():
    rows = []
    for k in range(6):
        rows += [(f"P{k}", "correct", "w", 90.0, 90.0 if k < 3 else 50.0, "right")] * 5
    pairs, m, d = _pairs(rows)
    ci = ev.bootstrap_ci(pairs, m, d, resamples=500, seed=1)
    lo, hi = ci["correct_gain"]
    assert lo <= 0.5 <= hi and lo >= 0.0  # 점추정 +50%p, 모델이 절대 나쁘지 않음


def test_per_word_floor_and_left_hand_gap():
    rows = [("P1", "correct", "a", 90.0, 90.0, "right")] * 9 + [("P1", "correct", "b", 50.0, 90.0, "right")] * 9
    rows += [("P2", "correct", "a", 90.0, 90.0, "left")] * 5 + [("P2", "correct", "a", 40.0, 90.0, "left")] * 5
    pairs, m, d = _pairs(rows)
    words = ev.per_word(pairs, m, d)
    assert words["b"]["model_pass_rate"] == 0.0 and words["b"]["below_floor"]
    assert words["a"]["model_pass_rate"] == 14 / 19 and words["a"]["below_floor"]  # 73.7% < 80% 바닥선
    assert words["a"]["dtw_pass_rate"] == 1.0
    only_a = [i for i, p in enumerate(pairs) if p.target == "a"]  # 오른손 a 9/9, 왼손 a 5/10
    lh = ev.left_hand_gap([pairs[i] for i in only_a], m[only_a])
    assert lh["status"] == "measured" and lh["right"]["pass_rate"] == 1.0 and lh["left"]["pass_rate"] == 0.5
    assert lh["gap"] == 0.5 and lh["ok"] is False


def test_left_hand_no_data_does_not_block():
    pairs, m, _ = _pairs([("P1", "correct", "a", 90.0, 90.0, "right")])
    assert ev.left_hand_gap(pairs, m)["ok"] is None


def test_decide_groups_and_disabled_words(small):
    clips, refs = small
    lm = LabelMap.from_clips(clips)
    wt = {c.word_key: c.word_type for c in clips if c.word_key in lm.words}
    pairs = ev.make_pairs(clips, lm)
    d = ev.score_pairs_dtw(pairs, refs, wt)
    out = ev.decide(pairs, d, d, wt, seed=0)  # 모델 = DTW로 두면 "확실히 나음"이 아니므로 No-Go
    assert set(out["groups"]) == {"dynamic", "static"}
    assert not any(g["decision"]["go"] for g in out["groups"].values())
    assert out["model_enabled_words"] == []
    assert set(out["per_word"]) == set(lm.words)


# ── end-to-end (합성, 작게) ──


def test_pipeline_end_to_end_synthetic(tmp_path):
    out = tmp_path / "run"
    proc = subprocess.run(
        [sys.executable, "tools/train_pipeline.py", "--synthetic", "--participants", "3", "--epochs", "2",
         "--max-folds", "1", "--out", str(out)],
        cwd=ROOT, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    for f in ("report.json", "report.md", "labels.json", "manifest.draft.json", "conversion-vectors.json",
              "artifacts/android/model.tflite", "artifacts/ios/model.tflite", "artifacts/web/model.json"):
        assert (out / f).exists(), f
    report = json.loads((out / "report.json").read_text())
    assert report["synthetic"] and report["conversion"]["tflite"]["pass"]
    manifest = json.loads((out / "manifest.draft.json").read_text())
    assert manifest["enabled"] is False and manifest["artifacts"]["web"]["enabled"] is False  # CTO 조건
    assert "disabled_words" in manifest and report["final"]["tau_oof"] is not None
    assert set(report["decision"]["groups"]) == {"dynamic", "static"}
