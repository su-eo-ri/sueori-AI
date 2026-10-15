import json
import math
import random

import pytest

from sueori_ai import reference as ref
from sueori_ai import spec
from sueori_ai.synthetic import HandPose, hand_frame, jitter, motion

OPEN = hand_frame(HandPose())
FIST = hand_frame(HandPose(thumb=0.9, index=1, middle=1, ring=1, pinky=1))


# ── §2 정규화 ──


def test_normalize_wrist_is_origin_and_palm_scale_is_one():
    v = ref.normalize_frame(OPEN)
    assert v[0:3] == [0.0, 0.0, 0.0]
    pts = [v[i * 3 : i * 3 + 3] for i in range(21)]
    assert ref.palm_scale(pts) == pytest.approx(1.0, abs=1e-12)


def test_normalize_invariant_to_translation_and_scale_not_rotation():
    base = ref.normalize_frame(OPEN)
    moved = ref.normalize_frame(hand_frame(HandPose(x=0.2, y=0.4, scale=1.7)))
    rotated = ref.normalize_frame(hand_frame(HandPose(roll=0.8)))
    assert max(abs(a - b) for a, b in zip(base, moved)) < 1e-9
    # 회전은 정규화하지 않는다(손바닥 방향이 의미를 구분하므로)
    assert max(abs(a - b) for a, b in zip(base, rotated)) > 0.1


def test_degenerate_scale_falls_back_to_one():
    frame = [[0.3, 0.3, 0.0]] * 21
    assert ref.palm_scale(frame) == 1.0
    assert ref.normalize_frame(frame) == [0.0] * 63


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_nonfinite_coordinates_are_rejected(bad):
    frame = [list(p) for p in OPEN]
    frame[5][0] = bad
    with pytest.raises(ref.InvalidInput):
        ref.normalize_frame(frame)


def test_wrong_landmark_count_is_rejected():
    with pytest.raises(ref.InvalidInput):
        ref.normalize_frame(OPEN[:20])


# ── §4 리샘플 ──


def test_resample_keeps_endpoints_and_length():
    vecs = [ref.normalize_frame(f) for f in motion([HandPose(), HandPose(index=1)], 45)]
    out = ref.resample(vecs)
    assert len(out) == spec.SEQUENCE_LENGTH
    assert out[0] == vecs[0]
    assert out[-1] == vecs[-1]


def test_resample_identity_when_n_equals_t():
    vecs = [ref.normalize_frame(f) for f in motion([HandPose(), HandPose(index=1)], 32)]
    assert ref.resample(vecs) == vecs


def test_resample_single_frame_repeats():
    v = ref.normalize_frame(OPEN)
    assert ref.resample([v]) == [v] * 32


def test_resample_midpoint_is_linear_interpolation():
    a, b = [0.0] * 63, [1.0] * 63
    out = ref.resample([a, b], length=3)
    assert out[1] == [0.5] * 63


def test_dynamic_tensor_requires_eight_frames():
    frames = motion([HandPose(), HandPose(index=1)], 8)
    assert len(ref.dynamic_tensor(frames)) == 32
    with pytest.raises(ref.InvalidInput):
        ref.dynamic_tensor(frames[:7])


def test_static_tensor_repeats_normalized_frame():
    t = ref.static_tensor(OPEN)
    assert len(t) == 32 and all(row == t[0] for row in t)


# ── §10 폴백 ──


def test_static_score_identical_is_100_and_different_pose_is_low():
    assert ref.static_score(OPEN, OPEN).score == 100.0
    assert ref.static_score(OPEN, FIST).score < 40


def test_landmark_distance_is_mean_of_point_distances():
    a = [0.0] * 63
    b = [0.0] * 63
    b[0:3] = [3.0, 4.0, 0.0]  # 점 0만 거리 5
    assert ref.landmark_distance(a, b) == 5.0 / 21


def test_dtw_identical_sequence_scores_100_with_diagonal_path():
    seq = motion([HandPose(), HandPose(index=1)], 20)
    r = ref.dtw_score(seq, seq)
    assert (r.score, r.avg_cost, r.path_length) == (100.0, 0.0, 20)


def test_dtw_is_speed_tolerant():
    keys = [HandPose(index=1, y=0.8), HandPose(y=0.6)]
    rng = random.Random(1)
    slow = [jitter(f, rng, 0.001) for f in motion(keys, 50, warp=1.5)]
    other = motion([HandPose(yaw=-0.6, x=0.4), HandPose(yaw=0.6, x=0.6)], 30)
    same = ref.dtw_score(motion(keys, 25), slow).score
    diff = ref.dtw_score(motion(keys, 25), other).score
    assert same > 80 > diff


def test_dtw_empty_sequence_scores_zero():
    r = ref.dtw_score([], [OPEN])
    assert r.score == 0.0 and r.path_length == 0


def _path_length_with_order(d, n, m, order):
    i, j, length = n, m, 0
    while i > 0 or j > 0:
        length += 1
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            cand = {"diag": d[i - 1][j - 1], "up": d[i - 1][j], "left": d[i][j - 1]}
            best = min(cand.values())
            pick = next(k for k in order if cand[k] == best)
            if pick == "diag":
                i, j = i - 1, j - 1
            elif pick == "up":
                i -= 1
            else:
                j -= 1
    return length


V_SIGN = hand_frame(HandPose(thumb=0.8, ring=1, pinky=1))


def _dp(ref_frames, cand_frames):
    a = [ref.normalize_frame(f) for f in ref_frames]
    b = [ref.normalize_frame(f) for f in cand_frames]
    n, m = len(a), len(b)
    d = [[math.inf] * (m + 1) for _ in range(n + 1)]
    d[0][0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = ref.landmark_distance(a[i - 1], b[j - 1]) + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])
    return d, n, m


def test_tie_break_rule_is_actually_exercised():
    """벡터의 동률 케이스는 스펙 순서(diag→up→left)일 때만 기대 경로 길이가 나와야 한다.

    다른 순서로 구현한 클라이언트가 벡터를 통과하지 못하게 하려는 것이다.
    """
    import itertools

    r, c = [OPEN, V_SIGN, OPEN, V_SIGN], [OPEN, FIST, V_SIGN]
    d, n, m = _dp(r, c)
    spec_len = ref.dtw_score(r, c).path_length
    assert spec_len == _path_length_with_order(d, n, m, ["diag", "up", "left"]) == 4
    for order in itertools.permutations(["diag", "up", "left"]):
        if list(order) != ["diag", "up", "left"]:
            assert _path_length_with_order(d, n, m, list(order)) != spec_len, order


def test_score_dynamic_rejects_short_candidate():
    seq = motion([HandPose(), HandPose(index=1)], 20)
    with pytest.raises(ref.InvalidInput):
        ref.score_dynamic(seq, seq[:7])


def test_best_track_picks_highest_and_skips_invalid():
    seq = motion([HandPose(), HandPose(index=1)], 20)
    nan_frame = [list(p) for p in OPEN]
    nan_frame[0][0] = math.nan
    tracks = {"Left": [nan_frame] * 20, "Right": seq, "": seq[:3]}
    assert ref.best_track_score(seq, tracks, dynamic=True) == ("Right", 100.0)
    assert ref.best_track_score(seq, {"Left": [nan_frame] * 20}, dynamic=True) is None


def test_best_track_keeps_first_on_tie():
    seq = motion([HandPose(), HandPose(index=1)], 20)
    assert ref.best_track_score(seq, {"A": seq, "B": seq}, dynamic=True)[0] == "A"


@pytest.mark.parametrize(
    "first,last,expected",
    [(None, None, 3000), (0, None, 3000), (10, 10, 3000), (0, 1000, 2000), (0, 2500, 3000), (0, 9000, 8000)],
)
def test_recording_duration(first, last, expected):
    assert ref.recording_duration_ms(first, last) == expected


def test_recording_duration_rounds_half_away_from_zero():
    # 2083.75 × 1.2 = 2500.5 (정확히 표현됨) → 2501
    assert ref.recording_duration_ms(0, 2083.75) == 2501
    # floor(x + 0.5)는 0.49999999999999994 + 0.5 = 1.0으로 틀린다 — 소수부 비교로 피함
    assert ref.recording_duration_ms(0, (2000 + 0.49999999999999994) / 1.2) in (2000, 2001)


# ── v1 폴백 좌우 반전 ──


def _left(frame):
    return [[1.0 - x, y, z] for x, y, z in frame]


def test_v0_fails_left_hand_and_v1_fixes_it():
    seq = motion([HandPose(), HandPose(index=1, roll=0.5)], 20)
    left = [_left(f) for f in seq]
    v0 = ref.dtw_score(seq, left, spec_version="ai-io-v0")
    v1 = ref.dtw_score(seq, left, spec_version="ai-io-v1")
    assert v0.score < 50 and not v0.mirrored
    assert v1.score > 99.9 and v1.mirrored
    s0 = ref.static_score(V_SIGN, _left(V_SIGN), spec_version="ai-io-v0")
    s1 = ref.static_score(V_SIGN, _left(V_SIGN), spec_version="ai-io-v1")
    assert s0.score < 80 and s1.score > 99.9 and s1.mirrored


def test_v1_right_hand_scores_unchanged():
    seq = motion([HandPose(), HandPose(index=1, roll=0.5)], 20)
    other = motion([HandPose(yaw=-0.6, x=0.4), HandPose(yaw=0.6, x=0.6)], 25)
    for cand in (seq, other):
        a, b = ref.dtw_score(seq, cand, spec_version="ai-io-v0"), ref.dtw_score(seq, cand, spec_version="ai-io-v1")
        assert b.score >= a.score  # v1은 max라 낮아질 수 없다


def test_mirror_tie_keeps_original():
    flat = [[OPEN[0][0], y, z] for _, y, z in OPEN]  # 정규화 후 x가 전부 0 → 반전해도 같음
    r = ref.static_score(OPEN, flat, spec_version="ai-io-v1")
    assert not r.mirrored


def test_unknown_spec_version_rejected():
    with pytest.raises(ValueError):
        ref.static_score(OPEN, OPEN, spec_version="ai-io-v9")
