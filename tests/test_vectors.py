"""커밋된 공통 테스트 벡터가 기준 구현과 일치하는지 확인한다.

클라이언트(TS·Dart·Swift) 테스트도 이 파일과 같은 방식으로 벡터를 읽어 검증하면 된다.
"""

import json
import math
from pathlib import Path

import pytest

from sueori_ai import reference as ref
from sueori_ai import spec

VECTORS = Path(__file__).resolve().parent.parent / "vectors" / f"{spec.SPEC_VERSION}-vectors.json"
DATA = json.loads(VECTORS.read_text(encoding="utf-8"))
TOL = DATA["tolerances"]


def _dec(x):
    if isinstance(x, str):
        return {"NaN": math.nan, "Infinity": math.inf, "-Infinity": -math.inf}[x]
    if isinstance(x, list):
        return [_dec(v) for v in x]
    return x


def _ids(cases):
    return [c["name"] for c in cases]


def _max_err(a, b):
    if isinstance(a, list):
        return max(_max_err(x, y) for x, y in zip(a, b, strict=True))
    return abs(a - b)


def test_spec_version():
    assert DATA["spec_version"] == spec.SPEC_VERSION


def test_vectors_file_is_up_to_date():
    from tools_gen import render  # noqa: E402  (tests/conftest.py가 tools/를 경로에 추가)

    assert VECTORS.read_text(encoding="utf-8") == render(), "tools/gen_vectors.py를 다시 실행할 것"


@pytest.mark.parametrize("case", DATA["normalize"], ids=_ids(DATA["normalize"]))
def test_normalize(case):
    frame = _dec(case["frame"])
    if case.get("invalid"):
        with pytest.raises(ref.InvalidInput):
            ref.normalize_frame(frame)
        return
    assert _max_err(ref.normalize_frame(frame), case["expected"]) <= TOL["normalize_f64"]


@pytest.mark.parametrize("case", DATA["tensor"], ids=_ids(DATA["tensor"]))
def test_tensor(case):
    frames = _dec(case["frames"])
    fn = ref.static_tensor if case["kind"] == "static" else ref.dynamic_tensor
    arg = frames[0] if case["kind"] == "static" else frames
    if case.get("invalid"):
        with pytest.raises(ref.InvalidInput):
            fn(arg)
        return
    got = fn(arg)
    assert len(got) == spec.SEQUENCE_LENGTH and len(got[0]) == spec.FEATURES
    assert _max_err(got, case["expected"]) <= TOL["tensor_f64"]


@pytest.mark.parametrize("case", DATA["static_score"], ids=_ids(DATA["static_score"]))
def test_static_score(case):
    r, c = _dec(case["reference"]), _dec(case["candidate"])
    if case.get("invalid"):
        with pytest.raises(ref.InvalidInput):
            ref.static_score(r, c)
        return
    got = ref.static_score(r, c)
    assert abs(got.distance - case["expected"]["distance"]) <= TOL["distance"]
    assert abs(got.score - case["expected"]["score"]) <= TOL["score"]


@pytest.mark.parametrize("case", DATA["dtw"], ids=_ids(DATA["dtw"]))
def test_dtw(case):
    r, c = _dec(case["reference"]), _dec(case["candidate"])
    if case.get("invalid"):
        with pytest.raises(ref.InvalidInput):
            ref.score_dynamic(r, c)
        return
    # 기대값은 §10.2 알고리즘 수준(dtw_score). §4 유효성(N<8)은 invalid 케이스에서 score_dynamic으로 확인한다
    got = ref.dtw_score(r, c)
    assert got.path_length == case["expected"]["path_length"]
    assert abs(got.avg_cost - case["expected"]["avg_cost"]) <= TOL["avg_cost"]
    assert abs(got.score - case["expected"]["score"]) <= TOL["score"]


@pytest.mark.parametrize("case", DATA["best_track"], ids=_ids(DATA["best_track"]))
def test_best_track(case):
    tracks = {t["label"]: _dec(t["frames"]) for t in case["tracks"]}
    got = ref.best_track_score(_dec(case["reference"]), tracks, dynamic=case["kind"] == "dynamic")
    if case["expected"] is None:
        assert got is None
    else:
        assert got[0] == case["expected"]["label"]
        assert abs(got[1] - case["expected"]["score"]) <= TOL["score"]


@pytest.mark.parametrize("case", DATA["recording_duration"])
def test_recording_duration(case):
    assert ref.recording_duration_ms(case["first_t_ms"], case["last_t_ms"]) == case["expected"]
