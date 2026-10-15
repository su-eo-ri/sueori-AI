"""커밋된 공통 테스트 벡터가 기준 구현과 일치하는지 확인한다.

클라이언트(TS·Dart·Swift) 테스트도 이 파일과 같은 방식으로 벡터를 읽어 검증하면 된다.
"""

import json
import math
from pathlib import Path

import pytest

from sueori_ai import reference as ref
from sueori_ai import spec

VECTORS_DIR = Path(__file__).resolve().parent.parent / "vectors"
FILES = {v: json.loads((VECTORS_DIR / f"{v}-vectors.json").read_text(encoding="utf-8")) for v in spec.SUPPORTED_VERSIONS}


def _cases(section):
    """(버전, 케이스) 목록 — 모든 지원 버전 파일의 케이스를 해당 버전 규칙으로 검증한다."""
    return [(v, c) for v, d in FILES.items() for c in d[section]]


def _ids(cases):
    return [f"{v}:{c.get('name', '')}" for v, c in cases]


def _dec(x):
    if isinstance(x, str):
        return {"NaN": math.nan, "Infinity": math.inf, "-Infinity": -math.inf}[x]
    if isinstance(x, list):
        return [_dec(v) for v in x]
    return x


def _max_err(a, b):
    if isinstance(a, list):
        return max(_max_err(x, y) for x, y in zip(a, b, strict=True))
    return abs(a - b)


@pytest.mark.parametrize("version", spec.SUPPORTED_VERSIONS)
def test_spec_version(version):
    assert FILES[version]["spec_version"] == version


@pytest.mark.parametrize("version", spec.SUPPORTED_VERSIONS)
def test_vectors_file_is_up_to_date(version):
    from tools_gen import render  # noqa: E402  (tests/conftest.py가 tools/를 경로에 추가)

    text = (VECTORS_DIR / f"{version}-vectors.json").read_text(encoding="utf-8")
    assert text == render(version), "tools/gen_vectors.py를 다시 실행할 것"


def test_v1_keeps_v0_cases_and_scores_for_right_hand():
    """v1 파일은 v0 케이스를 모두 포함하고, 반전이 쓰이지 않은 케이스는 점수가 v0과 같다."""
    for section in ("static_score", "dtw"):
        v0 = {c["name"]: c for c in FILES["ai-io-v0"][section]}
        for c in FILES["ai-io-v1"][section]:
            if c["name"] in v0 and not c.get("invalid") and not c["expected"]["mirrored"]:
                assert c["expected"]["score"] == v0[c["name"]]["expected"]["score"], c["name"]
        assert set(v0) <= {c["name"] for c in FILES["ai-io-v1"][section]}


def test_v1_has_mirror_cases():
    """CTO 조건: 반전 우세·동률 케이스를 정적·동적 각각 1개 이상."""
    for section in ("static_score", "dtw"):
        names = {c["name"]: c["expected"] for c in FILES["ai-io-v1"][section] if c["name"].startswith("v1_")}
        assert any(e["mirrored"] for e in names.values()), section
        assert any("tie" in n and not e["mirrored"] for n, e in names.items()), section


@pytest.mark.parametrize("version,case", _cases("normalize"), ids=_ids(_cases("normalize")))
def test_normalize(version, case):
    TOL = FILES[version]["tolerances"]
    frame = _dec(case["frame"])
    if case.get("invalid"):
        with pytest.raises(ref.InvalidInput):
            ref.normalize_frame(frame)
        return
    assert _max_err(ref.normalize_frame(frame), case["expected"]) <= TOL["normalize_f64"]


@pytest.mark.parametrize("version,case", _cases("tensor"), ids=_ids(_cases("tensor")))
def test_tensor(version, case):
    TOL = FILES[version]["tolerances"]
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


@pytest.mark.parametrize("version,case", _cases("static_score"), ids=_ids(_cases("static_score")))
def test_static_score(version, case):
    TOL = FILES[version]["tolerances"]
    r, c = _dec(case["reference"]), _dec(case["candidate"])
    if case.get("invalid"):
        with pytest.raises(ref.InvalidInput):
            ref.static_score(r, c, spec_version=version)
        return
    got = ref.static_score(r, c, spec_version=version)
    if "mirrored" in case["expected"]:
        assert got.mirrored == case["expected"]["mirrored"]
    assert abs(got.distance - case["expected"]["distance"]) <= TOL["distance"]
    assert abs(got.score - case["expected"]["score"]) <= TOL["score"]


@pytest.mark.parametrize("version,case", _cases("dtw"), ids=_ids(_cases("dtw")))
def test_dtw(version, case):
    TOL = FILES[version]["tolerances"]
    r, c = _dec(case["reference"]), _dec(case["candidate"])
    if case.get("invalid"):
        with pytest.raises(ref.InvalidInput):
            ref.score_dynamic(r, c, spec_version=version)
        return
    # 기대값은 §10.2 알고리즘 수준(dtw_score). §4 유효성(N<8)은 invalid 케이스에서 score_dynamic으로 확인한다
    got = ref.dtw_score(r, c, spec_version=version)
    if "mirrored" in case["expected"]:
        assert got.mirrored == case["expected"]["mirrored"]
    assert got.path_length == case["expected"]["path_length"]
    assert abs(got.avg_cost - case["expected"]["avg_cost"]) <= TOL["avg_cost"]
    assert abs(got.score - case["expected"]["score"]) <= TOL["score"]


@pytest.mark.parametrize("version,case", _cases("best_track"), ids=_ids(_cases("best_track")))
def test_best_track(version, case):
    TOL = FILES[version]["tolerances"]
    tracks = {t["label"]: _dec(t["frames"]) for t in case["tracks"]}
    got = ref.best_track_score(_dec(case["reference"]), tracks, dynamic=case["kind"] == "dynamic", spec_version=version)
    if case["expected"] is None:
        assert got is None
    else:
        assert got[0] == case["expected"]["label"]
        assert abs(got[1] - case["expected"]["score"]) <= TOL["score"]


@pytest.mark.parametrize("version,case", _cases("recording_duration"))
def test_recording_duration(version, case):
    assert ref.recording_duration_ms(case["first_t_ms"], case["last_t_ms"]) == case["expected"]
