"""numpy 빠른 경로가 기준 구현·공통 테스트 벡터와 비트 단위로 같은지 확인한다."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from sueori_ai import fast
from sueori_ai import reference as ref
from sueori_ai import spec

_DIR = Path(__file__).resolve().parent.parent / "vectors"
FILES = {v: json.loads((_DIR / f"{v}-vectors.json").read_text()) for v in spec.SUPPORTED_VERSIONS}
DATA = FILES["ai-io-v0"]  # 정규화·텐서는 버전과 무관(v0 = v1)


def _cases(section, valid=True):
    return [(v, c) for v, d in FILES.items() for c in d[section] if bool(c.get("invalid")) != valid]


def _dec(x):
    if isinstance(x, str):
        return {"NaN": math.nan, "Infinity": math.inf, "-Infinity": -math.inf}[x]
    if isinstance(x, list):
        return [_dec(v) for v in x]
    return x


@pytest.mark.parametrize("case", [c for c in DATA["normalize"] if not c.get("invalid")], ids=lambda c: c["name"])
def test_normalize_bit_exact(case):
    assert fast.normalize_frames(case["frame"])[0].tolist() == case["expected"]


@pytest.mark.parametrize("case", [c for c in DATA["normalize"] if c.get("invalid")], ids=lambda c: c["name"])
def test_normalize_rejects_nonfinite(case):
    with pytest.raises(ref.InvalidInput):
        fast.normalize_frames(_dec(case["frame"]))


@pytest.mark.parametrize("case", [c for c in DATA["tensor"] if not c.get("invalid")], ids=lambda c: c["name"])
def test_tensor_bit_exact(case):
    got = fast.static_tensor(case["frames"][0]) if case["kind"] == "static" else fast.dynamic_tensor(case["frames"])
    assert got.tolist() == case["expected"]


@pytest.mark.parametrize("version,case", _cases("static_score"), ids=lambda x: x if isinstance(x, str) else x["name"])
def test_static_score_bit_exact(version, case):
    assert fast.static_score(case["reference"], case["candidate"], spec_version=version) == case["expected"]["score"]


@pytest.mark.parametrize("version,case", _cases("dtw"), ids=lambda x: x if isinstance(x, str) else x["name"])
def test_dtw_bit_exact(version, case):
    got = fast.dtw_score(case["reference"], case["candidate"], spec_version=version)
    exp = case["expected"]
    assert (got.score, got.avg_cost, got.path_length) == (exp["score"], exp["avg_cost"], exp["path_length"])
    assert got.mirrored == exp.get("mirrored", False)


def test_mirror_matches_reference_bitwise():
    rng = np.random.default_rng(3)
    v = rng.normal(size=(5, 63))
    assert fast.mirror_vectors(v).tolist() == [ref.mirror_vector(row) for row in v.tolist()]


def test_flat_x_tie_is_exact():
    """동률 케이스는 원본과 반전의 거리가 비트 단위로 같아야 한다(구현이 동률 규칙을 확인할 수 있게)."""
    case = next(c for c in FILES["ai-io-v1"]["static_score"] if c["name"] == "v1_mirror_tie_flat_x")
    r, c = fast.normalize_frames(case["reference"])[0], fast.normalize_frames(case["candidate"])[0]
    assert fast.landmark_distance(r, c) == fast.landmark_distance(r, fast.mirror_vectors(c))


def test_resample_matches_reference_on_random_lengths():
    rng = np.random.default_rng(0)
    for n in (1, 2, 7, 31, 33, 90):
        v = rng.normal(size=(n, 63))
        assert fast.resample(v).tolist() == ref.resample(v.tolist())


def test_dtw_empty():
    r = fast.dtw_score([], DATA["dtw"][0]["candidate"])
    assert r.score == 0.0 and r.path_length == 0
