"""numpy 빠른 경로가 기준 구현·공통 테스트 벡터와 비트 단위로 같은지 확인한다."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from sueori_ai import fast
from sueori_ai import reference as ref
from sueori_ai import spec

DATA = json.loads((Path(__file__).resolve().parent.parent / "vectors" / f"{spec.SPEC_VERSION}-vectors.json").read_text())


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


@pytest.mark.parametrize("case", [c for c in DATA["static_score"] if not c.get("invalid")], ids=lambda c: c["name"])
def test_static_score_bit_exact(case):
    assert fast.static_score(case["reference"], case["candidate"]) == case["expected"]["score"]


@pytest.mark.parametrize("case", [c for c in DATA["dtw"] if not c.get("invalid")], ids=lambda c: c["name"])
def test_dtw_bit_exact(case):
    got = fast.dtw_score(case["reference"], case["candidate"])
    exp = case["expected"]
    assert (got.score, got.avg_cost, got.path_length) == (exp["score"], exp["avg_cost"], exp["path_length"])


def test_resample_matches_reference_on_random_lengths():
    rng = np.random.default_rng(0)
    for n in (1, 2, 7, 31, 33, 90):
        v = rng.normal(size=(n, 63))
        assert fast.resample(v).tolist() == ref.resample(v.tolist())


def test_dtw_empty():
    r = fast.dtw_score([], DATA["dtw"][0]["candidate"])
    assert r.score == 0.0 and r.path_length == 0
