"""변환 일치성 비교·벡터(스펙 §8)."""

import numpy as np
import tensorflow as tf

from sueori_ai import convert, spec
from sueori_ai.model import build_cnn


def test_tflite_input_is_fixed_batch_1(tmp_path):
    # 동적 배치(-1)면 LiteRT CompiledModel.createInputBuffers()가 실패한다
    model = build_cnn(5)
    path, msg = convert.to_tflite(model, tmp_path / "model.tflite")
    assert path is not None, msg
    inp = tf.lite.Interpreter(model_path=str(path)).get_input_details()[0]
    expected = [1, spec.SEQUENCE_LENGTH, spec.FEATURES]
    assert inp["shape"].tolist() == expected and inp["shape_signature"].tolist() == expected
    x = convert.fixed_inputs()
    r = convert.compare(convert.tflite_predict(path, x), model.predict(x, verbose=0))
    assert r["pass"], r


def test_compare_passes_identical_logits():
    z = np.random.default_rng(0).normal(size=(4, 48)).astype(np.float32)
    r = convert.compare(z.copy(), z, tau=1.3)
    assert r["pass"] and r["max_logit_err"] == 0 and r["verdict_same"]


def test_compare_checks_scores_of_non_argmax_classes():
    # 클래스 0이 argmax, 클래스 1은 부분 점수 구간(약 50점). 클래스 1의 점수만 0.1점 넘게 어긋난 경우
    ref = np.log(np.array([[0.5, 0.45, 0.05]]))
    got = np.log(np.array([[0.5, 0.448, 0.052]]))
    r = convert.compare(got, ref, tau=1.0)
    assert r["verdict_same"] and r["max_score_err"] > convert.TOL_SCORE and not r["pass"]


def test_compare_catches_verdict_flip_at_threshold():
    ref = np.log(np.array([[0.7999999, 0.1, 0.1000001]]))
    got = np.log(np.array([[0.8000001, 0.1, 0.0999999]]))
    r = convert.compare(got, ref)
    assert r["max_logit_err"] <= convert.TOL_LOGIT and r["max_score_err"] <= convert.TOL_SCORE
    assert not r["verdict_same"] and not r["pass"]


def test_parity_vectors_are_self_consistent():
    x = convert.fixed_inputs()
    z = np.random.default_rng(1).normal(scale=3, size=(len(x), 5))
    v = convert.parity_vectors(x, z, tau=0.8, model_version="v0.0.0-test")
    s = np.asarray(v["expected_scores"])
    assert v["spec_version"] == spec.SPEC_VERSION and v["model_version"] == "v0.0.0-test"
    assert np.asarray(v["inputs"]).shape == (len(x), spec.SEQUENCE_LENGTH, spec.FEATURES)
    assert s.shape == np.asarray(v["keras_logits"]).shape
    np.testing.assert_allclose(s.sum(axis=1), 100, atol=1e-9)
    np.testing.assert_allclose(s, convert.scores(z, 0.8))
    assert v["expected_pass"] == (s >= spec.PASS_THRESHOLD).tolist()
