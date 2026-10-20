"""Keras 모델 → TFLite(Android·iOS) / TF.js(웹) 변환과 변환 일치성 검사(스펙 §7.1·§8).

- v0은 float32 변환만 한다(양자화 없음, 스펙 §7.1)
- TFLite는 기본 연산만 쓴다(Flex/SELECT_TF_OPS 없이) — 앱 용량을 늘리지 않기 위함
- TF.js 확인은 tools/tfjs_smoke(Node)로 실제 런타임에서 로드·추론한다
"""

from __future__ import annotations

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import hashlib  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Optional  # noqa: E402

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
import tensorflowjs as tfjs  # noqa: E402

from . import reference as ref  # noqa: E402
from . import spec  # noqa: E402
from .synthetic import HandPose, hand_frame, motion  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
NODE_DIR = ROOT / "tools" / "tfjs_smoke"

TOL_LOGIT = 1e-4  # 스펙 §8
TOL_SCORE = 0.1


def fixed_inputs() -> np.ndarray:
    """변환 일치성 검사용 고정 입력 4개 — **합성 데이터만**(public 레포 규칙: 실제 촬영자 데이터 금지).

    정적 2(편 손, 주먹), 동적 2(30·45프레임). 모델이 바뀌어도 입력은 같게 유지한다.
    """
    tensors = [
        ref.static_tensor(hand_frame(HandPose())),
        ref.static_tensor(hand_frame(HandPose(thumb=0.9, index=1, middle=1, ring=1, pinky=1))),
        ref.dynamic_tensor(motion([HandPose(index=1, y=0.8), HandPose(roll=0.4, y=0.6)], 30)),
        ref.dynamic_tensor(motion([HandPose(yaw=-0.6, x=0.4), HandPose(yaw=0.6, x=0.6)], 45, warp=1.3)),
    ]
    return np.asarray(tensors, dtype=np.float32)


def to_tflite(model, path: Path) -> tuple[Optional[Path], str]:
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    try:
        blob = conv.convert()
    except Exception as e:  # 변환기 지원 여부 확인이 목적이라 실패도 결과로 돌려준다
        return None, f"변환 실패: {type(e).__name__}: {str(e).splitlines()[0][:200]}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
    return path, f"ok ({len(blob):,} bytes)"


def tflite_predict(path: Path, x: np.ndarray) -> np.ndarray:
    interp = tf.lite.Interpreter(model_path=str(path))
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    interp.resize_tensor_input(inp["index"], [1, spec.SEQUENCE_LENGTH, spec.FEATURES])
    interp.allocate_tensors()
    ys = []
    for row in x:
        interp.set_tensor(inp["index"], row[None, ...].astype(np.float32))
        interp.invoke()
        ys.append(interp.get_tensor(out["index"])[0])
    return np.asarray(ys)


def to_tfjs(model, out_dir: Path) -> tuple[Optional[Path], str]:
    try:
        tfjs.converters.save_keras_model(model, str(out_dir))
    except Exception as e:
        return None, f"변환 실패: {type(e).__name__}: {str(e).splitlines()[0][:200]}"
    return out_dir, "ok"


def tfjs_predict(model_dir: Path, x: np.ndarray, expected: np.ndarray) -> tuple[Optional[np.ndarray], str]:
    """Node TF.js 런타임으로 추론. tools/tfjs_smoke에서 npm ci가 안 돼 있으면 건너뛴다."""
    model_dir = model_dir.resolve()  # Node는 tools/tfjs_smoke에서 실행되므로 절대 경로로 넘긴다
    fixture = model_dir.parent / f"{model_dir.name}-fixture.json"
    fixture.write_text(json.dumps({"inputs": x.tolist(), "keras_logits": expected.tolist()}), encoding="utf-8")
    if not (NODE_DIR / "node_modules").exists():
        return None, "Node 확인 건너뜀: tools/tfjs_smoke에서 npm ci 필요"
    proc = subprocess.run(["node", "check.mjs", str(model_dir), str(fixture)], cwd=NODE_DIR, capture_output=True, text=True)
    if proc.returncode != 0:
        return None, f"TF.js 로드/추론 실패: {proc.stderr.strip()[:300]}"
    res = json.loads(proc.stdout)
    return np.asarray(res["logits"]), f"ok (load {res['load_ms']:.0f}ms, predict {res['predict_ms']:.1f}ms/샘플, Node CPU)"


def _softmax(z: np.ndarray, tau: float) -> np.ndarray:
    z = z / tau
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def compare(got: np.ndarray, keras_logits: np.ndarray, *, tau: float = 1.0, targets: Optional[np.ndarray] = None) -> dict:
    """스펙 §8 변환 일치성: 로짓 ≤ 1e-4, 점수 ≤ 0.1, 80점 통과 판정 일치."""
    if targets is None:
        targets = keras_logits.argmax(axis=-1)
    idx = np.arange(len(targets))
    s_ref = 100 * _softmax(keras_logits, tau)[idx, targets]
    s_got = 100 * _softmax(got, tau)[idx, targets]
    logit_err = float(np.abs(got - keras_logits).max())
    score_err = float(np.abs(s_got - s_ref).max())
    same = bool(np.array_equal(s_ref >= spec.PASS_THRESHOLD, s_got >= spec.PASS_THRESHOLD))
    return {
        "max_logit_err": logit_err,
        "max_score_err": score_err,
        "verdict_same": same,
        "pass": logit_err <= TOL_LOGIT and score_err <= TOL_SCORE and same,
    }


def artifact_files(d: Path) -> list[dict]:
    """매니페스트 `artifacts.*.files` 형식(스펙 §7.1)."""
    return [
        {"path": p.relative_to(d).as_posix(), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in sorted(d.rglob("*"))
        if p.is_file()
    ]
