"""변환 스모크: 더미 모델을 TF.js(웹)와 TFLite(Android·iOS)로 변환하고 Keras 원본과 출력이 같은지 확인한다.

    uv run python tools/smoke_convert.py                 # cnn, gru, gru_compat 전부
    uv run python tools/smoke_convert.py --arch cnn
    (TF.js 확인은 Node가 필요: cd tools/tfjs_smoke && npm ci 를 한 번 해 둘 것)

목적(CTO 2026-10-15): 고정한 툴체인으로 ① 두 변환기가 모두 동작하는지 ② 쓰려는 레이어가 양쪽에서
지원되는지 ③ 변환 일치성(스펙 §8: 로짓 ≤ 1e-4, 점수 ≤ 0.1점, 80점 통과 판정 일치)을 확인한다.
가중치는 무작위 초기화이고, 산출물은 build/smoke/ 아래에만 쓴다(.gitignore — 커밋 금지).
입력은 합성 손 데이터로 만든 [1, 32, 63] 텐서다.
"""

from __future__ import annotations

import os

# tf.keras를 Keras 2(tf_keras)로 고정한다. tensorflowjs 변환기와 Keras 3의 호환 문제를 피하기 위함.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
import tensorflowjs as tfjs  # noqa: E402
import tf_keras as keras  # noqa: E402

from sueori_ai import reference as ref  # noqa: E402
from sueori_ai import spec  # noqa: E402
from sueori_ai.synthetic import HandPose, hand_frame, motion  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "build" / "smoke"
NODE_DIR = ROOT / "tools" / "tfjs_smoke"
NUM_CLASSES = 48  # 지문자 40 + 가족 7 + other (스펙 §5)

TOL_LOGIT = 1e-4
TOL_SCORE = 0.1


def build_model(arch: str) -> keras.Model:
    inp = keras.Input(shape=(spec.SEQUENCE_LENGTH, spec.FEATURES), name="landmarks")
    if arch == "cnn":
        x = keras.layers.Conv1D(64, 5, padding="same", activation="relu")(inp)
        x = keras.layers.Conv1D(64, 5, padding="same", activation="relu")(x)
        x = keras.layers.GlobalAveragePooling1D()(x)
    elif arch == "gru":
        x = keras.layers.GRU(64)(inp)  # Keras 기본값(reset_after=True) 그대로
    elif arch == "gru_compat":
        # 변환 호환 설정: TF.js는 reset_after=True를 지원하지 않고, TFLite 기본 연산(Flex 없이)은
        # 루프를 펼쳐야(unroll) 변환된다. T=32로 고정이라 unroll 비용은 감당 가능한 수준.
        x = keras.layers.GRU(64, reset_after=False, unroll=True)(inp)
    else:
        raise ValueError(arch)
    x = keras.layers.Dense(64, activation="relu")(x)
    out = keras.layers.Dense(NUM_CLASSES, name="logits")(x)
    return keras.Model(inp, out, name=f"smoke_{arch}")


def fixed_inputs() -> np.ndarray:
    """합성 데이터로 만든 고정 입력 4개: 정적 2(편 손, 주먹), 동적 2(30·45프레임)."""
    tensors = [
        ref.static_tensor(hand_frame(HandPose())),
        ref.static_tensor(hand_frame(HandPose(thumb=0.9, index=1, middle=1, ring=1, pinky=1))),
        ref.dynamic_tensor(motion([HandPose(index=1, y=0.8), HandPose(roll=0.4, y=0.6)], 30)),
        ref.dynamic_tensor(motion([HandPose(yaw=-0.6, x=0.4), HandPose(yaw=0.6, x=0.6)], 45, warp=1.3)),
    ]
    return np.asarray(tensors, dtype=np.float32)


def scores(logits: np.ndarray) -> np.ndarray:
    """§6 점수 = 100·softmax(logits)[target]. 더미 모델이므로 τ=1, target = Keras 원본의 argmax."""
    z = logits - logits.max(axis=-1, keepdims=True)
    p = np.exp(z) / np.exp(z).sum(axis=-1, keepdims=True)
    return 100.0 * p


def run_tflite(model: keras.Model, path: Path, x: np.ndarray) -> tuple[np.ndarray | None, str]:
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]  # Flex(SELECT_TF_OPS) 없이 되는지 확인
    try:
        blob = conv.convert()
    except Exception as e:  # 변환기 지원 여부 확인이 목적이므로 실패도 결과로 보고한다
        return None, f"변환 실패: {type(e).__name__}: {str(e).splitlines()[0][:200]}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
    interp = tf.lite.Interpreter(model_path=str(path))
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    interp.resize_tensor_input(inp["index"], [1, spec.SEQUENCE_LENGTH, spec.FEATURES])
    interp.allocate_tensors()
    ys = []
    for row in x:
        interp.set_tensor(inp["index"], row[None, ...])
        interp.invoke()
        ys.append(interp.get_tensor(out["index"])[0])
    return np.asarray(ys), f"ok ({len(blob):,} bytes)"


def run_tfjs(model: keras.Model, out_dir: Path, x: np.ndarray, expected: np.ndarray) -> tuple[np.ndarray | None, str]:
    try:
        tfjs.converters.save_keras_model(model, str(out_dir))
    except Exception as e:
        return None, f"변환 실패: {type(e).__name__}: {str(e).splitlines()[0][:200]}"
    (out_dir.parent / "fixture.json").write_text(
        json.dumps({"inputs": x.tolist(), "keras_logits": expected.tolist()}), encoding="utf-8"
    )
    if not (NODE_DIR / "node_modules").exists():
        return None, "Node 확인 건너뜀: tools/tfjs_smoke에서 npm ci 필요"
    proc = subprocess.run(
        ["node", "check.mjs", str(out_dir), str(out_dir.parent / "fixture.json")],
        cwd=NODE_DIR,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return None, f"TF.js 로드/추론 실패: {proc.stderr.strip()[:300]}"
    res = json.loads(proc.stdout)
    return np.asarray(res["logits"]), f"ok (load {res['load_ms']:.0f}ms, predict {res['predict_ms']:.1f}ms/샘플, Node CPU)"


def artifact_files(d: Path) -> list[dict]:
    """매니페스트 `artifacts.*.files` 형식(스펙 §7.1)."""
    return [
        {"path": p.relative_to(d).as_posix(), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in sorted(d.rglob("*"))
        if p.is_file()
    ]


def compare(name: str, got: np.ndarray, keras_logits: np.ndarray) -> dict:
    target = keras_logits.argmax(axis=-1)
    s_ref = scores(keras_logits)[np.arange(len(target)), target]
    s_got = scores(got)[np.arange(len(target)), target]
    logit_err = float(np.abs(got - keras_logits).max())
    score_err = float(np.abs(s_got - s_ref).max())
    verdict_same = bool(np.array_equal(s_ref >= spec.PASS_THRESHOLD, s_got >= spec.PASS_THRESHOLD))
    ok = logit_err <= TOL_LOGIT and score_err <= TOL_SCORE and verdict_same
    return {"target": name, "max_logit_err": logit_err, "max_score_err": score_err, "verdict_same": verdict_same, "pass": ok}


def smoke(arch: str) -> bool:
    keras.utils.set_random_seed(7)
    model = build_model(arch)
    x = fixed_inputs()
    keras_logits = model.predict(x, verbose=0)
    d = OUT / arch
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)

    print(f"\n=== {arch}: {model.count_params():,} params, layers={[l.__class__.__name__ for l in model.layers]}")
    results = []

    t0 = time.perf_counter()
    tfl, msg = run_tflite(model, d / "android" / "model.tflite", x)
    print(f"  TFLite  : {msg} [{(time.perf_counter() - t0):.1f}s]")
    if tfl is not None:
        results.append(compare("tflite", tfl, keras_logits))

    t0 = time.perf_counter()
    web, msg = run_tfjs(model, d / "web", x, keras_logits)
    print(f"  TF.js   : {msg} [{(time.perf_counter() - t0):.1f}s]")
    if web is not None:
        results.append(compare("tfjs", web, keras_logits))
        print(f"  web files: {json.dumps(artifact_files(d / 'web'), ensure_ascii=False)}")

    for r in results:
        mark = "✓" if r["pass"] else "✗"
        print(
            f"  {mark} {r['target']:6s} logit Δ {r['max_logit_err']:.2e} (≤{TOL_LOGIT:g}) · "
            f"score Δ {r['max_score_err']:.2e} (≤{TOL_SCORE:g}) · 통과 판정 일치 {r['verdict_same']}"
        )
    ok = len(results) == 2 and all(r["pass"] for r in results)
    print(f"  → {arch}: {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["cnn", "gru", "gru_compat"], action="append")
    args = ap.parse_args()
    archs = args.arch or ["cnn", "gru", "gru_compat"]
    print(f"tensorflow {tf.__version__} · tf_keras {keras.__version__} · tensorflowjs {tfjs.__version__}")
    results = {a: smoke(a) for a in archs}
    print("\n요약:", ", ".join(f"{a}={'PASS' if ok else 'FAIL'}" for a, ok in results.items()))
    # cnn은 v0 기본 구조라 반드시 통과해야 한다. gru 계열은 후보 레이어 지원 여부 확인용(실패해도 보고만).
    return 0 if results.get("cnn", True) else 1


if __name__ == "__main__":
    sys.exit(main())
