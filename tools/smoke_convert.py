"""변환 스모크: 더미 모델을 TF.js(웹)와 TFLite(Android·iOS)로 변환하고 Keras 원본과 출력이 같은지 확인한다.

    uv run python tools/smoke_convert.py                 # cnn, gru, gru_compat 전부
    uv run python tools/smoke_convert.py --arch cnn
    (TF.js 확인은 Node가 필요: cd tools/tfjs_smoke && npm ci 를 한 번 해 둘 것)

목적(CTO 2026-10-15): 고정한 툴체인으로 ① 두 변환기가 모두 동작하는지 ② 쓰려는 레이어가 양쪽에서
지원되는지 ③ 변환 일치성(스펙 §8: 로짓 ≤ 1e-4, 점수 ≤ 0.1점, 80점 통과 판정 일치)을 확인한다.
가중치는 무작위 초기화이고, 산출물은 build/smoke/ 아래에만 쓴다(.gitignore — 커밋 금지).
입력은 합성 손 데이터로 만든 [1, 32, 63] 텐서다. 변환·비교 코드는 sueori_ai.convert와 같다.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from sueori_ai import convert
from sueori_ai import spec

import tensorflow as tf  # noqa: E402  (convert가 TF_USE_LEGACY_KERAS를 먼저 설정)
import tensorflowjs as tfjs  # noqa: E402
import tf_keras as keras  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "build" / "smoke"
NUM_CLASSES = 48  # 지문자 40 + 가족 7 + other (스펙 §5)


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


def _report(name: str, r: dict) -> None:
    mark = "✓" if r["pass"] else "✗"
    print(
        f"  {mark} {name:6s} logit Δ {r['max_logit_err']:.2e} (≤{convert.TOL_LOGIT:g}) · "
        f"score Δ {r['max_score_err']:.2e} (≤{convert.TOL_SCORE:g}) · 통과 판정 일치 {r['verdict_same']}"
    )


def smoke(arch: str) -> bool:
    keras.utils.set_random_seed(7)
    model = build_model(arch)
    x = convert.fixed_inputs()
    keras_logits = model.predict(x, verbose=0)
    d = OUT / arch
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)

    print(f"\n=== {arch}: {model.count_params():,} params, layers={[l.__class__.__name__ for l in model.layers]}")
    passed = []

    t0 = time.perf_counter()
    path, msg = convert.to_tflite(model, d / "android" / "model.tflite")
    print(f"  TFLite  : {msg} [{(time.perf_counter() - t0):.1f}s]")
    if path is not None:
        r = convert.compare(convert.tflite_predict(path, x), keras_logits)
        _report("tflite", r)
        passed.append(r["pass"])

    t0 = time.perf_counter()
    web, msg = convert.to_tfjs(model, d / "web")
    if web is not None:
        got, msg = convert.tfjs_predict(web, x, keras_logits)
        print(f"  TF.js   : {msg} [{(time.perf_counter() - t0):.1f}s]")
        if got is not None:
            print(f"  web files: {json.dumps(convert.artifact_files(web), ensure_ascii=False)}")
            r = convert.compare(got, keras_logits)
            _report("tfjs", r)
            passed.append(r["pass"])
    else:
        print(f"  TF.js   : {msg}")

    ok = len(passed) == 2 and all(passed)
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
