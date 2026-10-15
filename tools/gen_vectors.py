"""공통 테스트 벡터 생성: vectors/ai-io-v0-vectors.json

    uv run python tools/gen_vectors.py          # 파일 갱신
    uv run python tools/gen_vectors.py --check  # 커밋된 파일이 코드와 같은지만 확인(CI용)

입력은 합성 데이터만 쓴다(sueori_ai.synthetic). 기대값은 sueori_ai.reference(정본)가 계산한다.
JSON 실수는 Python repr(가장 짧게 왕복되는 표현)로 써서 Float64 값이 그대로 복원된다.
NaN·Infinity는 JSON 숫자로 쓸 수 없으므로 문자열 "NaN", "Infinity", "-Infinity"로 쓴다.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

from sueori_ai import reference as ref
from sueori_ai import spec
from sueori_ai.synthetic import HandPose, hand_frame, jitter, motion, round_frame

OUT = Path(__file__).resolve().parent.parent / "vectors" / f"{spec.SPEC_VERSION}-vectors.json"
SEED = 20261015


def _enc(x):
    if isinstance(x, float) and not math.isfinite(x):
        return "NaN" if math.isnan(x) else ("Infinity" if x > 0 else "-Infinity")
    if isinstance(x, list):
        return [_enc(v) for v in x]
    if isinstance(x, dict):
        return {k: _enc(v) for k, v in x.items()}
    return x


def _r(frames):
    return [round_frame(f) for f in frames]


def build() -> dict:
    rng = random.Random(SEED)

    open_hand = HandPose()
    fist = HandPose(thumb=0.9, index=1.0, middle=1.0, ring=1.0, pinky=1.0)
    v_sign = HandPose(thumb=0.8, ring=1.0, pinky=1.0)  # 검지·중지만 편 손
    turned = HandPose(yaw=1.1, roll=0.3, scale=1.3, x=0.42, y=0.7)

    # 동적 동작 3개: A(주먹 → 편 손, 위로 이동), B(A와 같은 동작, 다른 사람·속도), C(다른 동작)
    a_keys = [
        HandPose(thumb=0.9, index=1, middle=1, ring=1, pinky=1, y=0.8),
        HandPose(thumb=0.3, index=0.3, middle=0.3, ring=0.3, pinky=0.3, roll=0.2, y=0.7),
        HandPose(roll=0.4, y=0.6),
    ]
    a_ref = _r(motion(a_keys, 30))
    a_slow = _r([jitter(f, rng, 0.002) for f in motion(a_keys, 45, warp=1.4)])
    c_keys = [
        HandPose(yaw=-0.6, index=0, middle=1, ring=1, pinky=1, x=0.4),
        HandPose(yaw=0.6, index=0, middle=1, ring=1, pinky=1, x=0.6),
        HandPose(yaw=-0.6, index=0, middle=1, ring=1, pinky=1, x=0.4),
    ]
    c_seq = _r(motion(c_keys, 36))
    still = _r([jitter(hand_frame(a_keys[0]), rng, 0.0015) for _ in range(30)])

    f_open = round_frame(hand_frame(open_hand))
    f_open_noisy = round_frame(jitter(f_open, rng, 0.003))
    f_fist = round_frame(hand_frame(fist))
    f_v = round_frame(hand_frame(v_sign))
    f_turned = round_frame(hand_frame(turned))
    f_degenerate = [[0.5, 0.5, 0.0] for _ in range(21)]  # 스케일 0 → 1.0 대체
    f_nan = [list(p) for p in f_open]
    f_nan[8][1] = math.nan
    f_inf = [list(p) for p in f_open]
    f_inf[0][2] = math.inf

    # ── §2 정규화 ──
    normalize_cases = []
    for name, frame in [
        ("open_hand", f_open),
        ("fist", f_fist),
        ("turned_scaled_moved", f_turned),
        ("degenerate_scale_zero", f_degenerate),
    ]:
        normalize_cases.append({"name": name, "frame": frame, "expected": ref.normalize_frame(frame)})
    for name, frame in [("nan_coordinate", f_nan), ("infinite_coordinate", f_inf)]:
        normalize_cases.append({"name": name, "frame": frame, "invalid": True})

    # ── §4 텐서 ──
    tensor_cases = [
        {"name": "static_v_sign", "kind": "static", "frames": [f_v], "expected": ref.static_tensor(f_v)},
    ]
    for name, frames in [
        ("dynamic_n8_min", a_ref[:8]),
        ("dynamic_n30", a_ref),
        ("dynamic_n45", a_slow),
        ("dynamic_n32_identity", c_seq[:32]),
    ]:
        tensor_cases.append(
            {"name": name, "kind": "dynamic", "frames": frames, "expected": ref.dynamic_tensor(frames)}
        )
    tensor_cases.append({"name": "dynamic_n7_too_short", "kind": "dynamic", "frames": a_ref[:7], "invalid": True})

    # ── §10.1 정적 거리 점수 ──
    static_cases = []
    for name, r, c in [
        ("same_pose_noise", f_open, f_open_noisy),
        ("same_pose_turned_moved", f_open, round_frame(hand_frame(HandPose(scale=0.7, x=0.3, y=0.5)))),
        ("open_vs_fist", f_open, f_fist),
        ("v_sign_vs_open", f_v, f_open),
        ("identical", f_v, f_v),
    ]:
        res = ref.static_score(r, c)
        static_cases.append(
            {"name": name, "reference": r, "candidate": c, "expected": {"score": res.score, "distance": res.distance}}
        )
    static_cases.append({"name": "nan_candidate", "reference": f_open, "candidate": f_nan, "invalid": True})

    # ── §10.2 DTW ──
    dtw_cases = []
    for name, r, c in [
        ("same_motion_slower_noisy", a_ref, a_slow),
        ("identical", a_ref, a_ref),
        ("different_motion", a_ref, c_seq),
        ("still_pose", a_ref, still),
        # 동률 규칙 검증: 스펙 순서(diag→up→left)로만 L=4, 다른 5가지 순서는 전부 L=5
        ("ties_all_other_orders_differ", [f_open, f_v, f_open, f_v], [f_open, f_fist, f_v]),
        # 같은 프레임 반복(비용 0 동률): diag 우선이면 L=2, up/left 우선이면 L=3
        ("ties_repeated_identical", [f_open, f_open], [f_open, f_open]),
        ("single_vs_many", [f_v], [f_v, f_v, f_open, f_v]),
    ]:
        res = ref.dtw_score(r, c)
        dtw_cases.append(
            {
                "name": name,
                "reference": r,
                "candidate": c,
                "expected": {"score": res.score, "avg_cost": res.avg_cost, "path_length": res.path_length},
            }
        )
    dtw_cases.append(
        {"name": "candidate_too_short_n7", "reference": a_ref, "candidate": a_slow[:7], "invalid": True}
    )

    # ── §3 트랙별 최고점 ──
    best_cases = []
    tracks = [("Left", c_seq), ("Right", a_slow), ("", still)]
    best = ref.best_track_score(a_ref, dict(tracks), dynamic=True)
    best_cases.append(
        {
            "name": "dynamic_three_tracks",
            "kind": "dynamic",
            "reference": a_ref,
            "tracks": [{"label": k, "frames": v} for k, v in tracks],
            "expected": {"label": best[0], "score": best[1]},
        }
    )
    tracks = [("Left", [f_nan]), ("Right", [f_open_noisy])]
    best = ref.best_track_score([f_open], dict(tracks), dynamic=False)
    best_cases.append(
        {
            "name": "static_skips_nan_track",
            "kind": "static",
            "reference": [f_open],
            "tracks": [{"label": k, "frames": v} for k, v in tracks],
            "expected": {"label": best[0], "score": best[1]},
        }
    )
    tracks = [("Left", a_slow[:5]), ("Right", [f_nan] * 10)]
    best_cases.append(
        {
            "name": "dynamic_no_valid_track",
            "kind": "dynamic",
            "reference": a_ref,
            "tracks": [{"label": k, "frames": v} for k, v in tracks],
            "expected": None,
        }
    )

    # ── §10.3 녹화 길이 ──
    duration_cases = [
        {"first_t_ms": f, "last_t_ms": l, "expected": ref.recording_duration_ms(f, l)}
        for f, l in [
            (None, None),
            (0, None),
            (500, 500),
            (900, 100),
            (0, 1000),  # 1200 → 하한 2000
            (0, 2500),  # 3000
            (33, 1867),  # 2200.8 → 2201
            (0, 1875.4),  # 2250.48 → 2250
            (0, 9000),  # 10800 → 상한 8000
        ]
    ]

    return {
        "spec_version": spec.SPEC_VERSION,
        "generator": {"script": "tools/gen_vectors.py", "seed": SEED},
        "data_source": "synthetic (sueori_ai.synthetic) — 실제 촬영자 데이터·국립국어원 데이터 미포함",
        "encoding": {
            "frame": "21개 점 [x, y, z] (MediaPipe landmarks 이미지 좌표)",
            "vector": "정규화 63차원 [x0, y0, z0, …, x20, y20, z20]",
            "nonfinite": "NaN·Infinity는 문자열 \"NaN\" / \"Infinity\" / \"-Infinity\"",
            "invalid": "invalid: true인 케이스는 채점·정규화가 거부(예외 또는 채점 불가 반환)되어야 함",
            "dtw": "dtw 기대값은 §10.2 알고리즘 수준(유효성 N<8 미적용, 동률 검증용 짧은 시퀀스 포함). invalid 케이스만 §4 유효성까지 포함한 동적 채점 진입점으로 확인",
        },
        "tolerances": {
            "normalize_f64": 1e-6,
            "tensor_f64": 1e-6,
            "tensor_f32": 1e-5,
            "distance": 1e-12,
            "avg_cost": 1e-12,
            "score": 1e-9,
            "path_length": 0,
            "recording_duration_ms": 0,
        },
        "normalize": normalize_cases,
        "tensor": tensor_cases,
        "static_score": static_cases,
        "dtw": dtw_cases,
        "best_track": best_cases,
        "recording_duration": duration_cases,
    }


def render() -> str:
    return json.dumps(_enc(build()), ensure_ascii=False, separators=(",", ":")) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="파일을 쓰지 않고 커밋된 파일과 같은지만 확인")
    args = ap.parse_args()
    text = render()
    if args.check:
        if not OUT.exists() or OUT.read_text(encoding="utf-8") != text:
            print(f"✗ {OUT.name}가 코드와 다름 — tools/gen_vectors.py를 다시 실행할 것", file=sys.stderr)
            return 1
        print(f"✓ {OUT.name} 최신")
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT} ({len(text):,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
