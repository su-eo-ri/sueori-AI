"""학습 파이프라인: 클립 → 촬영자 단위 교차 검증(모델 vs DTW) → 최종 학습 → 변환·일치성 → 리포트.

    # 합성 데이터로 끝까지(수집 게이트 전에도 가능)
    uv run python tools/train_pipeline.py --synthetic
    uv run python tools/train_pipeline.py --synthetic --participants 3 --epochs 5 --max-folds 1   # 빠른 확인

    # 실제 데이터(수집 DB 스냅샷 JSONL + 기준 시퀀스) — ⚠️ 이 레포 밖 경로를 쓸 것
    uv run python tools/train_pipeline.py --data /path/to/snapshot --references /path/to/references.json

산출물은 모두 --out(기본 build/train/<run>) 아래에만 쓴다(.gitignore). 모델·변환 파일·실제 데이터는 커밋하지 않는다.
⚠️ 합성 데이터의 성능 수치는 의미가 없다(파이프라인 동작 확인용).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from sueori_ai import convert  # noqa: F401  (TF_USE_LEGACY_KERAS를 먼저 설정)
from sueori_ai import evaluate as ev
from sueori_ai import spec, synthetic_clips
from sueori_ai.clips import from_record, load_clips, write_clips
from sueori_ai.dataset import LabelMap, build_examples, load_references, participant_folds, save_references
from sueori_ai.model import TrainConfig, calibrate_temperature, train

ROOT = Path(__file__).resolve().parent.parent


def _fmt(v, pct=False):
    if v is None:
        return "—"
    return f"{v * 100:.1f}%" if pct else f"{v:.1f}"


def _table(model: dict, dtw: dict) -> str:
    rows = ["| 범주 | n | 모델 통과율 | DTW 통과율 | 모델 중앙값 | DTW 중앙값 | 모델 최대 | DTW 최대 | §6 목표 |", "|---|---|---|---|---|---|---|---|---|"]
    goals = {
        "correct": "통과율 ≥95%, 중앙값 85~95",
        "other_word": "통과율 ≤1%, 최대 ≤60",
        "still": "통과율 ≤1%, 최대 ≤70",
        "unrelated": "통과율 ≤1%, 최대 ≤70",
        "near_miss": "중앙값 40~75",
    }
    for c in ev.CATEGORIES:
        m, d = model.get(c), dtw.get(c)
        if not m:
            continue
        rows.append(
            f"| {c} | {m['n']} | {_fmt(m['pass_rate'], True)} | {_fmt(d['pass_rate'], True)} | {_fmt(m['median'])} | "
            f"{_fmt(d['median'])} | {_fmt(m['max'])} | {_fmt(d['max'])} | {goals[c]} |"
        )
    return "\n".join(rows)


def cross_validate(clips, label_map, references, word_types, cfg: TrainConfig, max_folds: int | None):
    """촬영자 단위 교차 검증. 평가 촬영자의 로짓(out-of-fold)도 모아 최종 τ 보정에 쓴다(CTO 2026-10-16)."""
    all_pairs, m_vals, d_vals, folds, oof_logits, oof_y = [], [], [], [], [], []
    participants = [c.participant_id for c in clips]
    for k, (train_ps, held) in enumerate(participant_folds(participants)):
        if max_folds is not None and k >= max_folds:
            break
        t0 = time.perf_counter()
        tr = [c for c in clips if c.participant_id in train_ps]
        te = [c for c in clips if c.participant_id == held]
        model, tau, info = train(build_examples(tr, label_map), len(label_map.classes), cfg)
        pairs = ev.make_pairs(te, label_map)

        def predict(x):
            return model.predict(x, verbose=0)

        m = ev.score_pairs_model(pairs, predict, label_map, word_types, tau)
        d = ev.score_pairs_dtw(pairs, references, word_types)
        top1 = ev.top1_accuracy(pairs, predict, label_map, word_types)
        try:
            te_ex = build_examples(te, label_map)
            oof_logits.append(predict(te_ex.x))
            oof_y.append(te_ex.y)
        except ValueError:
            pass
        folds.append({"held_out": held, "tau": tau, "top1": top1, **info, "seconds": round(time.perf_counter() - t0, 1)})
        print(f"  fold {k + 1}: 평가 {held}, τ={tau:.2f}, top1={_fmt(top1, True)}, epochs={info['epochs_run']}")
        all_pairs += pairs
        m_vals.append(m)
        d_vals.append(d)
    oof = (np.concatenate(oof_logits), np.concatenate(oof_y)) if oof_logits else None
    return all_pairs, np.concatenate(m_vals), np.concatenate(d_vals), folds, oof


def main() -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--synthetic", action="store_true", help="합성 데이터를 만들어 사용")
    src.add_argument("--data", type=Path, help="클립 JSONL 파일 또는 디렉토리(수집 DB 스냅샷)")
    ap.add_argument("--references", type=Path, help="기준 시퀀스 JSON {word_key: frames} (--data일 때 필수)")
    ap.add_argument("--participants", type=int, default=5, help="(합성) 촬영자 수")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--max-folds", type=int, default=None, help="교차 검증 fold 수 제한(빠른 확인용)")
    ap.add_argument("--model-version", default="v0.0.0-dev")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--release", action="store_true",
                    help="배포 후보 모드: TFLite·TF.js 일치성이 둘 다 PASS가 아니면 실패로 끝내고 매니페스트를 만들지 않음")
    args = ap.parse_args()

    run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.out or ROOT / "build" / "train" / run
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    if args.synthetic:
        records, references = synthetic_clips.generate(n_participants=args.participants, seed=args.seed)
        write_clips(records, out / "synthetic" / "clips.jsonl")
        save_references(references, out / "synthetic" / "references.json")
        clips = [from_record(r) for r in records]
        data_source = "synthetic"
    else:
        if not args.references:
            ap.error("--data에는 --references가 필요함")
        clips = load_clips(args.data)
        references = load_references(args.references)
        data_source = str(args.data)

    label_map = LabelMap.from_clips(clips)
    word_types = {c.word_key: c.word_type for c in clips if c.word_key in label_map.words}
    missing = [w for w in label_map.words if w not in references]
    if missing:
        print(f"✗ 기준 시퀀스가 없는 단어: {missing}", file=sys.stderr)
        return 1
    n_people = len({c.participant_id for c in clips})
    print(f"데이터: {data_source} · 클립 {len(clips)} · 촬영자 {n_people} · 단어 {len(label_map.words)} (+other)")

    cfg = TrainConfig(epochs=args.epochs, seed=args.seed)
    print("교차 검증(촬영자 단위):")
    pairs, m_vals, d_vals, folds, oof = cross_validate(clips, label_map, references, word_types, cfg, args.max_folds)
    m_sum, d_sum = ev.summarize(pairs, m_vals), ev.summarize(pairs, d_vals)
    decision = ev.decide(pairs, m_vals, d_vals, word_types, seed=args.seed)
    # 최종 τ = 교차 검증에서 학습에 쓰지 않은 촬영자의 로짓으로 보정(스펙 §6 기준, CTO 2026-10-16)
    tau_oof = calibrate_temperature(*oof) if oof else None

    print("최종 학습(전체 촬영자) + 변환:")
    model, tau_self, info = train(build_examples(clips, label_map), len(label_map.classes), cfg)
    tau = tau_oof if tau_oof is not None else tau_self
    print(f"  τ: out-of-fold {tau_oof if tau_oof is None else round(tau_oof, 2)} (사용) · 최종 학습 검증 {tau_self:.2f} (참고)")
    if tau_oof is None and info["tau_source"] == "train":
        print("  ⚠️ τ를 학습 데이터로 보정함(촬영자 1명·검증셋 없음) — 이 산출물은 평가·배포에 쓰지 말 것")
    model.save(out / "model.h5")
    labels = label_map.to_labels_json(model_version=args.model_version, temperature=tau)
    (out / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")

    x = convert.fixed_inputs()
    keras_logits = model.predict(x, verbose=0)
    conv = {}
    tfl, msg = convert.to_tflite(model, out / "artifacts" / "android" / "model.tflite")
    conv["tflite"] = {"status": msg}
    if tfl is not None:
        # iOS도 Android와 같은 .tflite 파일을 쓴다(모델 1개, 파일 형식 2가지)
        (out / "artifacts" / "ios").mkdir(parents=True, exist_ok=True)
        shutil.copy(tfl, out / "artifacts" / "ios" / "model.tflite")
        conv["tflite"].update(convert.compare(convert.tflite_predict(tfl, x), keras_logits, tau=tau))
    web, msg = convert.to_tfjs(model, out / "artifacts" / "web")
    conv["tfjs"] = {"status": msg}
    if web is not None:
        got, msg = convert.tfjs_predict(web, x, keras_logits)
        conv["tfjs"]["status"] = msg
        if got is not None:
            conv["tfjs"].update(convert.compare(got, keras_logits, tau=tau))
    for k, v in conv.items():
        print(f"  {k}: {v['status']}" + (f" · {'PASS' if v.get('pass') else 'FAIL'}" if "pass" in v else ""))

    # 변환 일치성 벡터 후보(스펙 §8) — 입력이 합성이라 커밋 가능. 실제 모델 PR에서 vectors/에 올린다
    (out / "conversion-vectors.json").write_text(
        json.dumps(
            {
                "spec_version": spec.SPEC_VERSION,
                "model_version": args.model_version,
                "temperature": tau,
                "data_source": "synthetic fixed inputs (sueori_ai.convert.fixed_inputs)",
                "inputs": x.tolist(),
                "keras_logits": keras_logits.tolist(),
                "tolerances": {"logit": convert.TOL_LOGIT, "score": convert.TOL_SCORE, "verdict": "exact"},
            }
        ),
        encoding="utf-8",
    )

    conversion_ok = all(v.get("pass") is True for v in conv.values())
    release_blocked = args.release and not conversion_ok

    # 매니페스트 초안(스펙 §7.1). 웹은 FE 연동 확인 전까지 enabled=false(CTO 2026-10-16).
    # 배포 후보 모드에서 변환 일치성이 둘 다 PASS가 아니면 만들지 않는다(CTO 2026-10-16)
    if not release_blocked:
        artifacts = {}
        for plat, sub, entry in (("web", "web", "model.json"), ("android", "android", "model.tflite"), ("ios", "ios", "model.tflite")):
            d = out / "artifacts" / sub
            if d.exists():
                artifacts[plat] = {
                    "url": f"…/models/{args.model_version}/{sub}/{entry}",
                    "enabled": plat != "web",
                    "min_app_version": None,
                    "files": convert.artifact_files(d),
                }
        manifest = {
            "spec_version": spec.SPEC_VERSION, "model_version": args.model_version, "enabled": False,
            "labels_url": f"…/models/{args.model_version}/labels.json", "artifacts": artifacts,
            # 단어별 끄기(스펙 §7 제안, PM 2026-10-16): 모델에 있어도 이 단어들은 DTW로 채점
            "disabled_words": decision["disabled_words"],
            "note": "초안 — 전역 enabled=false. 배포 시 저장소 URL·sha256(labels)·min_app_version을 채움",
        }
        (out / "manifest.draft.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {
        "run": run,
        "data_source": data_source,
        "synthetic": data_source == "synthetic",
        "spec_version": spec.SPEC_VERSION,
        "model_version": args.model_version,
        "clips": len(clips),
        "participants": n_people,
        "words": list(label_map.words),
        "folds": folds,
        "model": m_sum,
        "dtw": d_sum,
        "decision": decision,
        "final": {"tau": tau, "tau_oof": tau_oof, "tau_final_val": tau_self, **info},
        "conversion": conv,
        "release": {"requested": args.release, "conversion_ok": conversion_ok, "blocked": release_blocked},
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "report.md").write_text(render_report(report, m_sum, d_sum), encoding="utf-8")
    print(f"\n{(out / 'report.md').read_text(encoding='utf-8')}\n→ {out}")
    if release_blocked:
        print("✗ 배포 후보 모드: 변환 일치성이 둘 다 PASS가 아니라 매니페스트를 만들지 않았음", file=sys.stderr)
        return 2
    return 0


def render_report(r: dict, m_sum: dict, d_sum: dict) -> str:
    dec = r["decision"]
    taus = ", ".join(f"{f['tau']:.2f}" for f in r["folds"])
    tau_oof = r["final"]["tau_oof"]
    lines = [f"# 학습 리포트 {r['run']}", ""]
    if r["synthetic"]:
        lines += ["> ⚠️ **합성 데이터** — 성능 수치는 의미가 없다. 파이프라인 동작 확인용.", ""]
    lines += [
        "> 라벨 한계(프로토콜 §7.4): 수어 사용자 검수가 없으면 점수는 '기준 영상 모방 정확도'다.",
        "",
        f"- 데이터: {r['data_source']} · 클립 {r['clips']} · 촬영자 {r['participants']} · 단어 {len(r['words'])} + other",
        f"- 교차 검증: 촬영자 단위 {len(r['folds'])} fold · fold별 τ(검증 = 학습 촬영자 중 1명) {taus}",
        f"- 최종 τ {r['final']['tau']:.2f} (out-of-fold 로짓 보정: {'—' if tau_oof is None else f'{tau_oof:.2f}'}, "
        f"최종 학습 검증 참고값 {r['final']['tau_final_val']:.2f})",
        f"- 변환: TFLite {r['conversion']['tflite'].get('pass', '—')} / TF.js {r['conversion']['tfjs'].get('pass', '—')}"
        + (" · ⛔ 배포 후보 모드 차단" if r["release"]["blocked"] else ""),
        "",
        "## 스펙 §6 보정 목표 — 모델 vs DTW 폴백 (전체)",
        "",
        _table(m_sum, d_sum),
        "",
        f"## Go/No-Go — 그룹별 ({dec['note']})",
        "",
        "| 그룹 | 판정 | §6 목표 | DTW보다 확실히 나음 | 나빠진 범주 | 표본 부족 | 왼손 격차 |",
        "|---|---|---|---|---|---|---|",
    ]
    for g, v in dec["groups"].items():
        d, lh = v["decision"], v["left_hand"]
        gap = "데이터 없음" if lh["gap"] is None else f"{lh['gap'] * 100:+.1f}%p ({'OK' if lh['ok'] else '초과'})"
        lines.append(
            f"| {g} | **{'GO' if d['go'] else 'NO-GO'}** | {d['meets_spec_targets']} | "
            f"{'; '.join(d['better_reasons']) or '아니오'} | {', '.join(d['regressions']) or '없음'} | "
            f"{', '.join(d['insufficient_samples']) or '없음'} | {gap} |"
        )
    lines += ["", "신뢰구간(촬영자 단위 부트스트랩, 95%)", ""]
    for g, v in dec["groups"].items():
        cis = ", ".join(f"{k} [{lo:+.3f}, {hi:+.3f}]" for k, (lo, hi) in v["ci"].items())
        lines.append(f"- {g}: {cis or '—'}")
    lines += ["", f"## 단어별 정답 통과율 (바닥선 {dec['criteria']['word_floor'] * 100:.0f}%)", "",
              "| 단어 | n | 모델 | DTW | 바닥선 미달 |", "|---|---|---|---|---|"]
    for w, v in dec["per_word"].items():
        lines.append(f"| {w} | {v['n']} | {_fmt(v['model_pass_rate'], True)} | {_fmt(v['dtw_pass_rate'], True)} | "
                     f"{'⚠️ DTW 유지' if v['below_floor'] else ''} |")
    lines += ["", "## 촬영자별 정답 통과율", "", "| 촬영자 | 손 | n | 모델 | DTW |", "|---|---|---|---|---|"]
    for pid, v in dec["per_participant"].items():
        lines.append(f"| {pid} | {v['hand'] or '—'} | {v['n']} | {_fmt(v['model_pass_rate'], True)} | {_fmt(v['dtw_pass_rate'], True)} |")
    lines += ["", f"- 모델로 채점할 단어: {', '.join(dec['model_enabled_words']) or '없음'}",
              f"- DTW 유지 단어(disabled_words): {', '.join(dec['disabled_words']) or '없음'}", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
