"""세 SDK(TS·Dart·Swift)와 Python 기준 구현의 버전·스펙 상수가 맞는지 확인한다.

- SDK 버전은 lockstep(세 SDK가 같은 버전, CTO 2026-10-18). 레포 태그 v<버전>과 같아야 한다(릴리스 워크플로가 확인)
- 스펙 버전과 상수(감쇠 계수, 통과선 등)는 Python spec.py가 정본
"""

import json
import re
from pathlib import Path

from sueori_ai import spec

ROOT = Path(__file__).resolve().parent.parent
TS_PKG = ROOT / "sdk/ts/package.json"
TS_SPEC = ROOT / "sdk/ts/src/spec.ts"
DART_PUB = ROOT / "sdk/dart/sueori_scoring/pubspec.yaml"
DART_SPEC = ROOT / "sdk/dart/sueori_scoring/lib/src/spec.dart"
SWIFT_SPEC = ROOT / "sdk/swift/Sources/SueoriScoring/Spec.swift"


def _const(path: Path, name: str) -> str:
    m = re.search(rf"{name}\s*(?::\s*\w+\s*)?=\s*([^;\n]+)", path.read_text(encoding="utf-8"))
    assert m, f"{path.name}: {name} 없음"
    return m.group(1).strip().strip('"').strip("'").rstrip(";").strip()


def sdk_versions() -> dict[str, str]:
    return {
        "ts package.json": json.loads(TS_PKG.read_text())["version"],
        "ts spec.ts": _const(TS_SPEC, "SDK_VERSION"),
        "dart pubspec": re.search(r"^version:\s*(\S+)", DART_PUB.read_text(), re.M).group(1),
        "dart spec.dart": _const(DART_SPEC, "sdkVersion"),
        "swift Spec.swift": _const(SWIFT_SPEC, "sdkVersion"),
    }


def test_sdk_versions_lockstep():
    v = sdk_versions()
    assert len(set(v.values())) == 1, f"SDK 버전이 다름: {v}"


def test_sdk_spec_version_matches_python():
    assert _const(TS_SPEC, "SPEC_VERSION") == spec.SPEC_VERSION
    assert _const(DART_SPEC, "specVersion") == spec.SPEC_VERSION
    assert _const(SWIFT_SPEC, "specVersion") == spec.SPEC_VERSION


def test_sdk_constants_match_python():
    pairs = {
        "STATIC_DECAY": ("staticDecay", spec.STATIC_DECAY),
        "DTW_DECAY": ("dtwDecay", spec.DTW_DECAY),
        "PASS_THRESHOLD": ("passThreshold", spec.PASS_THRESHOLD),
        "SCALE_EPSILON": ("scaleEpsilon", spec.SCALE_EPSILON),
        "RECORDING_FACTOR": ("recordingFactor", spec.RECORDING_FACTOR),
        "MIN_DYNAMIC_FRAMES": ("minDynamicFrames", spec.MIN_DYNAMIC_FRAMES),
        "SEQUENCE_LENGTH": ("sequenceLength", spec.SEQUENCE_LENGTH),
        "DEFAULT_RECORDING_MS": ("defaultRecordingMs", spec.DEFAULT_RECORDING_MS),
        "MIN_RECORDING_MS": ("minRecordingMs", spec.MIN_RECORDING_MS),
        "MAX_RECORDING_MS": ("maxRecordingMs", spec.MAX_RECORDING_MS),
    }
    for ts_name, (camel, expected) in pairs.items():
        assert float(_const(TS_SPEC, ts_name)) == float(expected), ts_name
        assert float(_const(DART_SPEC, camel)) == float(expected), camel
        assert float(_const(SWIFT_SPEC, camel)) == float(expected), camel
