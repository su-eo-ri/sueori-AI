/**
 * 모델 입출력 스펙 상수 (ai-io-v1). 정본: sueori-AI `src/sueori_ai/spec.py`, 볼트 `06-AI/수어리 - 모델 입출력 스펙.md`.
 * 세 SDK(TS·Dart·Swift)와 Python 기준 구현이 같은 값을 쓴다(tests/test_sdk_versions.py가 확인).
 */
export const SPEC_VERSION = "ai-io-v1";
export const SDK_VERSION = "0.1.0";

export const NUM_LANDMARKS = 21;
export const FEATURES = 63;
export const PALM_INDICES = [0, 5, 9, 13, 17] as const;
export const SCALE_EPSILON = 1e-9;

export const SEQUENCE_LENGTH = 32;
export const MIN_DYNAMIC_FRAMES = 8;

export const STATIC_DECAY = 2.0;
export const DTW_DECAY = 6.5;
export const DEFAULT_RECORDING_MS = 3000;
export const MIN_RECORDING_MS = 2000;
export const MAX_RECORDING_MS = 8000;
export const RECORDING_FACTOR = 1.2;

export const PASS_THRESHOLD = 80.0;
