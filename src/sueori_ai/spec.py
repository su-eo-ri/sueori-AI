"""모델 입출력 스펙 ai-io-v0의 상수.

정본 문서: 볼트 `06-AI/수어리 - 모델 입출력 스펙 v0.md` (2026-10-15 고정).
여기 값을 바꾸면 클라이언트(TS·Dart·Swift) 결과가 달라지므로 SPEC_VERSION을 올려야 한다.
"""

SPEC_VERSION = "ai-io-v0"

# §1 손 랜드마크
NUM_LANDMARKS = 21
DIMS = 3
FEATURES = NUM_LANDMARKS * DIMS  # 63

# §2 정규화: 손바닥 5점(손목 0, MCP 5·9·13·17)
PALM_INDICES = (0, 5, 9, 13, 17)
SCALE_EPSILON = 1e-9

# §4 시간축
SEQUENCE_LENGTH = 32  # T
MIN_DYNAMIC_FRAMES = 8

# §10 폴백 채점
STATIC_DECAY = 2.0
DTW_DECAY = 6.5
DEFAULT_RECORDING_MS = 3000
MIN_RECORDING_MS = 2000
MAX_RECORDING_MS = 8000
RECORDING_FACTOR = 1.2

# §6 통과선
PASS_THRESHOLD = 80.0
