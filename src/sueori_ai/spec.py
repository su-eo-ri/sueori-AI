"""모델 입출력 스펙(ai-io-v0, ai-io-v1)의 상수.

정본 문서: 볼트 `06-AI/수어리 - 모델 입출력 스펙.md` (v0 2026-10-15 고정, v1 2026-10-16 결정).
여기 값을 바꾸면 클라이언트(TS·Dart·Swift) 결과가 달라지므로 SPEC_VERSION을 올려야 한다.
v0 → v1 변경: 폴백 좌우 반전(MIRROR_FALLBACK). 그 밖의 상수는 같다.
"""

SPEC_VERSION = "ai-io-v1"  # 현재 버전(클라이언트는 v1부터 구현, CTO 2026-10-16)
SUPPORTED_VERSIONS = ("ai-io-v0", "ai-io-v1")  # v0은 기록용(벡터 파일 유지)

# 버전별 차이 — v1: 폴백(§10)에서 입력을 정규화 후 x → −x로 반전한 버전도 채점해 더 나은 쪽을 쓴다
# (왼손 사용자, 스펙 TODO-9, 사용자 결정 2026-10-16). 모델 입력(§2·§4·§5)은 v0과 같다.
MIRROR_FALLBACK = {"ai-io-v0": False, "ai-io-v1": True}


def check_version(version: str) -> str:
    if version not in SUPPORTED_VERSIONS:
        raise ValueError(f"지원하지 않는 spec_version: {version} (지원: {SUPPORTED_VERSIONS})")
    return version


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
