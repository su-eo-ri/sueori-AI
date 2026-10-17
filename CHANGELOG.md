# Changelog

채점 SDK(TS·Dart·Swift)는 같은 버전으로 함께 올린다(lockstep). 바뀌지 않은 언어는 "변경 없음"으로 적는다.

## v0.1.0 — 2026-10-18

첫 릴리스. 폴백 전용(모델 추론 없음), 스펙 `ai-io-v1`.

- **공통**: `ScoringEngine.score(request) → ScoreResult`, `recordingDurationMs`. 트랙 선택(§3), 유효성(§10.0), 정규화(§2), 텐서(§4, 0.2 대비), 정적 거리·DTW 폴백(§10) + v1 좌우 반전(§10.5), 녹화 길이(§10.3). 입력 `frameSize`·`inputMirrored`는 받기만 함
- **TS** (`sueori-scoring`): 의존성 없음. GitHub Release tarball로 배포
- **Dart** (`sueori_scoring`): 순수 Dart, 의존성 없음
- **Swift** (`SueoriScoring`): Foundation만 사용. iOS 15+, macOS 12+
- 세 SDK 모두 공통 테스트 벡터 v1에서 Python 기준 구현과 비트 단위 일치(Node·Chrome·Dart VM·Android 에뮬레이터·macOS·iOS 시뮬레이터)
