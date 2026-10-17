# 수어리 채점 SDK (`sueori-scoring`)

앱은 **`score()` 하나만 호출**하고, 채점(트랙 선택·정규화·폴백·좌우 반전·모델)은 SDK가 맡는다. 채점은 기기 안에서 하며 손 좌표를 서버로 보내지 않는다.

| 언어 | 패키지 | 위치 |
|---|---|---|
| TypeScript (웹, React) | `sueori-scoring` | `sdk/ts` |
| Dart (Android, Flutter) | `sueori_scoring` | `sdk/dart/sueori_scoring` |
| Swift (iOS) | `SueoriScoring` | `sdk/swift` (루트 `Package.swift`) |

- 스펙: `ai-io-v1` (볼트 `06-AI/수어리 - 모델 입출력 스펙.md`)
- 세 SDK는 **같은 버전으로 함께 올린다**(lockstep). 레포 태그 `v0.1.0` 하나가 세 SDK의 버전이다
- **0.1 = 폴백 전용**(DTW/거리, v1 좌우 반전). 0.2에서 모델 추론을 추가해도 API는 그대로다

## 설치

**TypeScript** — GitHub Release에 첨부된 tarball(npm 레지스트리에 공개하지 않음)
```bash
npm install https://github.com/su-eo-ri/sueori-AI/releases/download/v0.1.0/sueori-scoring-0.1.0.tgz
```

**Dart / Flutter** — `pubspec.yaml`
```yaml
dependencies:
  sueori_scoring:
    git:
      url: https://github.com/su-eo-ri/sueori-AI.git
      ref: v0.1.0
      path: sdk/dart/sueori_scoring
```

**Swift** — Xcode "Add Package Dependency" → `https://github.com/su-eo-ri/sueori-AI.git`, 버전 `0.1.0`
```swift
.package(url: "https://github.com/su-eo-ri/sueori-AI.git", from: "0.1.0")
// target: .product(name: "SueoriScoring", package: "sueori-AI")
```

## 사용법

```ts
import { ScoringEngine } from "sueori-scoring";

const engine = new ScoringEngine();

// 동적 단어는 녹화 길이를 SDK에서 받는다(기준 동작 길이 × 1.2, 2~8초)
const ms = engine.recordingDurationMs({ referenceTMs });

const result = engine.score({
  word: { wordKey: "family/husband", type: "dynamic", referenceFrames, referenceTMs },
  capture: frames.map((f) => ({ tMs: f.captureTimeMs, hands: f.hands })), // 손이 없는 프레임은 hands: []
  frameSize: { width: 1280, height: 720, rotation: 0 }, // 0.1은 받기만 함
  inputMirrored: false,                                 // 0.1은 받기만 함
});

if (result.status === "ok") show(result.score, result.passed);
else showRetry(result.status); // "no_hand" | "insufficient_frames" | "invalid"
```

Dart(`ScoringEngine().score(ScoreRequest(...))`)와 Swift(`try ScoringEngine().score(ScoreRequest(...))`)도 같은 모양이다.

### 앱이 지킬 것 (캡처 규칙)
- 약 **33ms 간격(약 30fps)** 으로 캡처하고, `tMs`는 **프레임을 캡처한 시각**을 넣는다(추론이 끝난 시각이 아님). 같은 프레임을 두 번 넣지 않는다
- MediaPipe가 잡은 **손을 전부** 넣는다(`handedness` 포함). 손 고르기는 SDK가 한다
- 좌표는 MediaPipe `landmarks`(이미지 좌표) **그대로** 넣는다. 화면 표시용 좌우 반전을 좌표에 적용하지 않는다
- 정적 단어는 손모양을 유지하는 동안 여러 장을 넣어도 된다(SDK가 가운데 프레임을 쓴다)
- **iOS**: 기준 데이터(BE JSON)는 **`JSONDecoder`로 읽는다.** `JSONSerialization`은 일부 실수를 마지막 비트 하나 다르게 읽는다(실측 72,468개 중 285개)

### ScoreResult
| 필드 | 뜻 |
|---|---|
| `status` | `ok` / `no_hand` / `insufficient_frames`(동적 유효 프레임 < 8) / `invalid`(좌표 이상) |
| `score`, `passed`, `threshold` | 0~100, `score ≥ 80`, 80 |
| `method` | `dtw` / `distance` (0.2부터 `model`) |
| `fallbackReason` | 모델을 쓰지 않은 이유. 0.1은 항상 `model_not_configured` |
| `mirrored` | 좌우 반전 쪽으로 채점했는지(왼손) |
| `track` | 채점에 쓴 손(handedness 라벨) |
| `specVersion`, `modelVersion`, `sdkVersion` | 결과 저장(BE)용 |
| `details` | `distance` 또는 `avgCost`·`pathLength` |

기준 데이터가 잘못되면(빈 배열, 좌표 21개가 아님, NaN) 예외를 던진다(TS·Swift `InvalidInput`, Dart `InvalidInput`). 사용자 입력 문제는 예외가 아니라 `status`로 돌려준다.

## 검증
세 SDK 모두 `vectors/ai-io-v1-vectors.json`(Python 기준 구현이 만든 답안지)을 테스트에서 통과해야 한다. 2026-10-18 기준 **세 SDK 모두 Python과 비트 단위로 같다**(정적·동적 점수 18/18).

| 환경 | 결과 |
|---|---|
| Node 26 (TS) | 18/18 |
| Chrome 155 headless (TS) | 22/22 (정규화 포함) |
| Dart VM (macOS) | 18/18 |
| Android 17 에뮬레이터, Flutter release (Dart) | 18/18 |
| macOS (Swift) | 18/18 |
| iOS 26.5 시뮬레이터 (Swift) | 18/18 |

```bash
(cd sdk/ts && npm ci && npm test)
(cd sdk/dart/sueori_scoring && dart pub get && dart test)
swift test
```
