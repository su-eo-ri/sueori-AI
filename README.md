# sueori-AI

수어리 수어 채점 모델: 데이터 수집·학습·평가, Python 기준 구현, 공통 테스트 벡터

MediaPipe Hand Landmarker가 뽑은 손 랜드마크 시퀀스를 입력으로 받아 0~100점을 내는 **단일 모델**을
Keras로 학습하고, 웹(TF.js)과 Android·iOS(TFLite)로 변환해 온디바이스로 추론한다. 모델이 없거나
꺼졌을 때는 각 클라이언트가 같은 명세의 DTW/거리 채점으로 폴백한다.

- 입출력 계약: 모델 입출력 스펙 **`ai-io-v0`** (2026-10-15 고정) — 정규화, 리샘플, 텐서 `[1, 32, 63] → [1, 48]`,
  점수, 폴백 알고리즘, 테스트 벡터 허용 오차
- 이 레포가 **정본**인 것: Python 기준 구현(`src/sueori_ai/reference.py`)과 공통 테스트 벡터(`vectors/`)
- 클라이언트: `sueori-FE`(React/TS), `sueori-AOS`(Flutter/Dart), `sueori-IOS`(Swift)는 각자 구현하고 벡터를 통과해야 한다

## ⚠️ public 레포 — 커밋 금지

- 수집 데이터(실제 촬영자 랜드마크·샘플), 데이터셋 파일
- 동의서, 가명 ID 매핑, 키·시크릿·스토리지 자격증명(업로드 스크립트는 환경변수로만 받는다)
- 학습된 모델 가중치·변환 산출물(`*.h5`, `*.keras`, `*.tflite`, TF.js `*.bin` 등) — 커밋 여부는 별도 결정 전까지 금지
- 테스트 벡터에는 **합성 데이터만** 쓴다. 국립국어원 기준 데이터(CC BY-SA 2.0)에서 만든 파일을 넣는다면
  파일 메타에 출처·라이선스를 적고 PR 본문에 명시한다

`.gitignore`가 위 경로를 막고 있지만, 커밋 전에 `git status`로 한 번 더 확인한다.

## 구성

```
src/sueori_ai/
  spec.py           ai-io-v0 상수 (T=32, 손바닥 5점, 감쇠 계수 2.0 / 6.5, 통과선 80 …)
  reference.py      Python 기준 구현 — §2 정규화, §4 리샘플·텐서, §10 폴백(거리·DTW·트랙 선택·녹화 길이)
  synthetic.py      합성 손 랜드마크 생성기(벡터·스모크용)
tools/
  gen_vectors.py    공통 테스트 벡터 생성 → vectors/ai-io-v0-vectors.json
  smoke_convert.py  더미 모델 Keras → TFLite / TF.js 변환 + 출력 일치성 확인
  tfjs_smoke/       TF.js 런타임(Node)으로 변환 모델을 로드·추론
vectors/
  ai-io-v0-vectors.json  공통 테스트 벡터 (합성 데이터)
tests/              pytest
```

## 개발 환경

툴체인은 정확한 버전으로 고정한다(`pyproject.toml` + `uv.lock`).

| 항목 | 버전 | 비고 |
|---|---|---|
| Python | 3.11 | `.python-version` |
| tensorflow | 2.19.0 | |
| tf-keras | 2.19.0 | 모델은 **Keras 2(`tf_keras`)로 정의**한다. Keras 3 + tensorflowjs 호환 문제 회피 |
| tensorflowjs | 4.22.0 | 변환기. 웹 런타임 `@tensorflow/tfjs`도 4.22.0으로 맞춘다 |
| ydf | 0.15.0 | 0.16은 protobuf 6.x로 빌드돼 TF 2.19의 protobuf 5.x와 import 시 충돌 |
| setuptools | 80.10.2 | 81 이상에는 `pkg_resources`가 없어 tensorflowjs import 실패 |

```bash
uv sync                                   # .venv 생성 + 의존성 설치
uv run pytest                             # 기준 구현 + 벡터 테스트
uv run python tools/gen_vectors.py        # 벡터 재생성 (--check: 최신인지만 확인)

cd tools/tfjs_smoke && npm ci && cd ../.. # TF.js 런타임(최초 1회)
uv run python tools/smoke_convert.py      # 변환 스모크
```

툴체인 버전을 올릴 때는 `smoke_convert.py`가 통과하는지 먼저 확인한다. tensorflowjs 변환기는 TF/Keras
최신 버전을 늦게 따라가는 편이다.

## 공통 테스트 벡터 사용법 (클라이언트용)

`vectors/ai-io-v0-vectors.json`의 특정 버전을 클라이언트 레포에 복사해 두고, 테스트에서 파일 sha256과
`spec_version`을 확인한다.

- 실수는 가장 짧게 왕복되는 표현으로 적혀 있어 Float64 값이 그대로 복원된다
- NaN·Infinity 좌표는 문자열 `"NaN"`, `"Infinity"`, `"-Infinity"`
- `invalid: true` 케이스는 채점·정규화가 거부(예외 또는 채점 불가)되어야 한다
- `dtw` 기대값은 §10.2 알고리즘 수준이다(동률 검증용 짧은 시퀀스 포함). §4 유효성(N<8)은 `invalid` 케이스로 확인한다
- 허용 오차는 파일의 `tolerances`를 쓴다: DTW 경로 길이·녹화 길이는 정확히 일치, 거리·평균 비용 `1e-12`,
  점수 `1e-9`, 정규화·텐서(Float64) `1e-6`, 텐서(Float32) `1e-5`
- 같은 점수가 나오려면 스펙 §10.0의 계산 규칙을 지켜야 한다: Float64, 랜드마크 거리는 점 0→20 순서로 순차
  합산 후 ÷21, `Math.hypot`·병렬 합산·FMA 금지, DTW 동률은 diag → up → left.
  `ties_all_other_orders_differ` 케이스는 다른 동률 순서로 구현하면 경로 길이가 달라져 실패한다
- 벡터는 macOS arm64 / Python 3.11에서 생성했다. 다른 플랫폼에서 `--check`를 돌리면 `exp`의 마지막 비트 차이로
  문자열이 다를 수 있다(값은 허용 오차 안)

## 변환 스모크 결과 (2026-10-15, macOS arm64)

| 구조 | TFLite (Flex 없이) | TF.js (Node 로드·추론) | 출력 차이 (로짓 / 점수) |
|---|---|---|---|
| 1D-CNN (Conv1D×2 + GAP + Dense) | ✅ | ✅ | ≤ 3e-7 / ≤ 1e-6 |
| GRU (Keras 기본값) | ❌ TensorList 연산을 기본 연산으로 못 바꿈 | ❌ `reset_after=True` 미지원 | — |
| GRU (`reset_after=False, unroll=True`) | ✅ | ✅ (Node CPU 추론 약 15배 느림) | ≤ 3e-7 / ≤ 2e-6 |

→ v0은 **1D-CNN**으로 시작한다. RNN 계열을 쓰려면 위 호환 설정이 필수이고, 웹 추론 속도를 따로 확인해야 한다.
Node CPU 측정이라 브라우저(WebGL/WASM) 속도는 별도 스파이크에서 잰다.
