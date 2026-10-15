# sueori-AI

수어리 수어 채점 모델: 데이터 수집·학습·평가, Python 기준 구현, 공통 테스트 벡터

MediaPipe Hand Landmarker가 뽑은 손 랜드마크 시퀀스를 입력으로 받아 0~100점을 내는 **단일 모델**을
Keras로 학습하고, 웹(TF.js)과 Android·iOS(TFLite)로 변환해 온디바이스로 추론한다. 모델이 없거나
꺼졌을 때는 각 클라이언트가 같은 명세의 DTW/거리 채점으로 폴백한다.

- 입출력 계약: 모델 입출력 스펙 **`ai-io-v1`** (현재) — 정규화, 리샘플, 텐서 `[1, 32, 63] → [1, 48]`,
  점수, 폴백 알고리즘, 테스트 벡터 허용 오차
  - `ai-io-v0`(2026-10-15 고정)은 기록용이다. **클라이언트는 v1부터 구현한다**
  - **v0 → v1 변경은 폴백(§10)의 좌우 반전 하나뿐**이다. 입력을 정규화한 뒤 x를 반전(x → −x)한 버전도 채점해 비용이 더 작은 쪽을 쓰고,
    동률이면 원본을 쓴다. v0 폴백은 오른손 기준과 그대로 비교해서 왼손 사용자가 정확히 해도 통과할 수 없었다(실제 국립국어원 기준: 지문자 0/40,
    가족 0.3~13점). 반전으로 새로 생긴 오답은 실제 기준 지문자 0/1560쌍, 가족 0/42쌍이었다. 모델 입력(§2·§4·§5)은 v0과 같다
  - 다음 변경(학습 후 데이터 표준값)은 `ai-io-v2`로 낸다
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
  spec.py           스펙 상수와 버전(v0·v1, 버전별 폴백 반전 여부)
  reference.py      Python 기준 구현 — §2 정규화, §4 리샘플·텐서, §10 폴백(거리·DTW·v1 반전·트랙 선택·녹화 길이)
  synthetic.py      합성 손 랜드마크 생성기(벡터·스모크용)
  fast.py           numpy 빠른 경로(학습·평가용) — reference.py와 비트 단위로 같은 결과
  clips.py          수집 클립 레코드(데이터 수집 프로토콜 protocol-v0 §6) 읽기·검증·트랙 분리
  synthetic_clips.py  합성 클립 데이터셋(테스트 가명 T001…) + 기준 시퀀스
  dataset.py        라벨 맵(labels.json), 클립 → [32, 63] 예제, 촬영자 단위 분할, 좌우 반전 증강
  model.py          v0 1D-CNN, 학습, 온도(τ) 보정
  evaluate.py       모델 vs DTW 폴백 — 스펙 §6 목표 표, Go/No-Go 판정(초안)
  convert.py        TFLite / TF.js 변환, 변환 일치성 검사, 고정 입력(합성)
tools/
  gen_vectors.py    공통 테스트 벡터 생성 → vectors/ai-io-v0-vectors.json(기록용), ai-io-v1-vectors.json(현재)
  smoke_convert.py  더미 모델 Keras → TFLite / TF.js 변환 + 출력 일치성 확인
  train_pipeline.py 학습 파이프라인(교차 검증 → 최종 학습 → 변환 → 리포트)
  tfjs_smoke/       TF.js 런타임(Node)으로 변환 모델을 로드·추론
vectors/
  ai-io-v0-vectors.json  공통 테스트 벡터 v0 (기록용, 합성 데이터)
  ai-io-v1-vectors.json  공통 테스트 벡터 v1 (현재, 합성 데이터)
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

## 학습 파이프라인

```bash
uv run python tools/train_pipeline.py --synthetic                 # 합성 데이터로 끝까지 (약 20초)
uv run python tools/train_pipeline.py --synthetic --participants 3 --epochs 5 --max-folds 1   # 빠른 확인

# 실제 데이터: 수집 DB 스냅샷(JSONL, 클립 1개 = 1줄) + 기준 시퀀스 {word_key: frames}
# ⚠️ 실제 데이터는 이 레포 밖 경로에 두고 가리킨다(커밋 금지)
uv run python tools/train_pipeline.py --data /path/to/snapshot --references /path/to/references.json
```

흐름
1. 클립을 읽고 검증한다(`qc_status = reject`는 제외). 라벨 맵 = 정답 클립의 단어(정렬) + `other`
2. **촬영자 단위 교차 검증**(한 명씩 평가로 뺌): 학습 → 온도(τ) 보정 → 평가 촬영자의 (클립, 타깃 단어) 쌍을
   **모델과 DTW 폴백으로 각각** 채점한다. 채점은 클라이언트와 같은 방식이다(트랙별 최고점, 타깃 단어 유형에 맞는 입력)
3. 스펙 §6 보정 목표 표(정답 / 다른 단어 / 정지 자세 / 무관 동작 / 근접 오답)를 모델·DTW 양쪽으로 만들고 Go/No-Go를 판정한다
4. 전체 촬영자로 최종 학습 → `labels.json`, TFLite(Android·iOS 같은 파일), TF.js 변환 → 변환 일치성 검사
5. `build/train/<run>/`에 `report.md`·`report.json`, `labels.json`, `manifest.draft.json`(웹 `enabled=false`),
   `conversion-vectors.json`(합성 고정 입력 + Keras 로짓), `artifacts/{web,android,ios}`

> **Go/No-Go 기준은 초안이다**(`evaluate.GO_DRAFT`): §6 목표를 모두 충족하고, DTW보다 확실히 나으며(정답 통과율 +10%p 이상
> 또는 오답 통과율이 DTW의 절반 이하), 어느 범주에서도 DTW보다 2%p 넘게 나빠지지 않아야 한다. PM·사용자 합의가 필요하다.
> 합성 데이터의 수치는 의미가 없다.

## 공통 테스트 벡터 사용법 (클라이언트용)

**`vectors/ai-io-v1-vectors.json`**의 특정 버전을 클라이언트 레포에 복사해 두고, 테스트에서 파일 sha256과
`spec_version`을 확인한다. `ai-io-v0-vectors.json`은 기록용이다(코드가 바뀌어도 바이트 단위로 같게 재생성되는지 테스트한다).

- v1 파일 = v0 케이스 전부(반전이 쓰이지 않는 케이스는 점수가 v0과 같음) + 반전 전용 케이스(`v1_…`)
  - `v1_mirror_wins_left_hand*`: 왼손 입력(이미지 좌우 반전)에서 반전 쪽이 이김. 점수가 오른손 대응 케이스와 같음
  - `v1_mirror_tie_flat_x`: 정규화 후 x 성분이 전부 0이라 원본과 반전의 비용이 **비트 단위로 같음** → 원본을 써야 함
  - `v1_left_hand_track_wins_by_mirror`: 트랙 선택(§3)과 반전이 함께 쓰이는 경우
- `static_score`·`dtw`의 v1 기대값에는 `mirrored`(반전 쪽을 썼는지)가 있다. 점수는 동률에서 어느 쪽을 골라도 같으므로,
  동률 규칙을 확인하려면 `mirrored`도 비교한다
- 반전은 **정규화 후 x → −x**다. 이미지 좌표에서 1 − x로 뒤집으면 수학적으로는 같지만 마지막 비트가 달라질 수 있다

> **버전 계획 (2026-10-16 갱신)**: 지금은 **`ai-io-v1`로 구현**한다(v1 = 폴백 좌우 반전). 학습을 진행한 뒤 데이터 표준값(정규화 등)을
> 다시 정해 점수 결과가 바뀌면 `ai-io-v2`로 올리고 `vectors/ai-io-v2-vectors.json`을 **새 파일로** 추가한다.
> 이전 버전 파일은 지우지 않으며, 어느 버전을 통과하는지는 `spec_version`으로 구분한다.

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
| GRU (`reset_after=False, unroll=True`) | ✅ | ✅ (Node 순수 JS CPU 백엔드에서 CNN보다 약 15배 느림 — 브라우저 미측정) | ≤ 3e-7 / ≤ 2e-6 |

→ v0은 **1D-CNN**으로 시작한다. RNN 계열을 쓰려면 위 호환 설정이 필수다.

⚠️ 속도 수치는 `tools/tfjs_smoke`의 **Node 순수 JS CPU 백엔드** 측정값이라 브라우저(WebGL/WASM) 성능을 대표하지 않는다.
"GRU는 웹에서 15배 느리다"는 뜻이 아니다. 이 스모크의 목적은 변환·로드 가능 여부와 출력 일치성이다. RNN을 쓸지는
React TF.js 스파이크의 브라우저 실측(모델 로딩 시간, 저사양 Android Chrome 추론 ms)으로 판단한다.
