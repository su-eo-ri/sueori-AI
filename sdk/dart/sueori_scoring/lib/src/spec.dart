/// 모델 입출력 스펙 상수 (ai-io-v1). 정본: sueori-AI `src/sueori_ai/spec.py`.
/// 세 SDK(TS·Dart·Swift)와 Python 기준 구현이 같은 값을 쓴다(tests/test_sdk_versions.py가 확인).
library;

const String specVersion = 'ai-io-v1';
const String sdkVersion = '0.1.0';

const int numLandmarks = 21;
const int features = 63;
const List<int> palmIndices = [0, 5, 9, 13, 17];
const double scaleEpsilon = 1e-9;

const int sequenceLength = 32;
const int minDynamicFrames = 8;

const double staticDecay = 2.0;
const double dtwDecay = 6.5;
const int defaultRecordingMs = 3000;
const int minRecordingMs = 2000;
const int maxRecordingMs = 8000;
const double recordingFactor = 1.2;

const double passThreshold = 80.0;
