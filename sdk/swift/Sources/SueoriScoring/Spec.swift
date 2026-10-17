// 모델 입출력 스펙 상수 (ai-io-v1). 정본: sueori-AI `src/sueori_ai/spec.py`.
// 세 SDK(TS·Dart·Swift)와 Python 기준 구현이 같은 값을 쓴다(tests/test_sdk_versions.py가 확인).

public enum Spec {
    public static let specVersion = "ai-io-v1"
    public static let sdkVersion = "0.1.0"

    public static let numLandmarks = 21
    public static let features = 63
    public static let palmIndices = [0, 5, 9, 13, 17]
    public static let scaleEpsilon = 1e-9

    public static let sequenceLength = 32
    public static let minDynamicFrames = 8

    public static let staticDecay = 2.0
    public static let dtwDecay = 6.5
    public static let defaultRecordingMs = 3000
    public static let minRecordingMs = 2000
    public static let maxRecordingMs = 8000
    public static let recordingFactor = 1.2

    public static let passThreshold = 80.0
}
