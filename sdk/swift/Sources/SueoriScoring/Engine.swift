// 채점 엔진 — 앱이 호출하는 공개 API. 설계: 볼트 `06-AI/수어리 - 채점 SDK 설계 (초안).md`.
//
// SDK 0.1은 폴백 전용이다(모델 추론은 0.2에서 추가하며, 이 API는 그대로 유지한다).
//
// ⚠️ 기준 데이터(BE 응답 JSON)는 **JSONDecoder**로 읽을 것. Foundation `JSONSerialization`은 일부 실수를 마지막 비트
// 하나 다르게 읽어서(실측 72,468개 중 285개) 다른 플랫폼과 점수가 미세하게 달라질 수 있다(허용 오차 안이지만 비트 일치가 깨짐).

/// 정적(지문자) / 동적(단어). rawValue는 세 SDK 공통 문자열
public enum WordType: String, Sendable {
    case staticSign = "static"
    case dynamicSign = "dynamic"
}

public struct WordReference {
    /// 단어 정본 키(예: "family/husband"). 모델 라벨·disabled_words 대조에 쓴다
    public let wordKey: String
    public let type: WordType
    /// 기준 프레임들. 정적은 길이 1. 각 프레임은 21개 [x, y, z](MediaPipe 이미지 좌표)
    public let referenceFrames: [Frame]
    /// 기준 프레임별 시각(ms). 녹화 길이 계산에 쓴다(없으면 3초)
    public let referenceTMs: [Double]?

    public init(wordKey: String, type: WordType, referenceFrames: [Frame], referenceTMs: [Double]? = nil) {
        self.wordKey = wordKey
        self.type = type
        self.referenceFrames = referenceFrames
        self.referenceTMs = referenceTMs
    }
}

public struct CapturedHand {
    /// MediaPipe handedness 라벨("Left"/"Right"). 없으면 nil
    public let handedness: String?
    public let landmarks: Frame
    public init(handedness: String?, landmarks: Frame) {
        self.handedness = handedness
        self.landmarks = landmarks
    }
}

public struct CapturedFrame {
    /// 프레임 캡처 시각(ms, 추론 완료 시각이 아님)
    public let tMs: Double
    /// 잡힌 손 전부. 손이 없으면 빈 배열
    public let hands: [CapturedHand]
    public init(tMs: Double, hands: [CapturedHand]) {
        self.tMs = tMs
        self.hands = hands
    }
}

public struct FrameSize {
    public let width: Int
    public let height: Int
    public let rotation: Int
    public init(width: Int, height: Int, rotation: Int = 0) {
        self.width = width
        self.height = height
        self.rotation = rotation
    }
}

public struct ScoreRequest {
    public let word: WordReference
    public let capture: [CapturedFrame]
    /// 0.1은 받기만 한다(스펙 TODO-7 종횡비 보정 대비)
    public let frameSize: FrameSize?
    /// 0.1은 받기만 한다
    public let inputMirrored: Bool?

    public init(word: WordReference, capture: [CapturedFrame], frameSize: FrameSize? = nil, inputMirrored: Bool? = nil) {
        self.word = word
        self.capture = capture
        self.frameSize = frameSize
        self.inputMirrored = inputMirrored
    }
}

/// rawValue는 세 SDK 공통 문자열(BE 저장·로그용)
public enum ScoreStatus: String, Sendable {
    case ok
    case noHand = "no_hand"
    case insufficientFrames = "insufficient_frames"
    case invalid
}

public enum ScoreMethod: String, Sendable {
    case model, dtw, distance
}

public enum FallbackReason: String, Sendable {
    case modelNotConfigured = "model_not_configured"
    case modelDisabled = "model_disabled"
    case wordNotInModel = "word_not_in_model"
    case wordDisabled = "word_disabled"
    case loadFailed = "load_failed"
    case shaMismatch = "sha_mismatch"
    case specMismatch = "spec_mismatch"
    case appVersion = "app_version"
}

public struct ScoreResult: Equatable {
    public let status: ScoreStatus
    /// 0~100. status가 ok일 때만
    public let score: Double?
    public var passed: Bool { score.map { $0 >= Spec.passThreshold } ?? false }
    public var threshold: Double { Spec.passThreshold }
    public let method: ScoreMethod?
    /// 모델을 쓰지 않은 이유. 0.1은 항상 modelNotConfigured
    public let fallbackReason: FallbackReason?
    /// 폴백에서 좌우 반전 쪽을 썼는지(스펙 §10.5)
    public let mirrored: Bool
    /// 채점에 쓴 손(handedness 라벨, 없으면 "")
    public let track: String?
    public var specVersion: String { Spec.specVersion }
    public let modelVersion: String?
    public var sdkVersion: String { Spec.sdkVersion }
    /// 폴백: "distance" 또는 "avgCost"·"pathLength"
    public let details: [String: Double]

    static func empty(_ status: ScoreStatus) -> ScoreResult {
        ScoreResult(status: status, score: nil, method: nil, fallbackReason: .modelNotConfigured,
                    mirrored: false, track: nil, modelVersion: nil, details: [:])
    }

    /// 세 SDK 공통 필드 이름(TS ScoreResult와 같음)
    public func toDictionary() -> [String: Any?] {
        [
            "status": status.rawValue, "score": score, "passed": passed, "threshold": threshold,
            "method": method?.rawValue, "fallbackReason": fallbackReason?.rawValue, "mirrored": mirrored,
            "track": track, "specVersion": specVersion, "modelVersion": modelVersion, "sdkVersion": sdkVersion,
            "details": details,
        ]
    }
}

public final class ScoringEngine {
    /// manifestURL, enableModel은 0.2에서 쓰는 자리. 0.1은 무시하고 항상 폴백
    public init(manifestURL: String? = nil, enableModel: Bool = true) {}

    public var specVersion: String { Spec.specVersion }
    public var sdkVersion: String { Spec.sdkVersion }

    /// 동적 단어 녹화 길이(스펙 §10.3). 기준 시각이 없으면 3000ms
    public func recordingDurationMs(_ word: WordReference) -> Int {
        guard let t = word.referenceTMs, let first = t.first, let last = t.last else { return Spec.defaultRecordingMs }
        return Core.recordingDurationMs(firstTMs: first, lastTMs: last)
    }

    /// 채점. 트랙(손)마다 채점해 최고점을 쓴다(동점이면 먼저 나타난 손).
    /// 정적 단어는 트랙의 가운데 프레임 1장, 동적 단어는 트랙 전체(유효 프레임 8개 이상)를 쓴다.
    /// 기준 데이터가 잘못되면 InvalidInput을 던진다(앱·콘텐츠 오류). 사용자 입력 문제는 status로 돌려준다.
    public func score(_ request: ScoreRequest) throws -> ScoreResult {
        let word = request.word
        guard !word.referenceFrames.isEmpty else { throw InvalidInput(message: "기준 프레임이 없음") }
        try word.referenceFrames.forEach(Core.checkFrame)

        var order: [String] = []
        var byLabel: [String: [Frame]] = [:]
        for fr in request.capture {
            for hand in fr.hands {
                let label = hand.handedness ?? ""
                if byLabel[label] == nil {
                    byLabel[label] = []
                    order.append(label)
                }
                byLabel[label]!.append(hand.landmarks)
            }
        }
        if order.isEmpty { return .empty(.noHand) }

        let tracks = order.map { Core.TrackInput(label: $0, frames: byLabel[$0]!) }
        let out = Core.scoreTracks(word.referenceFrames, tracks, dynamic: word.type == .dynamicSign)
        guard let best = out.best else { return .empty(out.sawShort ? .insufficientFrames : .invalid) }

        switch best {
        case let .distance(label, r):
            return ScoreResult(status: .ok, score: r.score, method: .distance, fallbackReason: .modelNotConfigured,
                               mirrored: r.mirrored, track: label, modelVersion: nil, details: ["distance": r.distance])
        case let .dtw(label, r):
            return ScoreResult(status: .ok, score: r.score, method: .dtw, fallbackReason: .modelNotConfigured,
                               mirrored: r.mirrored, track: label, modelVersion: nil,
                               details: ["avgCost": r.avgCost, "pathLength": Double(r.pathLength)])
        }
    }
}
