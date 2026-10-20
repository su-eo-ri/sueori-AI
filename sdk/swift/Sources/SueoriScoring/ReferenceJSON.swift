// 기준 데이터 JSON 읽기 — 반드시 JSONDecoder를 쓴다(CTO 2026-10-18).
//
// Foundation `JSONSerialization`은 일부 실수를 마지막 비트 하나 다르게 읽는다(실측: 공통 테스트 벡터의 실수
// 72,468개 중 285개). 그러면 다른 플랫폼과 점수가 미세하게 달라진다. 앱이 이 함정에 빠지지 않도록 SDK가
// JSON → Frame 변환을 제공한다.

import Foundation

public enum ReferenceJSON {
    /// 기준 프레임 JSON을 읽는다. 두 형식을 받는다.
    /// - 숫자 배열: `[[[x, y, z] × 21], …]`
    /// - 객체(MVP `reference_landmarks.frames` 형식): `[{"landmarks": [{"x": …, "y": …, "z": …} × 21], "tMs": …}, …]`
    public static func frames(from data: Data) throws -> [Frame] {
        try JSONDecoder().decode([FrameJSON].self, from: data).map(\.frame)
    }

    /// 객체 형식에 `tMs`가 있으면 프레임별 시각을 돌려준다(하나라도 없으면 nil — 녹화 길이는 기본 3초)
    public static func tMs(from data: Data) throws -> [Double]? {
        let fs = try JSONDecoder().decode([FrameJSON].self, from: data)
        let t = fs.compactMap(\.tMs)
        return t.count == fs.count && !t.isEmpty ? t : nil
    }

    /// 기준 프레임 JSON에서 바로 WordReference를 만든다
    public static func word(wordKey: String, type: WordType, framesJSON: Data) throws -> WordReference {
        WordReference(wordKey: wordKey, type: type, referenceFrames: try frames(from: framesJSON),
                      referenceTMs: try tMs(from: framesJSON))
    }

    private struct Point: Decodable {
        let x: Double, y: Double, z: Double
    }

    private enum FrameJSON: Decodable {
        case array(Frame)
        case object([Point], Double?)

        private enum Keys: String, CodingKey { case landmarks, tMs }

        init(from decoder: Decoder) throws {
            if let a = try? decoder.singleValueContainer().decode(Frame.self) {
                self = .array(a)
                return
            }
            let c = try decoder.container(keyedBy: Keys.self)
            self = .object(try c.decode([Point].self, forKey: .landmarks), try c.decodeIfPresent(Double.self, forKey: .tMs))
        }

        var frame: Frame {
            switch self {
            case let .array(f): return f
            case let .object(ps, _): return ps.map { [$0.x, $0.y, $0.z] }
            }
        }

        var tMs: Double? {
            if case let .object(_, t) = self { return t }
            return nil
        }
    }
}
