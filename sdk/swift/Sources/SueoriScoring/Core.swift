// 채점 코어 — 스펙 ai-io-v1 §2·§3·§4·§10. 외부 의존성 없음(Foundation의 exp만 사용).
//
// 비트 단위 재현 규칙(§10.0)
// - 모든 계산은 Float64(Double)
// - 제곱합은 `dx * dx + dy * dy + dz * dz`를 왼쪽부터 계산한다
// - 21개 점 거리는 점 0→20 순서로 하나씩 더한 뒤 21로 나눈다
// - 반전(v1)은 정규화된 벡터의 x 성분 부호를 바꾼다(이미지 좌표 1 − x 금지)
// 공통 테스트 벡터(vectors/ai-io-v1-vectors.json)로 Python 기준 구현과 일치를 확인한다.

import Foundation

/// 21개 점, 각 점은 [x, y, z].
public typealias Frame = [[Double]]
public typealias Vector = [Double]

public struct InvalidInput: Error, CustomStringConvertible {
    public let message: String
    public var description: String { "InvalidInput: \(message)" }
}

public enum Core {
    // ── §10.0 유효성 ──

    public static func checkFrame(_ frame: Frame) throws {
        guard frame.count == Spec.numLandmarks else {
            throw InvalidInput(message: "랜드마크는 \(Spec.numLandmarks)개여야 함: \(frame.count)")
        }
        for p in frame {
            guard p.count == 3 else { throw InvalidInput(message: "점은 (x, y, z) 3개 값이어야 함") }
            for c in p where !c.isFinite { throw InvalidInput(message: "NaN·Infinity 좌표가 있음") }
        }
    }

    // ── §2 정규화 ──

    static func dist3(_ a: [Double], _ b: [Double]) -> Double {
        let dx = a[0] - b[0]
        let dy = a[1] - b[1]
        let dz = a[2] - b[2]
        return (dx * dx + dy * dy + dz * dz).squareRoot()
    }

    public static func palmScale(_ frame: Frame) -> Double {
        var scale = 0.0
        let idx = Spec.palmIndices
        for i in 0..<idx.count {
            for j in (i + 1)..<idx.count {
                let d = dist3(frame[idx[i]], frame[idx[j]])
                if d > scale { scale = d }
            }
        }
        return scale < Spec.scaleEpsilon ? 1.0 : scale
    }

    public static func normalizeFrame(_ frame: Frame) throws -> Vector {
        try checkFrame(frame)
        let w = frame[0]
        let scale = palmScale(frame)
        var out = Vector(repeating: 0.0, count: Spec.features)
        for i in 0..<Spec.numLandmarks {
            let p = frame[i]
            out[i * 3] = (p[0] - w[0]) / scale
            out[i * 3 + 1] = (p[1] - w[1]) / scale
            out[i * 3 + 2] = (p[2] - w[2]) / scale
        }
        return out
    }

    // ── §4 시간축 ──

    public static func resample(_ vectors: [Vector], length: Int = Spec.sequenceLength) throws -> [Vector] {
        let n = vectors.count
        guard n > 0 else { throw InvalidInput(message: "리샘플할 프레임이 없음") }
        if n == 1 { return Array(repeating: vectors[0], count: length) }
        var out: [Vector] = []
        out.reserveCapacity(length)
        for k in 0..<length {
            let s = Double(k * (n - 1)) / Double(length - 1)
            let lo = Int(s.rounded(.down))
            let hi = Int(s.rounded(.up))
            let w = s - Double(lo)
            let a = vectors[lo]
            let b = vectors[hi]
            var v = Vector(repeating: 0.0, count: a.count)
            for i in 0..<a.count { v[i] = a[i] + (b[i] - a[i]) * w }
            out.append(v)
        }
        return out
    }

    public static func staticTensor(_ frame: Frame) throws -> [Vector] {
        let v = try normalizeFrame(frame)
        return Array(repeating: v, count: Spec.sequenceLength)
    }

    public static func dynamicTensor(_ frames: [Frame]) throws -> [Vector] {
        guard frames.count >= Spec.minDynamicFrames else {
            throw InvalidInput(message: "유효 프레임 \(frames.count)개 < \(Spec.minDynamicFrames)")
        }
        return try resample(frames.map(normalizeFrame))
    }

    // ── §10 폴백 ──

    public static func landmarkDistance(_ a: Vector, _ b: Vector) -> Double {
        var total = 0.0
        for i in 0..<Spec.numLandmarks {
            let dx = a[i * 3] - b[i * 3]
            let dy = a[i * 3 + 1] - b[i * 3 + 1]
            let dz = a[i * 3 + 2] - b[i * 3 + 2]
            total += (dx * dx + dy * dy + dz * dz).squareRoot()
        }
        return total / Double(Spec.numLandmarks)
    }

    /// v1 반전: 정규화된 벡터의 x 성분 부호를 바꾼다(§10.5).
    public static func mirrorVector(_ v: Vector) -> Vector {
        var out = v
        var i = 0
        while i < out.count {
            out[i] = -out[i]
            i += 3
        }
        return out
    }

    static func clampScore(_ x: Double) -> Double { x < 0.0 ? 0.0 : (x > 100.0 ? 100.0 : x) }

    public struct StaticResult: Equatable {
        public let score: Double
        public let distance: Double
        public let mirrored: Bool
    }

    /// §10.1 정적 거리 점수 + §10.5 반전(거리가 더 작은 쪽, 동률이면 원본).
    public static func staticScore(_ reference: Frame, _ candidate: Frame) throws -> StaticResult {
        let r = try normalizeFrame(reference)
        let c = try normalizeFrame(candidate)
        var d = landmarkDistance(r, c)
        var mirrored = false
        let dm = landmarkDistance(r, mirrorVector(c))
        if dm < d {
            d = dm
            mirrored = true
        }
        return StaticResult(score: clampScore(100.0 * Foundation.exp(-Spec.staticDecay * d)), distance: d, mirrored: mirrored)
    }

    public struct DtwResult: Equatable {
        public let score: Double
        public let avgCost: Double
        public let pathLength: Int
        public let mirrored: Bool
    }

    static func dtwCore(_ a: [Vector], _ b: [Vector]) -> (Double, Int) {
        let n = a.count, m = b.count
        var d = Array(repeating: Array(repeating: Double.infinity, count: m + 1), count: n + 1)
        d[0][0] = 0.0
        for i in 1...n {
            for j in 1...m {
                let cost = landmarkDistance(a[i - 1], b[j - 1])
                d[i][j] = cost + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])
            }
        }
        // (n, m) → (0, 0) 되짚기. 동률이면 diag → up → left
        var i = n, j = m, length = 0
        while i > 0 || j > 0 {
            length += 1
            if i == 0 {
                j -= 1
            } else if j == 0 {
                i -= 1
            } else {
                let diag = d[i - 1][j - 1], up = d[i - 1][j], left = d[i][j - 1]
                let best = min(diag, up, left)
                if best == diag {
                    i -= 1
                    j -= 1
                } else if best == up {
                    i -= 1
                } else {
                    j -= 1
                }
            }
        }
        return (d[n][m] / Double(length), length)
    }

    /// §10.2 DTW 점수 + §10.5 반전(평균 정렬 비용이 더 작은 쪽, 동률이면 원본). 리샘플·윈도우 없음.
    public static func dtwScore(_ reference: [Frame], _ candidate: [Frame]) throws -> DtwResult {
        let a = try reference.map(normalizeFrame)
        let b = try candidate.map(normalizeFrame)
        if a.isEmpty || b.isEmpty { return DtwResult(score: 0.0, avgCost: .infinity, pathLength: 0, mirrored: false) }
        var (avg, len) = dtwCore(a, b)
        var mirrored = false
        let (am, lm) = dtwCore(a, b.map(mirrorVector))
        if am < avg {
            avg = am
            len = lm
            mirrored = true
        }
        return DtwResult(score: clampScore(100.0 * Foundation.exp(-Spec.dtwDecay * avg)), avgCost: avg, pathLength: len, mirrored: mirrored)
    }

    public struct TrackInput {
        public let label: String
        public let frames: [Frame]
        public init(label: String, frames: [Frame]) {
            self.label = label
            self.frames = frames
        }
    }

    public enum TrackScore {
        case distance(label: String, result: StaticResult)
        case dtw(label: String, result: DtwResult)

        public var label: String {
            switch self { case let .distance(l, _), let .dtw(l, _): return l }
        }
        public var score: Double {
            switch self { case let .distance(_, r): return r.score; case let .dtw(_, r): return r.score }
        }
        public var mirrored: Bool {
            switch self { case let .distance(_, r): return r.mirrored; case let .dtw(_, r): return r.mirrored }
        }
    }

    public struct TrackOutcome {
        public let best: TrackScore?
        /// 프레임이 모자라 건너뛴 트랙이 있었는지(동적 N<8)
        public let sawShort: Bool
    }

    /// §3 트랙별 채점 → 최고점(동점이면 먼저 나타난 트랙). 좌표가 이상한 트랙과 프레임이 모자란 동적 트랙은 건너뛴다.
    /// 정적은 트랙의 가운데 프레임(⌊N/2⌋), 동적은 트랙 전체.
    public static func scoreTracks(_ reference: [Frame], _ tracks: [TrackInput], dynamic: Bool) -> TrackOutcome {
        var best: TrackScore?
        var sawShort = false
        for t in tracks {
            if t.frames.isEmpty { continue }
            guard (try? t.frames.forEach(checkFrame)) != nil else { continue }
            let cur: TrackScore
            if !dynamic {
                guard let r = try? staticScore(reference[0], t.frames[t.frames.count / 2]) else { continue }
                cur = .distance(label: t.label, result: r)
            } else {
                if t.frames.count < Spec.minDynamicFrames {
                    sawShort = true
                    continue
                }
                guard let r = try? dtwScore(reference, t.frames) else { continue }
                cur = .dtw(label: t.label, result: r)
            }
            if best == nil || cur.score > best!.score { best = cur }
        }
        return TrackOutcome(best: best, sawShort: sawShort)
    }

    /// §10.3 녹화 길이. 반올림은 half away from zero(floor(x + 0.5)를 쓰지 않음).
    public static func recordingDurationMs(firstTMs: Double?, lastTMs: Double?) -> Int {
        guard let f = firstTMs, let l = lastTMs, l > f else { return Spec.defaultRecordingMs }
        let raw = (l - f) * Spec.recordingFactor
        let whole = raw.rounded(.down)
        let rounded = Int(whole) + (raw - whole >= 0.5 ? 1 : 0)
        return max(Spec.minRecordingMs, min(Spec.maxRecordingMs, rounded))
    }
}
