// 공통 테스트 벡터(vectors/ai-io-v1-vectors.json) 적합성 — Python 기준 구현과 같은 결과인지 확인한다.
import Foundation
import XCTest
@testable import SueoriScoring

/// JSON 값. 벡터는 **JSONDecoder**로 읽는다 — Foundation `JSONSerialization`은 일부 실수를 마지막 비트 하나
/// 다르게 읽는다(실측: 72,468개 중 285개). JSONDecoder는 전부 정확했다. 앱도 기준 데이터를 JSONDecoder로 읽을 것.
enum JSONValue: Decodable {
    case number(Double), string(String), bool(Bool), null, array([JSONValue]), object([String: JSONValue])

    init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if c.decodeNil() { self = .null }
        else if let b = try? c.decode(Bool.self) { self = .bool(b) }
        else if let d = try? c.decode(Double.self) { self = .number(d) }
        else if let s = try? c.decode(String.self) { self = .string(s) }
        else if let a = try? c.decode([JSONValue].self) { self = .array(a) }
        else { self = .object(try c.decode([String: JSONValue].self)) }
    }

    var any: Any {
        switch self {
        case let .number(d): return d
        case let .string(s): return s
        case let .bool(b): return b
        case .null: return NSNull()
        case let .array(a): return a.map(\.any)
        case let .object(o): return o.mapValues(\.any)
        }
    }
}

enum Vectors {
    static let data: [String: Any] = {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("vectors/ai-io-v1-vectors.json")
        let d = try! Data(contentsOf: url)
        return try! JSONDecoder().decode(JSONValue.self, from: d).any as! [String: Any]
    }()

    static func cases(_ section: String) -> [[String: Any]] { data[section] as! [[String: Any]] }
    static func tol(_ k: String) -> Double { num((data["tolerances"] as! [String: Any])[k]!) }

    static func num(_ x: Any) -> Double {
        if let s = x as? String { return s == "NaN" ? .nan : (s == "Infinity" ? .infinity : -.infinity) }
        return x as! Double
    }
    static func optNum(_ x: Any?) -> Double? { x is NSNull || x == nil ? nil : num(x!) }
    static func frame(_ f: Any) -> Frame { (f as! [Any]).map { ($0 as! [Any]).map(num) } }
    static func frames(_ fs: Any) -> [Frame] { (fs as! [Any]).map(frame) }
}

func maxErr(_ a: [Double], _ b: [Any]) -> Double {
    XCTAssertEqual(a.count, b.count)
    return zip(a, b).map { abs($0 - Vectors.num($1)) }.max() ?? 0
}

final class ConformanceTests: XCTestCase {
    func testSpecVersion() {
        XCTAssertEqual(Vectors.data["spec_version"] as? String, Spec.specVersion)
    }

    func testNormalize() throws {
        for c in Vectors.cases("normalize") {
            let f = Vectors.frame(c["frame"]!)
            if c["invalid"] as? Bool == true {
                XCTAssertThrowsError(try Core.normalizeFrame(f), "\(c["name"]!)")
                continue
            }
            XCTAssertLessThanOrEqual(maxErr(try Core.normalizeFrame(f), c["expected"] as! [Any]), Vectors.tol("normalize_f64"))
        }
    }

    func testTensor() throws {
        for c in Vectors.cases("tensor") {
            let fs = Vectors.frames(c["frames"]!)
            let fn = { c["kind"] as! String == "static" ? try Core.staticTensor(fs[0]) : try Core.dynamicTensor(fs) }
            if c["invalid"] as? Bool == true {
                XCTAssertThrowsError(try fn(), "\(c["name"]!)")
                continue
            }
            let got = try fn()
            let exp = c["expected"] as! [Any]
            XCTAssertEqual(got.count, exp.count)
            for (row, e) in zip(got, exp) {
                XCTAssertLessThanOrEqual(maxErr(row, e as! [Any]), Vectors.tol("tensor_f64"), "\(c["name"]!)")
            }
        }
    }

    func testStaticScore() throws {
        for c in Vectors.cases("static_score") {
            let r = Vectors.frame(c["reference"]!), k = Vectors.frame(c["candidate"]!)
            if c["invalid"] as? Bool == true {
                XCTAssertThrowsError(try Core.staticScore(r, k))
                continue
            }
            let got = try Core.staticScore(r, k)
            let e = c["expected"] as! [String: Any]
            XCTAssertLessThanOrEqual(abs(got.distance - Vectors.num(e["distance"]!)), Vectors.tol("distance"), "\(c["name"]!)")
            XCTAssertLessThanOrEqual(abs(got.score - Vectors.num(e["score"]!)), Vectors.tol("score"), "\(c["name"]!)")
            XCTAssertEqual(got.mirrored, e["mirrored"] as? Bool, "\(c["name"]!)")
        }
    }

    func testDtw() throws {
        for c in Vectors.cases("dtw") {
            let r = Vectors.frames(c["reference"]!), k = Vectors.frames(c["candidate"]!)
            if c["invalid"] as? Bool == true {
                // §4 유효성(N<8)은 트랙 단위 채점에서 걸러진다
                let out = Core.scoreTracks(r, [Core.TrackInput(label: "", frames: k)], dynamic: true)
                XCTAssertNil(out.best)
                XCTAssertTrue(out.sawShort)
                continue
            }
            let got = try Core.dtwScore(r, k)
            let e = c["expected"] as! [String: Any]
            XCTAssertEqual(got.pathLength, Int(Vectors.num(e["path_length"]!)), "\(c["name"]!)")
            XCTAssertLessThanOrEqual(abs(got.avgCost - Vectors.num(e["avg_cost"]!)), Vectors.tol("avg_cost"), "\(c["name"]!)")
            XCTAssertLessThanOrEqual(abs(got.score - Vectors.num(e["score"]!)), Vectors.tol("score"), "\(c["name"]!)")
            XCTAssertEqual(got.mirrored, e["mirrored"] as? Bool, "\(c["name"]!)")
        }
    }

    func testBestTrack() {
        for c in Vectors.cases("best_track") {
            let tracks = (c["tracks"] as! [[String: Any]]).map {
                Core.TrackInput(label: $0["label"] as! String, frames: Vectors.frames($0["frames"]!))
            }
            let out = Core.scoreTracks(Vectors.frames(c["reference"]!), tracks, dynamic: c["kind"] as! String == "dynamic")
            guard let e = c["expected"] as? [String: Any] else {  // null → NSNull
                XCTAssertNil(out.best, "\(c["name"]!)")
                continue
            }
            XCTAssertEqual(out.best?.label, e["label"] as? String)
            XCTAssertLessThanOrEqual(abs(out.best!.score - Vectors.num(e["score"]!)), Vectors.tol("score"))
        }
    }

    func testRecordingDuration() {
        for c in Vectors.cases("recording_duration") {
            let f = Vectors.optNum(c["first_t_ms"])
            let l = Vectors.optNum(c["last_t_ms"])
            XCTAssertEqual(Core.recordingDurationMs(firstTMs: f, lastTMs: l), Int(Vectors.num(c["expected"]!)))
        }
    }

    /// 비트 단위 재현(§10.0) — 허용 오차가 아니라 정확히 같은지. 빌드 플래그(fast-math 등) 회귀를 잡는다
    func testBitExactAgainstPythonReference() throws {
        var exact = 0, n = 0
        for c in Vectors.cases("dtw") where c["invalid"] as? Bool != true {
            let g = try Core.dtwScore(Vectors.frames(c["reference"]!), Vectors.frames(c["candidate"]!))
            let e = c["expected"] as! [String: Any]
            n += 1
            if g.score == Vectors.num(e["score"]!) && g.avgCost == Vectors.num(e["avg_cost"]!) { exact += 1 }
        }
        for c in Vectors.cases("static_score") where c["invalid"] as? Bool != true {
            let g = try Core.staticScore(Vectors.frame(c["reference"]!), Vectors.frame(c["candidate"]!))
            let e = c["expected"] as! [String: Any]
            n += 1
            if g.score == Vectors.num(e["score"]!) && g.distance == Vectors.num(e["distance"]!) { exact += 1 }
        }
        print("SWIFT bit-exact \(exact)/\(n)")
        XCTAssertEqual(exact, n)
    }
}
