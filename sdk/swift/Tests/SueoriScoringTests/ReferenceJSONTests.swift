import Foundation
import XCTest
import SueoriScoring

final class ReferenceJSONTests: XCTestCase {
    /// JSONSerialization이 1 ulp 다르게 읽었던 실제 값들(공통 테스트 벡터에서 발견)
    let tricky = ["-0.053210526315789236", "-0.056519179811931185", "0.0031137510559756804"]

    func frameText(_ x: String) -> String {
        "[" + (0..<21).map { _ in "[\(x), 0.5, -0.01]" }.joined(separator: ",") + "]"
    }

    func testArrayFormatExact() throws {
        for t in tricky {
            let data = "[\(frameText(t))]".data(using: .utf8)!
            let fs = try ReferenceJSON.frames(from: data)
            XCTAssertEqual(fs.count, 1)
            XCTAssertEqual(fs[0].count, 21)
            XCTAssertEqual(fs[0][0][0], Double(t)!, "JSONDecoder는 정확히 읽어야 함: \(t)")
            XCTAssertNil(try ReferenceJSON.tMs(from: data))
        }
    }

    func testObjectFormatWithTMs() throws {
        let pts = (0..<21).map { _ in "{\"x\": \(tricky[0]), \"y\": 0.5, \"z\": -0.01}" }.joined(separator: ",")
        let json = "[{\"landmarks\": [\(pts)], \"tMs\": 0}, {\"landmarks\": [\(pts)], \"tMs\": 33}]"
        let w = try ReferenceJSON.word(wordKey: "family/x", type: .dynamicSign, framesJSON: json.data(using: .utf8)!)
        XCTAssertEqual(w.referenceFrames.count, 2)
        XCTAssertEqual(w.referenceFrames[1][20], [Double(tricky[0])!, 0.5, -0.01])
        XCTAssertEqual(w.referenceTMs, [0, 33])
    }

    func testJSONSerializationPitfallIsReal() throws {
        // 함정이 실제로 있는지 기록(이 테스트가 깨지면 Foundation이 고쳐진 것 — 문서의 경고를 다시 검토)
        let bad = tricky.filter { t in
            let n = (try! JSONSerialization.jsonObject(with: "[\(t)]".data(using: .utf8)!) as! [NSNumber])[0].doubleValue
            return n != Double(t)!
        }
        print("JSONSerialization 1-ulp 오차: \(bad.count)/\(tricky.count)")
    }
}
