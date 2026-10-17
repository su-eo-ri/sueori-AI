// ScoringEngine(공개 API) 동작 — 상태 코드, 트랙 선택, 버전 정보.
import XCTest
import SueoriScoring

final class EngineTests: XCTestCase {
    let engine = ScoringEngine()

    func dtwCase(_ n: String) -> [String: Any] { Vectors.cases("dtw").first { $0["name"] as? String == n }! }
    func staticCase(_ n: String) -> [String: Any] { Vectors.cases("static_score").first { $0["name"] as? String == n }! }
    func capture(_ fs: [Frame], _ hand: String = "Right") -> [CapturedFrame] {
        fs.enumerated().map { CapturedFrame(tMs: Double(33 * $0.offset), hands: [CapturedHand(handedness: hand, landmarks: $0.element)]) }
    }
    func dyn(_ c: [String: Any]) -> WordReference {
        WordReference(wordKey: "w", type: .dynamicSign, referenceFrames: Vectors.frames(c["reference"]!))
    }

    func testDynamicCorrect() throws {
        let c = dtwCase("same_motion_slower_noisy")
        let r = try engine.score(ScoreRequest(word: dyn(c), capture: capture(Vectors.frames(c["candidate"]!))))
        let e = c["expected"] as! [String: Any]
        XCTAssertEqual(r.status, .ok)
        XCTAssertLessThanOrEqual(abs(r.score! - Vectors.num(e["score"]!)), 1e-9)
        XCTAssertEqual(r.method, .dtw)
        XCTAssertEqual(r.passed, r.score! >= 80)
        XCTAssertEqual(r.track, "Right")
        XCTAssertEqual(r.fallbackReason, .modelNotConfigured)
        XCTAssertEqual(r.specVersion, Spec.specVersion)
        XCTAssertEqual(r.sdkVersion, Spec.sdkVersion)
        XCTAssertEqual(r.details["pathLength"], Vectors.num(e["path_length"]!))
        XCTAssertEqual(r.toDictionary()["status"] as? String, "ok")
    }

    func testLeftHandMirrored() throws {
        let c = dtwCase("v1_mirror_wins_left_hand")
        let r = try engine.score(ScoreRequest(word: dyn(c), capture: capture(Vectors.frames(c["candidate"]!), "Left")))
        XCTAssertTrue(r.mirrored)
        XCTAssertLessThanOrEqual(abs(r.score! - Vectors.num((c["expected"] as! [String: Any])["score"]!)), 1e-9)
    }

    func testStaticUsesMiddleFrame() throws {
        let c = staticCase("same_pose_noise")
        let other = Vectors.frame(staticCase("open_vs_fist")["candidate"]!)
        let r = try engine.score(ScoreRequest(
            word: WordReference(wordKey: "s", type: .staticSign, referenceFrames: [Vectors.frame(c["reference"]!)]),
            capture: capture([other, Vectors.frame(c["candidate"]!), other])))
        XCTAssertEqual(r.method, .distance)
        XCTAssertLessThanOrEqual(abs(r.score! - Vectors.num((c["expected"] as! [String: Any])["score"]!)), 1e-9)
    }

    func testStatuses() throws {
        let c = dtwCase("identical")
        let cand = Vectors.frames(c["candidate"]!)
        XCTAssertEqual(try engine.score(ScoreRequest(word: dyn(c), capture: [CapturedFrame(tMs: 0, hands: [])])).status, .noHand)
        XCTAssertEqual(try engine.score(ScoreRequest(word: dyn(c), capture: capture(Array(cand.prefix(7))))).status, .insufficientFrames)
        let bad = cand.map { f in f.enumerated().map { $0.offset == 3 ? [Double.nan, $0.element[1], $0.element[2]] : $0.element } }
        XCTAssertEqual(try engine.score(ScoreRequest(word: dyn(c), capture: capture(bad))).status, .invalid)
        let mixed = cand.indices.map {
            CapturedFrame(tMs: Double(33 * $0), hands: [CapturedHand(handedness: "Left", landmarks: bad[$0]),
                                                        CapturedHand(handedness: "Right", landmarks: cand[$0])])
        }
        let r = try engine.score(ScoreRequest(word: dyn(c), capture: mixed))
        XCTAssertEqual(r.track, "Right")
        XCTAssertEqual(r.score, 100)
    }

    func testInvalidReferenceThrows() {
        XCTAssertThrowsError(try engine.score(ScoreRequest(
            word: WordReference(wordKey: "w", type: .dynamicSign, referenceFrames: []), capture: [])))
    }

    func testFrameSizeIgnoredIn01() throws {
        let c = dtwCase("same_motion_slower_noisy")
        let cap = capture(Vectors.frames(c["candidate"]!))
        let a = try engine.score(ScoreRequest(word: dyn(c), capture: cap))
        let b = try engine.score(ScoreRequest(word: dyn(c), capture: cap,
                                              frameSize: FrameSize(width: 720, height: 1280), inputMirrored: true))
        XCTAssertEqual(a, b)
    }

    func testRecordingDuration() {
        XCTAssertEqual(engine.recordingDurationMs(WordReference(wordKey: "w", type: .dynamicSign, referenceFrames: [])), 3000)
        XCTAssertEqual(engine.recordingDurationMs(
            WordReference(wordKey: "w", type: .dynamicSign, referenceFrames: [], referenceTMs: [33, 500, 1867])), 2201)
    }
}
