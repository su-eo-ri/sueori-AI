// swift-tools-version:5.9
//
// 수어리 채점 SDK (Swift) — 소스는 sdk/swift. SwiftPM이 레포 루트의 semver 태그(v0.1.0)만 버전으로 인식하므로
// Package.swift를 루트에 둔다(CTO 2026-10-18). 세 SDK(TS·Dart·Swift)는 같은 버전으로 함께 올린다.
//
// 결정성(스펙 §10.0): Float64, 순차 합산, FMA 금지. 이 패키지는 -Ounchecked·fast-math 계열 플래그를 쓰지 않는다.
// (SwiftPM은 원격 의존성의 unsafeFlags를 허용하지 않아 플래그로 "금지"를 걸 수는 없다. 대신 Swift는 기본으로
//  부동소수점 연산을 합치지 않으며, 공통 테스트 벡터 비트 단위 테스트가 회귀를 잡는다.)
import PackageDescription

let package = Package(
    name: "SueoriScoring",
    platforms: [.iOS(.v15), .macOS(.v12)],
    products: [
        .library(name: "SueoriScoring", targets: ["SueoriScoring"]),
    ],
    targets: [
        .target(name: "SueoriScoring", path: "sdk/swift/Sources/SueoriScoring"),
        .testTarget(
            name: "SueoriScoringTests",
            dependencies: ["SueoriScoring"],
            path: "sdk/swift/Tests/SueoriScoringTests"
        ),
    ]
)
