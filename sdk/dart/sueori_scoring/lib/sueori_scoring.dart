/// sueori_scoring — 수어리 채점 SDK (온디바이스).
///
/// ```dart
/// final engine = ScoringEngine();
/// final result = engine.score(ScoreRequest(word: word, capture: capture));
/// ```
///
/// `core`는 스펙 구현(공통 테스트 벡터 검증용)이며 앱에서 직접 쓸 필요는 없다.
library sueori_scoring;

export 'src/engine.dart';
export 'src/core.dart' show InvalidInput;
export 'src/spec.dart' show specVersion, sdkVersion, passThreshold;
