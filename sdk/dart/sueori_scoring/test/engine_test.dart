// ScoringEngine(공개 API) 동작 — 상태 코드, 트랙 선택, 버전 정보.
import 'dart:convert';
import 'dart:io';

import 'package:sueori_scoring/sueori_scoring.dart';
import 'package:test/test.dart';

final Map<String, dynamic> v =
    jsonDecode(File('../../../vectors/ai-io-v1-vectors.json').readAsStringSync()) as Map<String, dynamic>;
List<List<double>> frame(Object? f) => [for (final p in f as List) [for (final c in p as List) (c as num).toDouble()]];
List<List<List<double>>> frames(Object? fs) => [for (final f in fs as List) frame(f)];
Map<String, dynamic> dtwCase(String n) => (v['dtw'] as List).cast<Map<String, dynamic>>().firstWhere((c) => c['name'] == n);
Map<String, dynamic> staticCase(String n) =>
    (v['static_score'] as List).cast<Map<String, dynamic>>().firstWhere((c) => c['name'] == n);
List<CapturedFrame> capture(List<List<List<double>>> fs, [String hand = 'Right']) => [
      for (var k = 0; k < fs.length; k++)
        CapturedFrame(tMs: 33 * k, hands: [CapturedHand(handedness: hand, landmarks: fs[k])])
    ];
WordReference dyn(Map<String, dynamic> c) =>
    WordReference(wordKey: 'w', type: WordType.dynamicSign, referenceFrames: frames(c['reference']));

void main() {
  final engine = ScoringEngine();

  test('동적 정답: 벡터와 같은 점수, method=dtw, 버전 정보', () {
    final c = dtwCase('same_motion_slower_noisy');
    final r = engine.score(ScoreRequest(word: dyn(c), capture: capture(frames(c['candidate']))));
    final e = c['expected'] as Map<String, dynamic>;
    expect(r.status, ScoreStatus.ok);
    expect((r.score! - (e['score'] as num)).abs(), lessThanOrEqualTo(1e-9));
    expect(r.method, ScoreMethod.dtw);
    expect(r.passed, r.score! >= 80);
    expect(r.track, 'Right');
    expect(r.fallbackReason, FallbackReason.modelNotConfigured);
    final j = r.toJson();
    expect(j['status'], 'ok');
    expect(j['specVersion'], specVersion);
    expect(j['sdkVersion'], sdkVersion);
    expect(j['fallbackReason'], 'model_not_configured');
    expect((j['details'] as Map)['pathLength'], e['path_length']);
  });

  test('왼손(반전) 입력은 v1에서 반전으로 채점된다', () {
    final c = dtwCase('v1_mirror_wins_left_hand');
    final r = engine.score(ScoreRequest(word: dyn(c), capture: capture(frames(c['candidate']), 'Left')));
    expect(r.mirrored, isTrue);
    expect((r.score! - ((c['expected'] as Map)['score'] as num)).abs(), lessThanOrEqualTo(1e-9));
  });

  test('정적: 트랙 가운데 프레임으로 채점, method=distance', () {
    final c = staticCase('same_pose_noise');
    final other = frame(staticCase('open_vs_fist')['candidate']);
    final r = engine.score(ScoreRequest(
      word: WordReference(wordKey: 's', type: WordType.staticSign, referenceFrames: [frame(c['reference'])]),
      capture: capture([other, frame(c['candidate']), other]),
    ));
    expect(r.method, ScoreMethod.distance);
    expect((r.score! - ((c['expected'] as Map)['score'] as num)).abs(), lessThanOrEqualTo(1e-9));
  });

  test('no_hand / insufficient_frames / invalid', () {
    final c = dtwCase('identical');
    final cand = frames(c['candidate']);
    expect(engine.score(ScoreRequest(word: dyn(c), capture: [const CapturedFrame(tMs: 0, hands: [])])).status,
        ScoreStatus.noHand);
    expect(engine.score(ScoreRequest(word: dyn(c), capture: capture(cand.sublist(0, 7)))).status,
        ScoreStatus.insufficientFrames);
    final bad = [for (final f in cand) [for (var i = 0; i < f.length; i++) i == 3 ? [double.nan, f[i][1], f[i][2]] : f[i]]];
    expect(engine.score(ScoreRequest(word: dyn(c), capture: capture(bad))).status, ScoreStatus.invalid);
    final mixed = [
      for (var k = 0; k < cand.length; k++)
        CapturedFrame(tMs: 33 * k, hands: [
          CapturedHand(handedness: 'Left', landmarks: bad[k]),
          CapturedHand(handedness: 'Right', landmarks: cand[k]),
        ])
    ];
    final r = engine.score(ScoreRequest(word: dyn(c), capture: mixed));
    expect(r.track, 'Right');
    expect(r.score, 100.0);
  });

  test('기준 데이터가 없으면 InvalidInput', () {
    expect(
        () => engine.score(ScoreRequest(
            word: const WordReference(wordKey: 'w', type: WordType.dynamicSign, referenceFrames: []), capture: const [])),
        throwsA(isA<InvalidInput>()));
  });

  test('녹화 길이', () {
    expect(engine.recordingDurationMs(const WordReference(wordKey: 'w', type: WordType.dynamicSign, referenceFrames: [])),
        3000);
    expect(
        engine.recordingDurationMs(
            const WordReference(wordKey: 'w', type: WordType.dynamicSign, referenceFrames: [], referenceTMs: [33, 500, 1867])),
        2201);
  });
}
