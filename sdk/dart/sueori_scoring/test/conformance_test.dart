// 공통 테스트 벡터(vectors/ai-io-v1-vectors.json) 적합성 — Python 기준 구현과 같은 결과인지 확인한다.
import 'dart:convert';
import 'dart:io';

import 'package:sueori_scoring/src/core.dart';
import 'package:sueori_scoring/src/spec.dart' as s;
import 'package:test/test.dart';

final Map<String, dynamic> v =
    jsonDecode(File('../../../vectors/ai-io-v1-vectors.json').readAsStringSync()) as Map<String, dynamic>;
final Map<String, dynamic> tol = v['tolerances'] as Map<String, dynamic>;

double _num(Object? x) => x is String
    ? (x == 'NaN' ? double.nan : (x == 'Infinity' ? double.infinity : double.negativeInfinity))
    : (x as num).toDouble();
Frame frame(Object? f) => [for (final p in f as List) [for (final c in p as List) _num(c)]];
List<Frame> frames(Object? fs) => [for (final f in fs as List) frame(f)];
double t(String k) => (tol[k] as num).toDouble();

double maxErr(Object a, Object b) {
  if (a is List) {
    expect(a.length, (b as List).length);
    var m = 0.0;
    for (var i = 0; i < a.length; i++) {
      final e = maxErr(a[i] as Object, b[i] as Object);
      if (e > m) m = e;
    }
    return m;
  }
  return ((a as num) - (b as num)).abs().toDouble();
}

void main() {
  test('spec_version', () => expect(v['spec_version'], s.specVersion));

  for (final c in (v['normalize'] as List).cast<Map<String, dynamic>>()) {
    test('normalize ${c['name']}', () {
      final f = frame(c['frame']);
      if (c['invalid'] == true) return expect(() => normalizeFrame(f), throwsA(isA<InvalidInput>()));
      expect(maxErr(normalizeFrame(f), c['expected'] as Object), lessThanOrEqualTo(t('normalize_f64')));
    });
  }

  for (final c in (v['tensor'] as List).cast<Map<String, dynamic>>()) {
    test('tensor ${c['name']}', () {
      final fs = frames(c['frames']);
      List<Vector> fn() => c['kind'] == 'static' ? staticTensor(fs[0]) : dynamicTensor(fs);
      if (c['invalid'] == true) return expect(fn, throwsA(isA<InvalidInput>()));
      expect(maxErr(fn(), c['expected'] as Object), lessThanOrEqualTo(t('tensor_f64')));
    });
  }

  for (final c in (v['static_score'] as List).cast<Map<String, dynamic>>()) {
    test('static_score ${c['name']}', () {
      final r = frame(c['reference']);
      final k = frame(c['candidate']);
      if (c['invalid'] == true) return expect(() => staticScore(r, k), throwsA(isA<InvalidInput>()));
      final got = staticScore(r, k);
      final e = c['expected'] as Map<String, dynamic>;
      expect((got.distance - _num(e['distance'])).abs(), lessThanOrEqualTo(t('distance')));
      expect((got.score - _num(e['score'])).abs(), lessThanOrEqualTo(t('score')));
      expect(got.mirrored, e['mirrored']);
    });
  }

  for (final c in (v['dtw'] as List).cast<Map<String, dynamic>>()) {
    test('dtw ${c['name']}', () {
      final r = frames(c['reference']);
      final k = frames(c['candidate']);
      if (c['invalid'] == true) {
        // §4 유효성(N<8)은 트랙 단위 채점에서 걸러진다
        final out = scoreTracks(r, [TrackInput('', k)], dynamic: true);
        expect(out.best, isNull);
        expect(out.sawShort, isTrue);
        return;
      }
      final got = dtwScore(r, k);
      final e = c['expected'] as Map<String, dynamic>;
      expect(got.pathLength, e['path_length']);
      expect((got.avgCost - _num(e['avg_cost'])).abs(), lessThanOrEqualTo(t('avg_cost')));
      expect((got.score - _num(e['score'])).abs(), lessThanOrEqualTo(t('score')));
      expect(got.mirrored, e['mirrored']);
    });
  }

  for (final c in (v['best_track'] as List).cast<Map<String, dynamic>>()) {
    test('best_track ${c['name']}', () {
      final tracks = [
        for (final tr in (c['tracks'] as List).cast<Map<String, dynamic>>())
          TrackInput(tr['label'] as String, frames(tr['frames']))
      ];
      final out = scoreTracks(frames(c['reference']), tracks, dynamic: c['kind'] == 'dynamic');
      final e = c['expected'] as Map<String, dynamic>?;
      if (e == null) return expect(out.best, isNull);
      expect(out.best!.label, e['label']);
      expect((out.best!.score - _num(e['score'])).abs(), lessThanOrEqualTo(t('score')));
    });
  }

  for (final c in (v['recording_duration'] as List).cast<Map<String, dynamic>>()) {
    test('recording_duration ${c['first_t_ms']}→${c['last_t_ms']}', () {
      expect(recordingDurationMs(c['first_t_ms'] as num?, c['last_t_ms'] as num?), c['expected']);
    });
  }
}
