/// 채점 코어 — 스펙 ai-io-v1 §2·§3·§4·§10. 외부 의존성 없음.
///
/// 비트 단위 재현 규칙(§10.0)
/// - 모든 계산은 Float64(Dart double)
/// - 제곱합은 `dx * dx + dy * dy + dz * dz`를 왼쪽부터 계산한다
/// - 21개 점 거리는 점 0→20 순서로 하나씩 더한 뒤 21로 나눈다
/// - 반전(v1)은 정규화된 벡터의 x 성분 부호를 바꾼다(이미지 좌표 1 − x 금지)
/// 공통 테스트 벡터(vectors/ai-io-v1-vectors.json)로 Python 기준 구현과 일치를 확인한다.
library;

import 'dart:math' as math;

import 'spec.dart' as s;

/// 21개 점, 각 점은 [x, y, z].
typedef Frame = List<List<double>>;
typedef Vector = List<double>;

class InvalidInput implements Exception {
  InvalidInput(this.message);
  final String message;
  @override
  String toString() => 'InvalidInput: $message';
}

// ── §10.0 유효성 ──

void checkFrame(Frame frame) {
  if (frame.length != s.numLandmarks) {
    throw InvalidInput('랜드마크는 ${s.numLandmarks}개여야 함: ${frame.length}');
  }
  for (final p in frame) {
    if (p.length != 3) throw InvalidInput('점은 (x, y, z) 3개 값이어야 함');
    for (final c in p) {
      if (!c.isFinite) throw InvalidInput('NaN·Infinity 좌표가 있음');
    }
  }
}

// ── §2 정규화 ──

double _dist3(List<double> a, List<double> b) {
  final dx = a[0] - b[0];
  final dy = a[1] - b[1];
  final dz = a[2] - b[2];
  return math.sqrt(dx * dx + dy * dy + dz * dz);
}

double palmScale(Frame frame) {
  var scale = 0.0;
  const idx = s.palmIndices;
  for (var i = 0; i < idx.length; i++) {
    for (var j = i + 1; j < idx.length; j++) {
      final d = _dist3(frame[idx[i]], frame[idx[j]]);
      if (d > scale) scale = d;
    }
  }
  return scale < s.scaleEpsilon ? 1.0 : scale;
}

Vector normalizeFrame(Frame frame) {
  checkFrame(frame);
  final w = frame[0];
  final scale = palmScale(frame);
  final out = List<double>.filled(s.features, 0.0);
  for (var i = 0; i < s.numLandmarks; i++) {
    final p = frame[i];
    out[i * 3] = (p[0] - w[0]) / scale;
    out[i * 3 + 1] = (p[1] - w[1]) / scale;
    out[i * 3 + 2] = (p[2] - w[2]) / scale;
  }
  return out;
}

// ── §4 시간축 ──

List<Vector> resample(List<Vector> vectors, [int length = s.sequenceLength]) {
  final n = vectors.length;
  if (n == 0) throw InvalidInput('리샘플할 프레임이 없음');
  if (n == 1) return List.generate(length, (_) => List<double>.of(vectors[0]));
  final out = <Vector>[];
  for (var k = 0; k < length; k++) {
    final pos = (k * (n - 1)) / (length - 1);
    final lo = pos.floor();
    final hi = pos.ceil();
    final w = pos - lo;
    final a = vectors[lo];
    final b = vectors[hi];
    out.add(List<double>.generate(a.length, (i) => a[i] + (b[i] - a[i]) * w));
  }
  return out;
}

List<Vector> staticTensor(Frame frame) {
  final v = normalizeFrame(frame);
  return List.generate(s.sequenceLength, (_) => List<double>.of(v));
}

List<Vector> dynamicTensor(List<Frame> frames) {
  if (frames.length < s.minDynamicFrames) {
    throw InvalidInput('유효 프레임 ${frames.length}개 < ${s.minDynamicFrames}');
  }
  return resample(frames.map(normalizeFrame).toList());
}

// ── §10 폴백 ──

double landmarkDistance(Vector a, Vector b) {
  var total = 0.0;
  for (var i = 0; i < s.numLandmarks; i++) {
    final dx = a[i * 3] - b[i * 3];
    final dy = a[i * 3 + 1] - b[i * 3 + 1];
    final dz = a[i * 3 + 2] - b[i * 3 + 2];
    total += math.sqrt(dx * dx + dy * dy + dz * dz);
  }
  return total / s.numLandmarks;
}

/// v1 반전: 정규화된 벡터의 x 성분 부호를 바꾼다(§10.5).
Vector mirrorVector(Vector v) {
  final out = List<double>.of(v);
  for (var i = 0; i < out.length; i += 3) {
    out[i] = -out[i];
  }
  return out;
}

double _clampScore(double x) => x < 0.0 ? 0.0 : (x > 100.0 ? 100.0 : x);

class StaticResult {
  const StaticResult(this.score, this.distance, this.mirrored);
  final double score;
  final double distance;
  final bool mirrored;
}

/// §10.1 정적 거리 점수 + §10.5 반전(거리가 더 작은 쪽, 동률이면 원본).
StaticResult staticScore(Frame reference, Frame candidate) {
  final r = normalizeFrame(reference);
  final c = normalizeFrame(candidate);
  var d = landmarkDistance(r, c);
  var mirrored = false;
  final dm = landmarkDistance(r, mirrorVector(c));
  if (dm < d) {
    d = dm;
    mirrored = true;
  }
  return StaticResult(_clampScore(100.0 * math.exp(-s.staticDecay * d)), d, mirrored);
}

class DtwResult {
  const DtwResult(this.score, this.avgCost, this.pathLength, this.mirrored);
  final double score;
  final double avgCost;
  final int pathLength;
  final bool mirrored;
}

(double, int) _dtwCore(List<Vector> a, List<Vector> b) {
  final n = a.length;
  final m = b.length;
  final d = List.generate(n + 1, (_) => List<double>.filled(m + 1, double.infinity));
  d[0][0] = 0.0;
  for (var i = 1; i <= n; i++) {
    for (var j = 1; j <= m; j++) {
      final cost = landmarkDistance(a[i - 1], b[j - 1]);
      d[i][j] = cost + math.min(math.min(d[i - 1][j], d[i][j - 1]), d[i - 1][j - 1]);
    }
  }
  // (n, m) → (0, 0) 되짚기. 동률이면 diag → up → left
  var i = n, j = m, length = 0;
  while (i > 0 || j > 0) {
    length++;
    if (i == 0) {
      j--;
    } else if (j == 0) {
      i--;
    } else {
      final diag = d[i - 1][j - 1];
      final up = d[i - 1][j];
      final left = d[i][j - 1];
      final best = math.min(math.min(diag, up), left);
      if (best == diag) {
        i--;
        j--;
      } else if (best == up) {
        i--;
      } else {
        j--;
      }
    }
  }
  return (d[n][m] / length, length);
}

/// §10.2 DTW 점수 + §10.5 반전(평균 정렬 비용이 더 작은 쪽, 동률이면 원본). 리샘플·윈도우 없음.
DtwResult dtwScore(List<Frame> reference, List<Frame> candidate) {
  final a = reference.map(normalizeFrame).toList();
  final b = candidate.map(normalizeFrame).toList();
  if (a.isEmpty || b.isEmpty) return const DtwResult(0.0, double.infinity, 0, false);
  var (avg, len) = _dtwCore(a, b);
  var mirrored = false;
  final (am, lm) = _dtwCore(a, b.map(mirrorVector).toList());
  if (am < avg) {
    avg = am;
    len = lm;
    mirrored = true;
  }
  return DtwResult(_clampScore(100.0 * math.exp(-s.dtwDecay * avg)), avg, len, mirrored);
}

class TrackInput {
  const TrackInput(this.label, this.frames);
  final String label;
  final List<Frame> frames;
}

class TrackScore {
  const TrackScore.distance(this.label, StaticResult r)
      : staticResult = r,
        dtwResult = null;
  const TrackScore.dtw(this.label, DtwResult r)
      : dtwResult = r,
        staticResult = null;
  final String label;
  final StaticResult? staticResult;
  final DtwResult? dtwResult;
  double get score => staticResult?.score ?? dtwResult!.score;
  bool get mirrored => staticResult?.mirrored ?? dtwResult!.mirrored;
}

class TrackOutcome {
  const TrackOutcome(this.best, this.sawShort);
  final TrackScore? best;

  /// 프레임이 모자라 건너뛴 트랙이 있었는지(동적 N<8)
  final bool sawShort;
}

/// §3 트랙별 채점 → 최고점(동점이면 먼저 나타난 트랙). 좌표가 이상한 트랙과 프레임이 모자란 동적 트랙은 건너뛴다.
/// 정적은 트랙의 가운데 프레임(⌊N/2⌋), 동적은 트랙 전체.
TrackOutcome scoreTracks(List<Frame> reference, List<TrackInput> tracks, {required bool dynamic}) {
  TrackScore? best;
  var sawShort = false;
  for (final t in tracks) {
    if (t.frames.isEmpty) continue;
    try {
      t.frames.forEach(checkFrame);
    } on InvalidInput {
      continue;
    }
    final TrackScore cur;
    if (!dynamic) {
      cur = TrackScore.distance(t.label, staticScore(reference[0], t.frames[t.frames.length ~/ 2]));
    } else {
      if (t.frames.length < s.minDynamicFrames) {
        sawShort = true;
        continue;
      }
      cur = TrackScore.dtw(t.label, dtwScore(reference, t.frames));
    }
    if (best == null || cur.score > best.score) best = cur;
  }
  return TrackOutcome(best, sawShort);
}

/// §10.3 녹화 길이. 반올림은 half away from zero(floor(x + 0.5)를 쓰지 않음).
int recordingDurationMs(num? firstTMs, num? lastTMs) {
  if (firstTMs == null || lastTMs == null || !(lastTMs > firstTMs)) return s.defaultRecordingMs;
  final raw = (lastTMs - firstTMs) * s.recordingFactor;
  final whole = raw.floor();
  final rounded = whole + (raw - whole >= 0.5 ? 1 : 0);
  return math.max(s.minRecordingMs, math.min(s.maxRecordingMs, rounded));
}
