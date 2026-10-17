/// 채점 엔진 — 앱이 호출하는 공개 API. 설계: 볼트 `06-AI/수어리 - 채점 SDK 설계 (초안).md`.
///
/// SDK 0.1은 폴백 전용이다(모델 추론은 0.2에서 추가하며, 이 API는 그대로 유지한다).
library;

import 'core.dart';
import 'spec.dart' as s;

/// 정적(지문자) / 동적(단어). 세 SDK 공통 문자열은 [wire]("static"/"dynamic")
enum WordType {
  staticSign('static'),
  dynamicSign('dynamic');

  const WordType(this.wire);
  final String wire;
}

class WordReference {
  const WordReference({
    required this.wordKey,
    required this.type,
    required this.referenceFrames,
    this.referenceTMs,
  });

  /// 단어 정본 키(예: "family/husband"). 모델 라벨·disabled_words 대조에 쓴다
  final String wordKey;
  final WordType type;

  /// 기준 프레임들. 정적은 길이 1. 각 프레임은 21개 [x, y, z](MediaPipe 이미지 좌표)
  final List<Frame> referenceFrames;

  /// 기준 프레임별 시각(ms). 녹화 길이 계산에 쓴다(없으면 3초)
  final List<num>? referenceTMs;
}

class CapturedHand {
  const CapturedHand({this.handedness, required this.landmarks});

  /// MediaPipe handedness 라벨("Left"/"Right"). 없으면 null
  final String? handedness;
  final Frame landmarks;
}

class CapturedFrame {
  const CapturedFrame({required this.tMs, required this.hands});

  /// 프레임 캡처 시각(ms, 추론 완료 시각이 아님)
  final num tMs;

  /// 잡힌 손 전부. 손이 없으면 빈 리스트
  final List<CapturedHand> hands;
}

class FrameSize {
  const FrameSize({required this.width, required this.height, this.rotation = 0});
  final int width;
  final int height;
  final int rotation;
}

class ScoreRequest {
  const ScoreRequest({required this.word, required this.capture, this.frameSize, this.inputMirrored});
  final WordReference word;
  final List<CapturedFrame> capture;

  /// 0.1은 받기만 한다(스펙 TODO-7 종횡비 보정 대비)
  final FrameSize? frameSize;

  /// 0.1은 받기만 한다
  final bool? inputMirrored;
}

enum ScoreStatus { ok, noHand, insufficientFrames, invalid }

enum ScoreMethod { model, dtw, distance }

enum FallbackReason {
  modelNotConfigured,
  modelDisabled,
  wordNotInModel,
  wordDisabled,
  loadFailed,
  shaMismatch,
  specMismatch,
  appVersion,
}

/// 세 SDK 공통 문자열 값(BE 저장·로그용)
const _statusWire = {
  ScoreStatus.ok: 'ok',
  ScoreStatus.noHand: 'no_hand',
  ScoreStatus.insufficientFrames: 'insufficient_frames',
  ScoreStatus.invalid: 'invalid',
};
const _reasonWire = {
  FallbackReason.modelNotConfigured: 'model_not_configured',
  FallbackReason.modelDisabled: 'model_disabled',
  FallbackReason.wordNotInModel: 'word_not_in_model',
  FallbackReason.wordDisabled: 'word_disabled',
  FallbackReason.loadFailed: 'load_failed',
  FallbackReason.shaMismatch: 'sha_mismatch',
  FallbackReason.specMismatch: 'spec_mismatch',
  FallbackReason.appVersion: 'app_version',
};

class ScoreResult {
  const ScoreResult({
    required this.status,
    this.score,
    this.method,
    this.fallbackReason = FallbackReason.modelNotConfigured,
    this.mirrored = false,
    this.track,
    this.modelVersion,
    this.details = const {},
  });

  final ScoreStatus status;

  /// 0~100. status가 ok일 때만
  final double? score;
  bool get passed => score != null && score! >= s.passThreshold;
  double get threshold => s.passThreshold;
  final ScoreMethod? method;

  /// 모델을 쓰지 않은 이유. 0.1은 항상 modelNotConfigured
  final FallbackReason? fallbackReason;

  /// 폴백에서 좌우 반전 쪽을 썼는지(스펙 §10.5)
  final bool mirrored;

  /// 채점에 쓴 손(handedness 라벨, 없으면 "")
  final String? track;
  String get specVersion => s.specVersion;
  final String? modelVersion;
  String get sdkVersion => s.sdkVersion;

  /// 폴백: distance 또는 avgCost·pathLength
  final Map<String, num> details;

  /// 세 SDK 공통 필드 이름(TS ScoreResult와 같음)
  Map<String, Object?> toJson() => {
        'status': _statusWire[status],
        'score': score,
        'passed': passed,
        'threshold': threshold,
        'method': method?.name,
        'fallbackReason': fallbackReason == null ? null : _reasonWire[fallbackReason],
        'mirrored': mirrored,
        'track': track,
        'specVersion': specVersion,
        'modelVersion': modelVersion,
        'sdkVersion': sdkVersion,
        'details': details,
      };
}

class ScoringEngine {
  /// [manifestUrl], [enableModel]은 0.2에서 쓰는 자리. 0.1은 무시하고 항상 폴백
  ScoringEngine({String? manifestUrl, bool enableModel = true});

  String get specVersion => s.specVersion;
  String get sdkVersion => s.sdkVersion;

  /// 동적 단어 녹화 길이(스펙 §10.3). 기준 시각이 없으면 3000ms
  int recordingDurationMs(WordReference word) {
    final t = word.referenceTMs;
    if (t == null || t.isEmpty) return s.defaultRecordingMs;
    return recordingDurationMsFromTMs(t.first, t.last);
  }

  /// 채점. 트랙(손)마다 채점해 최고점을 쓴다(동점이면 먼저 나타난 손).
  /// 정적 단어는 트랙의 가운데 프레임 1장, 동적 단어는 트랙 전체(유효 프레임 8개 이상)를 쓴다.
  /// 기준 데이터가 잘못되면 [InvalidInput]을 던진다(앱·콘텐츠 오류). 사용자 입력 문제는 status로 돌려준다.
  ScoreResult score(ScoreRequest request) {
    final word = request.word;
    if (word.referenceFrames.isEmpty) throw InvalidInput('기준 프레임이 없음');
    word.referenceFrames.forEach(checkFrame);

    final byLabel = <String, List<Frame>>{};
    for (final fr in request.capture) {
      for (final hand in fr.hands) {
        byLabel.putIfAbsent(hand.handedness ?? '', () => []).add(hand.landmarks);
      }
    }
    if (byLabel.isEmpty) return const ScoreResult(status: ScoreStatus.noHand);

    final tracks = [for (final e in byLabel.entries) TrackInput(e.key, e.value)];
    final out = scoreTracks(word.referenceFrames, tracks, dynamic: word.type == WordType.dynamicSign);
    final best = out.best;
    if (best == null) {
      return ScoreResult(status: out.sawShort ? ScoreStatus.insufficientFrames : ScoreStatus.invalid);
    }
    final st = best.staticResult;
    final dt = best.dtwResult;
    return ScoreResult(
      status: ScoreStatus.ok,
      score: best.score,
      method: st != null ? ScoreMethod.distance : ScoreMethod.dtw,
      mirrored: best.mirrored,
      track: best.label,
      details: st != null ? {'distance': st.distance} : {'avgCost': dt!.avgCost, 'pathLength': dt.pathLength},
    );
  }
}

/// [recordingDurationMs] 코어 함수의 별칭(엔진 메서드와 이름이 겹치지 않게)
int recordingDurationMsFromTMs(num? firstTMs, num? lastTMs) => recordingDurationMs(firstTMs, lastTMs);
