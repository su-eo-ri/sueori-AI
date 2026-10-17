/**
 * 채점 엔진 — 앱이 호출하는 공개 API. 설계: 볼트 `06-AI/수어리 - 채점 SDK 설계 (초안).md`.
 *
 * SDK 0.1은 폴백 전용이다(모델 추론은 0.2에서 추가하며, 이 API는 그대로 유지한다).
 */
import {
  type Frame,
  type TrackInput,
  InvalidInput,
  checkFrame,
  recordingDurationMs as durationFromTMs,
  scoreTracks,
} from "./core.js";
import * as S from "./spec.js";

export type WordType = "static" | "dynamic";

export interface WordReference {
  /** 단어 정본 키(예: "family/husband"). 모델 라벨·disabled_words 대조에 쓴다 */
  wordKey: string;
  type: WordType;
  /** 기준 프레임들. 정적은 길이 1. 각 프레임은 21개 [x, y, z](MediaPipe 이미지 좌표) */
  referenceFrames: readonly Frame[];
  /** 기준 프레임별 시각(ms). 녹화 길이 계산에 쓴다(없으면 3초) */
  referenceTMs?: readonly number[] | null;
}

export interface CapturedHand {
  /** MediaPipe handedness 라벨("Left"/"Right"). 없으면 null */
  handedness?: string | null;
  landmarks: Frame;
}

export interface CapturedFrame {
  /** 프레임 캡처 시각(ms, 추론 완료 시각이 아님) */
  tMs: number;
  /** 잡힌 손 전부. 손이 없으면 빈 배열 */
  hands: readonly CapturedHand[];
}

export interface FrameSize {
  width: number;
  height: number;
  rotation: number;
}

export interface ScoreRequest {
  word: WordReference;
  capture: readonly CapturedFrame[];
  /** 0.1은 받기만 한다(스펙 TODO-7 종횡비 보정 대비) */
  frameSize?: FrameSize | null;
  /** 0.1은 받기만 한다 */
  inputMirrored?: boolean | null;
}

export type ScoreStatus = "ok" | "no_hand" | "insufficient_frames" | "invalid";
export type ScoreMethod = "model" | "dtw" | "distance";
export type FallbackReason =
  | "model_not_configured"
  | "model_disabled"
  | "word_not_in_model"
  | "word_disabled"
  | "load_failed"
  | "sha_mismatch"
  | "spec_mismatch"
  | "app_version";

export interface ScoreResult {
  status: ScoreStatus;
  /** 0~100. status가 "ok"일 때만 */
  score: number | null;
  passed: boolean;
  threshold: number;
  method: ScoreMethod | null;
  /** 모델을 쓰지 않은 이유. 0.1은 항상 "model_not_configured" */
  fallbackReason: FallbackReason | null;
  /** 폴백에서 좌우 반전 쪽을 썼는지(스펙 §10.5) */
  mirrored: boolean;
  /** 채점에 쓴 손(handedness 라벨, 없으면 "") */
  track: string | null;
  specVersion: string;
  modelVersion: string | null;
  sdkVersion: string;
  details: { distance?: number; avgCost?: number; pathLength?: number };
}

export interface ScoringEngineOptions {
  /** 0.2에서 쓰는 자리. 0.1은 무시하고 항상 폴백 */
  manifestUrl?: string;
  enableModel?: boolean;
}

/** 스펙 §3: handedness 라벨별 트랙(처음 나타난 순서). 손이 없는 프레임은 버린다. */
function splitTracks(capture: readonly CapturedFrame[]): TrackInput[] {
  const byLabel = new Map<string, Frame[]>();
  for (const fr of capture) {
    for (const hand of fr.hands ?? []) {
      const label = hand.handedness ?? "";
      let frames = byLabel.get(label);
      if (!frames) byLabel.set(label, (frames = []));
      frames.push(hand.landmarks);
    }
  }
  return [...byLabel].map(([label, frames]) => ({ label, frames }));
}

function emptyResult(status: ScoreStatus): ScoreResult {
  return {
    status,
    score: null,
    passed: false,
    threshold: S.PASS_THRESHOLD,
    method: null,
    fallbackReason: "model_not_configured",
    mirrored: false,
    track: null,
    specVersion: S.SPEC_VERSION,
    modelVersion: null,
    sdkVersion: S.SDK_VERSION,
    details: {},
  };
}

export class ScoringEngine {
  readonly specVersion = S.SPEC_VERSION;
  readonly sdkVersion = S.SDK_VERSION;

  constructor(_options: ScoringEngineOptions = {}) {}

  /** 동적 단어 녹화 길이(스펙 §10.3). 기준 시각이 없으면 3000ms */
  recordingDurationMs(word: Pick<WordReference, "referenceTMs">): number {
    const t = word.referenceTMs;
    if (!t || t.length === 0) return S.DEFAULT_RECORDING_MS;
    return durationFromTMs(t[0], t[t.length - 1]);
  }

  /**
   * 채점. 트랙(손)마다 채점해 최고점을 쓴다(동점이면 먼저 나타난 손).
   * 정적 단어는 트랙의 가운데 프레임 1장, 동적 단어는 트랙 전체(유효 프레임 8개 이상)를 쓴다.
   * 기준 데이터가 잘못되면 InvalidInput을 던진다(앱·콘텐츠 오류). 사용자 입력 문제는 status로 돌려준다.
   */
  score(request: ScoreRequest): ScoreResult {
    const { word } = request;
    if (word.type !== "static" && word.type !== "dynamic") throw new InvalidInput(`word.type 값 이상: ${word.type}`);
    if (!word.referenceFrames || word.referenceFrames.length === 0) throw new InvalidInput("기준 프레임이 없음");
    word.referenceFrames.forEach(checkFrame);

    const tracks = splitTracks(request.capture ?? []);
    if (tracks.length === 0) return emptyResult("no_hand");

    const { best, sawShort } = scoreTracks(word.referenceFrames, tracks, word.type === "dynamic");
    if (best) {
      const score = best.result.score;
      const details =
        best.kind === "distance"
          ? { distance: best.result.distance }
          : { avgCost: best.result.avgCost, pathLength: best.result.pathLength };
      return {
        ...emptyResult("ok"),
        score,
        passed: score >= S.PASS_THRESHOLD,
        method: best.kind,
        mirrored: best.result.mirrored,
        track: best.label,
        details,
      };
    }
    return emptyResult(sawShort ? "insufficient_frames" : "invalid");
  }
}
