/**
 * 채점 코어 — 스펙 ai-io-v1 §2·§3·§4·§10. 외부 의존성 없음.
 *
 * 비트 단위 재현 규칙(§10.0)
 * - 모든 계산은 Float64(JS number)
 * - 제곱합은 `dx * dx + dy * dy + dz * dz`를 왼쪽부터 계산한다(Math.hypot 금지)
 * - 21개 점 거리는 점 0→20 순서로 하나씩 더한 뒤 21로 나눈다(reduce 순서를 바꾸지 않는다)
 * - 반전(v1)은 정규화된 벡터의 x 성분 부호를 바꾼다(이미지 좌표 1 − x 금지)
 * 공통 테스트 벡터(vectors/ai-io-v1-vectors.json)로 Python 기준 구현과 일치를 확인한다.
 */
import * as S from "./spec.js";

export type Point = readonly [number, number, number] | readonly number[];
export type Frame = readonly Point[];
export type Vector = number[];

export class InvalidInput extends Error {
  constructor(message: string) {
    super(message);
    this.name = "InvalidInput";
  }
}

// ── §10.0 유효성 ──

export function checkFrame(frame: Frame): void {
  if (!Array.isArray(frame) || frame.length !== S.NUM_LANDMARKS) {
    throw new InvalidInput(`랜드마크는 ${S.NUM_LANDMARKS}개여야 함: ${frame?.length}`);
  }
  for (const p of frame) {
    if (!Array.isArray(p) || p.length !== 3) throw new InvalidInput("점은 (x, y, z) 3개 값이어야 함");
    for (const c of p) {
      if (typeof c !== "number" || !Number.isFinite(c)) throw new InvalidInput("NaN·Infinity 좌표가 있음");
    }
  }
}

// ── §2 정규화 ──

function dist3(a: Point, b: Point): number {
  const dx = a[0] - b[0];
  const dy = a[1] - b[1];
  const dz = a[2] - b[2];
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

export function palmScale(frame: Frame): number {
  let scale = 0.0;
  const idx = S.PALM_INDICES;
  for (let i = 0; i < idx.length; i++) {
    for (let j = i + 1; j < idx.length; j++) {
      const d = dist3(frame[idx[i]], frame[idx[j]]);
      if (d > scale) scale = d;
    }
  }
  return scale < S.SCALE_EPSILON ? 1.0 : scale;
}

export function normalizeFrame(frame: Frame): Vector {
  checkFrame(frame);
  const w = frame[0];
  const scale = palmScale(frame);
  const out: Vector = new Array(S.FEATURES);
  for (let i = 0; i < S.NUM_LANDMARKS; i++) {
    const p = frame[i];
    out[i * 3] = (p[0] - w[0]) / scale;
    out[i * 3 + 1] = (p[1] - w[1]) / scale;
    out[i * 3 + 2] = (p[2] - w[2]) / scale;
  }
  return out;
}

// ── §4 시간축 ──

export function resample(vectors: readonly Vector[], length: number = S.SEQUENCE_LENGTH): Vector[] {
  const n = vectors.length;
  if (n === 0) throw new InvalidInput("리샘플할 프레임이 없음");
  if (n === 1) return Array.from({ length }, () => vectors[0].slice());
  const out: Vector[] = [];
  for (let k = 0; k < length; k++) {
    const s = (k * (n - 1)) / (length - 1);
    const lo = Math.floor(s);
    const hi = Math.ceil(s);
    const w = s - lo;
    const a = vectors[lo];
    const b = vectors[hi];
    const v: Vector = new Array(a.length);
    for (let i = 0; i < a.length; i++) v[i] = a[i] + (b[i] - a[i]) * w;
    out.push(v);
  }
  return out;
}

export function staticTensor(frame: Frame): Vector[] {
  const v = normalizeFrame(frame);
  return Array.from({ length: S.SEQUENCE_LENGTH }, () => v.slice());
}

export function dynamicTensor(frames: readonly Frame[]): Vector[] {
  if (frames.length < S.MIN_DYNAMIC_FRAMES) {
    throw new InvalidInput(`유효 프레임 ${frames.length}개 < ${S.MIN_DYNAMIC_FRAMES}`);
  }
  return resample(frames.map(normalizeFrame));
}

// ── §10 폴백 ──

export function landmarkDistance(a: Vector, b: Vector): number {
  let total = 0.0;
  for (let i = 0; i < S.NUM_LANDMARKS; i++) {
    const dx = a[i * 3] - b[i * 3];
    const dy = a[i * 3 + 1] - b[i * 3 + 1];
    const dz = a[i * 3 + 2] - b[i * 3 + 2];
    total += Math.sqrt(dx * dx + dy * dy + dz * dz);
  }
  return total / S.NUM_LANDMARKS;
}

/** v1 반전: 정규화된 벡터의 x 성분 부호를 바꾼다(§10.5). */
export function mirrorVector(v: Vector): Vector {
  const out = v.slice();
  for (let i = 0; i < out.length; i += 3) out[i] = -out[i];
  return out;
}

function clampScore(x: number): number {
  return x < 0.0 ? 0.0 : x > 100.0 ? 100.0 : x;
}

export interface StaticResult {
  score: number;
  distance: number;
  mirrored: boolean;
}

/** §10.1 정적 거리 점수 + §10.5 반전(거리가 더 작은 쪽, 동률이면 원본). */
export function staticScore(reference: Frame, candidate: Frame): StaticResult {
  const r = normalizeFrame(reference);
  const c = normalizeFrame(candidate);
  let d = landmarkDistance(r, c);
  let mirrored = false;
  const dm = landmarkDistance(r, mirrorVector(c));
  if (dm < d) {
    d = dm;
    mirrored = true;
  }
  return { score: clampScore(100.0 * Math.exp(-S.STATIC_DECAY * d)), distance: d, mirrored };
}

export interface DtwResult {
  score: number;
  avgCost: number;
  pathLength: number;
  mirrored: boolean;
}

function dtwCore(a: readonly Vector[], b: readonly Vector[]): [number, number] {
  const n = a.length;
  const m = b.length;
  const d: number[][] = [];
  for (let i = 0; i <= n; i++) d.push(new Array(m + 1).fill(Infinity));
  d[0][0] = 0.0;
  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      const cost = landmarkDistance(a[i - 1], b[j - 1]);
      d[i][j] = cost + Math.min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1]);
    }
  }
  // (n, m) → (0, 0) 되짚기. 동률이면 diag → up → left
  let i = n;
  let j = m;
  let length = 0;
  while (i > 0 || j > 0) {
    length++;
    if (i === 0) j--;
    else if (j === 0) i--;
    else {
      const diag = d[i - 1][j - 1];
      const up = d[i - 1][j];
      const left = d[i][j - 1];
      const best = Math.min(diag, up, left);
      if (best === diag) {
        i--;
        j--;
      } else if (best === up) i--;
      else j--;
    }
  }
  return [d[n][m] / length, length];
}

/** §10.2 DTW 점수 + §10.5 반전(평균 정렬 비용이 더 작은 쪽, 동률이면 원본). 리샘플·윈도우 없음. */
export function dtwScore(reference: readonly Frame[], candidate: readonly Frame[]): DtwResult {
  const a = reference.map(normalizeFrame);
  const b = candidate.map(normalizeFrame);
  if (a.length === 0 || b.length === 0) return { score: 0.0, avgCost: Infinity, pathLength: 0, mirrored: false };
  let [avg, len] = dtwCore(a, b);
  let mirrored = false;
  const [am, lm] = dtwCore(a, b.map(mirrorVector));
  if (am < avg) {
    avg = am;
    len = lm;
    mirrored = true;
  }
  return { score: clampScore(100.0 * Math.exp(-S.DTW_DECAY * avg)), avgCost: avg, pathLength: len, mirrored };
}

export interface TrackInput {
  label: string;
  frames: readonly Frame[];
}

export type TrackScore =
  | { label: string; kind: "distance"; result: StaticResult }
  | { label: string; kind: "dtw"; result: DtwResult };

export interface TrackOutcome {
  best: TrackScore | null;
  /** 프레임이 모자라 건너뛴 트랙이 있었는지(동적 N<8) */
  sawShort: boolean;
}

/**
 * §3 트랙별 채점 → 최고점(동점이면 먼저 나타난 트랙). 좌표가 이상한 트랙과 프레임이 모자란 동적 트랙은 건너뛴다.
 * 정적은 트랙의 가운데 프레임(⌊N/2⌋), 동적은 트랙 전체.
 */
export function scoreTracks(reference: readonly Frame[], tracks: readonly TrackInput[], dynamic: boolean): TrackOutcome {
  let best: TrackScore | null = null;
  let sawShort = false;
  for (const t of tracks) {
    if (t.frames.length === 0) continue;
    try {
      t.frames.forEach(checkFrame);
    } catch {
      continue;
    }
    let cur: TrackScore;
    if (!dynamic) {
      cur = { label: t.label, kind: "distance", result: staticScore(reference[0], t.frames[Math.floor(t.frames.length / 2)]) };
    } else {
      if (t.frames.length < S.MIN_DYNAMIC_FRAMES) {
        sawShort = true;
        continue;
      }
      cur = { label: t.label, kind: "dtw", result: dtwScore(reference, t.frames) };
    }
    if (best === null || cur.result.score > best.result.score) best = cur;
  }
  return { best, sawShort };
}

/** §10.3 녹화 길이. 반올림은 half away from zero(floor(x + 0.5)를 쓰지 않음). */
export function recordingDurationMs(firstTMs?: number | null, lastTMs?: number | null): number {
  if (firstTMs == null || lastTMs == null || !(lastTMs > firstTMs)) return S.DEFAULT_RECORDING_MS;
  const raw = (lastTMs - firstTMs) * S.RECORDING_FACTOR;
  const whole = Math.floor(raw);
  const rounded = whole + (raw - whole >= 0.5 ? 1 : 0);
  return Math.max(S.MIN_RECORDING_MS, Math.min(S.MAX_RECORDING_MS, rounded));
}
