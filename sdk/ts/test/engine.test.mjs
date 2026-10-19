// ScoringEngine(공개 API) 동작 — 상태 코드, 트랙 선택, 버전 정보.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { InvalidInput, ScoringEngine, spec } from "../dist/index.js";

const V = JSON.parse(readFileSync(new URL("../../../vectors/ai-io-v1-vectors.json", import.meta.url), "utf8"));
const dtwCase = (name) => V.dtw.find((c) => c.name === name);
const staticCase = (name) => V.static_score.find((c) => c.name === name);
const capture = (frames, handedness = "Right") => frames.map((lm, k) => ({ tMs: 33 * k, hands: [{ handedness, landmarks: lm }] }));

const engine = new ScoringEngine();

test("동적 정답: 벡터와 같은 점수, method=dtw, 버전 정보", () => {
  const c = dtwCase("same_motion_slower_noisy");
  const r = engine.score({ word: { wordKey: "w", type: "dynamic", referenceFrames: c.reference }, capture: capture(c.candidate) });
  assert.equal(r.status, "ok");
  assert.ok(Math.abs(r.score - c.expected.score) <= 1e-9);
  assert.equal(r.method, "dtw");
  assert.equal(r.passed, r.score >= 80);
  assert.equal(r.track, "Right");
  assert.equal(r.specVersion, spec.SPEC_VERSION);
  assert.equal(r.sdkVersion, spec.SDK_VERSION);
  assert.equal(r.modelVersion, null);
  assert.equal(r.fallbackReason, "model_not_configured");
  assert.equal(r.details.pathLength, c.expected.path_length);
});

test("왼손(반전) 입력은 v1에서 반전으로 채점된다", () => {
  const c = dtwCase("v1_mirror_wins_left_hand");
  const r = engine.score({ word: { wordKey: "w", type: "dynamic", referenceFrames: c.reference }, capture: capture(c.candidate, "Left") });
  assert.equal(r.mirrored, true);
  assert.ok(Math.abs(r.score - c.expected.score) <= 1e-9);
});

test("정적: 트랙 가운데 프레임으로 채점, method=distance", () => {
  const c = staticCase("same_pose_noise");
  const other = staticCase("open_vs_fist").candidate;
  // 가운데(⌊3/2⌋=1번) 프레임만 정답 자세
  const r = engine.score({
    word: { wordKey: "s", type: "static", referenceFrames: [c.reference] },
    capture: capture([other, c.candidate, other]),
  });
  assert.equal(r.method, "distance");
  assert.ok(Math.abs(r.score - c.expected.score) <= 1e-9);
});

test("손이 한 번도 안 잡히면 no_hand", () => {
  const c = dtwCase("identical");
  const r = engine.score({ word: { wordKey: "w", type: "dynamic", referenceFrames: c.reference }, capture: [{ tMs: 0, hands: [] }] });
  assert.equal(r.status, "no_hand");
  assert.equal(r.score, null);
  assert.equal(r.passed, false);
});

test("동적 유효 프레임 8개 미만이면 insufficient_frames", () => {
  const c = dtwCase("identical");
  const r = engine.score({ word: { wordKey: "w", type: "dynamic", referenceFrames: c.reference }, capture: capture(c.candidate.slice(0, 7)) });
  assert.equal(r.status, "insufficient_frames");
});

test("좌표가 이상한 트랙만 있으면 invalid, 정상 트랙이 있으면 그 트랙으로", () => {
  const c = dtwCase("identical");
  const bad = c.candidate.map((f) => f.map((p, i) => (i === 3 ? [NaN, p[1], p[2]] : p)));
  const word = { wordKey: "w", type: "dynamic", referenceFrames: c.reference };
  assert.equal(engine.score({ word, capture: capture(bad) }).status, "invalid");
  const mixed = c.candidate.map((lm, k) => ({ tMs: 33 * k, hands: [{ handedness: "Left", landmarks: bad[k] }, { handedness: "Right", landmarks: lm }] }));
  const r = engine.score({ word, capture: mixed });
  assert.equal(r.status, "ok");
  assert.equal(r.track, "Right");
  assert.equal(r.score, 100);
});

test("기준 데이터가 잘못되면 InvalidInput을 던진다", () => {
  assert.throws(() => engine.score({ word: { wordKey: "w", type: "dynamic", referenceFrames: [] }, capture: [] }), InvalidInput);
  assert.throws(() => engine.score({ word: { wordKey: "w", type: "other", referenceFrames: [[]] }, capture: [] }), InvalidInput);
});

test("frameSize·inputMirrored는 받기만 하고 결과에 영향 없음", () => {
  const c = dtwCase("same_motion_slower_noisy");
  const word = { wordKey: "w", type: "dynamic", referenceFrames: c.reference };
  const a = engine.score({ word, capture: capture(c.candidate) });
  const b = engine.score({ word, capture: capture(c.candidate), frameSize: { width: 720, height: 1280, rotation: 0 }, inputMirrored: true });
  assert.deepEqual(a, b);
});

test("녹화 길이", () => {
  assert.equal(engine.recordingDurationMs({}), 3000);
  assert.equal(engine.recordingDurationMs({ referenceTMs: [0, 2500] }), 3000);
  assert.equal(engine.recordingDurationMs({ referenceTMs: [33, 500, 1867] }), 2201);
});
