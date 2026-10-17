// 공통 테스트 벡터(vectors/ai-io-v1-vectors.json) 적합성 — Python 기준 구현과 같은 결과인지 확인한다.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { core, spec } from "../dist/index.js";

const V = JSON.parse(readFileSync(new URL("../../../vectors/ai-io-v1-vectors.json", import.meta.url), "utf8"));
const TOL = V.tolerances;

const dec = (x) =>
  typeof x === "string" ? { NaN: NaN, Infinity: Infinity, "-Infinity": -Infinity }[x] : Array.isArray(x) ? x.map(dec) : x;

function maxErr(a, b) {
  if (Array.isArray(a)) {
    assert.equal(a.length, b.length);
    return Math.max(0, ...a.map((x, i) => maxErr(x, b[i])));
  }
  return Math.abs(a - b);
}

test("spec_version", () => assert.equal(V.spec_version, spec.SPEC_VERSION));

for (const c of V.normalize) {
  test(`normalize ${c.name}`, () => {
    const f = dec(c.frame);
    if (c.invalid) return assert.throws(() => core.normalizeFrame(f), core.InvalidInput);
    assert.ok(maxErr(core.normalizeFrame(f), c.expected) <= TOL.normalize_f64);
  });
}

for (const c of V.tensor) {
  test(`tensor ${c.name}`, () => {
    const frames = dec(c.frames);
    const fn = () => (c.kind === "static" ? core.staticTensor(frames[0]) : core.dynamicTensor(frames));
    if (c.invalid) return assert.throws(fn, core.InvalidInput);
    assert.ok(maxErr(fn(), c.expected) <= TOL.tensor_f64);
  });
}

for (const c of V.static_score) {
  test(`static_score ${c.name}`, () => {
    const r = dec(c.reference);
    const k = dec(c.candidate);
    if (c.invalid) return assert.throws(() => core.staticScore(r, k), core.InvalidInput);
    const got = core.staticScore(r, k);
    assert.ok(Math.abs(got.distance - c.expected.distance) <= TOL.distance, `distance ${got.distance}`);
    assert.ok(Math.abs(got.score - c.expected.score) <= TOL.score, `score ${got.score}`);
    assert.equal(got.mirrored, c.expected.mirrored);
  });
}

for (const c of V.dtw) {
  test(`dtw ${c.name}`, () => {
    const r = dec(c.reference);
    const k = dec(c.candidate);
    if (c.invalid) {
      // §4 유효성(N<8)은 트랙 단위 채점에서 걸러진다
      const out = core.scoreTracks(r, [{ label: "", frames: k }], true);
      assert.equal(out.best, null);
      assert.equal(out.sawShort, true);
      return;
    }
    const got = core.dtwScore(r, k);
    assert.equal(got.pathLength, c.expected.path_length);
    assert.ok(Math.abs(got.avgCost - c.expected.avg_cost) <= TOL.avg_cost, `avgCost ${got.avgCost}`);
    assert.ok(Math.abs(got.score - c.expected.score) <= TOL.score, `score ${got.score}`);
    assert.equal(got.mirrored, c.expected.mirrored);
  });
}

for (const c of V.best_track) {
  test(`best_track ${c.name}`, () => {
    const tracks = c.tracks.map((t) => ({ label: t.label, frames: dec(t.frames) }));
    const out = core.scoreTracks(dec(c.reference), tracks, c.kind === "dynamic");
    if (c.expected === null) return assert.equal(out.best, null);
    assert.equal(out.best.label, c.expected.label);
    assert.ok(Math.abs(out.best.result.score - c.expected.score) <= TOL.score);
  });
}

for (const c of V.recording_duration) {
  test(`recording_duration ${c.first_t_ms}→${c.last_t_ms}`, () => {
    assert.equal(core.recordingDurationMs(c.first_t_ms, c.last_t_ms), c.expected);
  });
}
