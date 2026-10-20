/**
 * sueori-scoring — 수어리 채점 SDK (온디바이스).
 *
 *   import { ScoringEngine } from "sueori-scoring";
 *   const engine = new ScoringEngine();
 *   const result = engine.score({ word, capture });
 *
 * `core`는 스펙 구현(공통 테스트 벡터 검증용)이며 앱에서 직접 쓸 필요는 없다.
 */
export * from "./engine.js";
export { InvalidInput } from "./core.js";
export * as core from "./core.js";
export * as spec from "./spec.js";
