// TF.js 런타임으로 변환된 모델을 실제로 로드·추론한다(tools/smoke_convert.py가 호출).
//   node check.mjs <model_dir> <fixture.json>
// 출력(stdout): {"logits": [[...]], "load_ms": n, "predict_ms": n}
// Node의 순수 JS CPU 백엔드라 속도는 브라우저(WebGL/WASM) 성능을 대표하지 않는다. 정확도 확인용.
import { readFile } from "node:fs/promises";
import path from "node:path";
import * as tf from "@tensorflow/tfjs";

const [modelDir, fixturePath] = process.argv.slice(2);

const fsHandler = {
  async load() {
    const modelJson = JSON.parse(await readFile(path.join(modelDir, "model.json"), "utf8"));
    const weightSpecs = [];
    const buffers = [];
    for (const group of modelJson.weightsManifest) {
      weightSpecs.push(...group.weights);
      for (const p of group.paths) buffers.push(await readFile(path.join(modelDir, p)));
    }
    const all = Buffer.concat(buffers);
    return {
      modelTopology: modelJson.modelTopology,
      format: modelJson.format,
      generatedBy: modelJson.generatedBy,
      convertedBy: modelJson.convertedBy,
      weightSpecs,
      weightData: all.buffer.slice(all.byteOffset, all.byteOffset + all.byteLength),
    };
  },
};

await tf.setBackend("cpu");
const fixture = JSON.parse(await readFile(fixturePath, "utf8"));

let t0 = performance.now();
const model = await tf.loadLayersModel(fsHandler);
const loadMs = performance.now() - t0;

const logits = [];
t0 = performance.now();
for (const input of fixture.inputs) {
  const y = model.predict(tf.tensor3d([input]));
  logits.push(Array.from(await y.data()));
  y.dispose();
}
const predictMs = (performance.now() - t0) / fixture.inputs.length;

process.stdout.write(JSON.stringify({ logits, load_ms: loadMs, predict_ms: predictMs, tfjs: tf.version.tfjs }));
