// Test : le moteur JS donne-t-il les mêmes scores que la référence numpy ?
const fs = require("fs"), path = require("path");
const CNN = require("../js/cnn.js");
const dir = process.argv[2];
function npy(f) {               // lecteur .npy minimal (float32 / float64)
  const b = fs.readFileSync(f), hl = b.readUInt16LE(8), hdr = b.slice(10, 10 + hl).toString();
  const data = b.slice(10 + hl), ab = data.buffer.slice(data.byteOffset, data.byteOffset + data.length);
  return hdr.includes("<f8") ? Float32Array.from(new Float64Array(ab)) : new Float32Array(ab);
}
const meta = JSON.parse(fs.readFileSync(path.join(dir, "model/model.json")));
const wb = fs.readFileSync(path.join(dir, "model/weights.bin"));
const model = new CNN.Model(meta, wb.buffer.slice(wb.byteOffset, wb.byteOffset + wb.length));
const x = npy(path.join(dir, "test_input.npy")), ref = npy(path.join(dir, "folded_logits.npy"));
const t0 = Date.now(); const taps = {};
const z = model.forward(x, (n, t) => taps[n] = [t.C, t.H, t.W]);
const ms = Date.now() - t0;
let maxAbs = 0, maxRef = 0, am = 0, ar = 0;
for (let i = 0; i < z.length; i++) {
  maxAbs = Math.max(maxAbs, Math.abs(z[i] - ref[i])); maxRef = Math.max(maxRef, Math.abs(ref[i]));
  if (z[i] > z[am]) am = i; if (ref[i] > ref[ar]) ar = i;
}
console.log(JSON.stringify({ ms, argmax_js: am, argmax_ref: ar, max_abs_diff: maxAbs, max_ref: maxRef, rel: maxAbs / maxRef, taps }));
