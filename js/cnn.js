/* cnn.js — un ResNet18 écrit à la main, sans librairie.
   Chaque couche est une boucle lisible : convolution, ReLU, max pooling, moyenne, couche dense.
   Utilisable dans un Web Worker (navigateur) ou dans Node (tests). */
(function (root) {
  "use strict";

  // ---- float16 -> float32
  function halfToFloat(u16) {
    const out = new Float32Array(u16.length);
    for (let i = 0; i < u16.length; i++) {
      const h = u16[i], s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
      out[i] = e === 0 ? s * Math.pow(2, -14) * (f / 1024)
             : e === 31 ? (f ? NaN : s * Infinity)
             : s * Math.pow(2, e - 15) * (1 + f / 1024);
    }
    return out;
  }

  function Model(meta, buffer) {
    this.meta = meta;
    this.W = halfToFloat(new Uint16Array(buffer));
  }
  Model.prototype.t = function (ref) {           // vue sur un tenseur de poids
    let n = 1; for (const d of ref.shape) n *= d;
    return this.W.subarray(ref.off, ref.off + n);
  };

  // ---- convolution 2D : x [C,H,W] -> [Co,Ho,Wo]
  function conv2d(x, C, H, W, w, b, Co, k, stride, pad, relu) {
    const Ho = Math.floor((H + 2 * pad - k) / stride) + 1, Wo = Math.floor((W + 2 * pad - k) / stride) + 1;
    const out = new Float32Array(Co * Ho * Wo), HW = H * W, HWo = Ho * Wo;
    for (let co = 0; co < Co; co++) {
      const ob = co * HWo, bias = b[co];
      for (let i = 0; i < HWo; i++) out[ob + i] = bias;
      for (let ci = 0; ci < C; ci++) {
        const ib = ci * HW, wb = (co * C + ci) * k * k;
        for (let ky = 0; ky < k; ky++) {
          // lignes de sortie dont la ligne d'entrée correspondante existe
          const oy0 = Math.max(0, Math.ceil((pad - ky) / stride));
          const oy1 = Math.min(Ho - 1, Math.floor((H - 1 + pad - ky) / stride));
          for (let kx = 0; kx < k; kx++) {
            const wv = w[wb + ky * k + kx];
            if (wv === 0) continue;
            const ox0 = Math.max(0, Math.ceil((pad - kx) / stride));
            const ox1 = Math.min(Wo - 1, Math.floor((W - 1 + pad - kx) / stride));
            for (let oy = oy0; oy <= oy1; oy++) {
              const iy = oy * stride - pad + ky, irow = ib + iy * W - pad + kx, orow = ob + oy * Wo;
              if (stride === 1) {
                for (let ox = ox0; ox <= ox1; ox++) out[orow + ox] += wv * x[irow + ox];
              } else {
                for (let ox = ox0; ox <= ox1; ox++) out[orow + ox] += wv * x[irow + ox * stride];
              }
            }
          }
        }
      }
      if (relu) for (let i = 0; i < HWo; i++) if (out[ob + i] < 0) out[ob + i] = 0;
    }
    return { d: out, C: Co, H: Ho, W: Wo };
  }

  function maxpool(t, k, stride, pad) {
    const { d, C, H, W } = t;
    const Ho = Math.floor((H + 2 * pad - k) / stride) + 1, Wo = Math.floor((W + 2 * pad - k) / stride) + 1;
    const out = new Float32Array(C * Ho * Wo);
    for (let c = 0; c < C; c++) for (let oy = 0; oy < Ho; oy++) for (let ox = 0; ox < Wo; ox++) {
      let m = -Infinity;
      for (let ky = 0; ky < k; ky++) {
        const iy = oy * stride - pad + ky; if (iy < 0 || iy >= H) continue;
        for (let kx = 0; kx < k; kx++) {
          const ix = ox * stride - pad + kx; if (ix < 0 || ix >= W) continue;
          const v = d[(c * H + iy) * W + ix]; if (v > m) m = v;
        }
      }
      out[(c * Ho + oy) * Wo + ox] = m;
    }
    return { d: out, C, H: Ho, W: Wo };
  }

  function convRef(model, t, ref, stride, pad, relu) {
    const s = ref.w.shape;  // [Co, Ci, k, k]
    return conv2d(t.d, t.C, t.H, t.W, model.t(ref.w), model.t(ref.b), s[0], s[2], stride, pad, relu);
  }

  // ---- passe avant complète ; onTap(nom, tenseur) reçoit les cartes intermédiaires
  Model.prototype.forward = function (input, onTap, onProgress) {
    let t = { d: input, C: 3, H: 224, W: 224 };
    const layers = this.meta.layers;
    for (let li = 0; li < layers.length; li++) {
      const L = layers[li];
      if (L.type === "conv") {
        const s = L.w.shape;
        t = conv2d(t.d, t.C, t.H, t.W, this.t(L.w), this.t(L.b), s[0], s[2], L.stride, L.pad, L.relu);
      } else if (L.type === "maxpool") {
        t = maxpool(t, L.k, L.stride, L.pad);
      } else if (L.type === "block") {           // bloc résiduel : 2 convolutions + raccourci
        const y1 = convRef(this, t, L.c1, L.stride, 1, true);
        const y2 = convRef(this, y1, L.c2, 1, 1, false);
        const idt = L.down ? convRef(this, t, L.down, L.stride, 0, false) : t;
        for (let i = 0; i < y2.d.length; i++) { const v = y2.d[i] + idt.d[i]; y2.d[i] = v > 0 ? v : 0; }
        t = y2;
      } else if (L.type === "gap") {             // moyenne de chaque carte -> vecteur de 512
        const v = new Float32Array(t.C), HW = t.H * t.W;
        for (let c = 0; c < t.C; c++) { let s = 0; for (let i = 0; i < HW; i++) s += t.d[c * HW + i]; v[c] = s / HW; }
        t = { d: v, C: t.C, H: 1, W: 1 };
      } else if (L.type === "fc") {              // couche dense : un score par espèce
        const w = this.t(L.w), b = this.t(L.b), [No, Ni] = L.w.shape, out = new Float32Array(No);
        for (let o = 0; o < No; o++) { let s = b[o]; const wb = o * Ni; for (let i = 0; i < Ni; i++) s += w[wb + i] * t.d[i]; out[o] = s; }
        t = { d: out, C: No, H: 1, W: 1 };
      }
      if (L.tap && onTap) onTap(L.tap, t);
      if (onProgress) onProgress((li + 1) / layers.length, L.name);
    }
    return t.d;
  };

  function softmax(z) {
    let m = -Infinity; for (const v of z) if (v > m) m = v;
    const e = new Float32Array(z.length); let s = 0;
    for (let i = 0; i < z.length; i++) { e[i] = Math.exp(z[i] - m); s += e[i]; }
    for (let i = 0; i < z.length; i++) e[i] /= s;
    return e;
  }

  // image RGBA 224x224 -> tenseur [3,224,224] normalisé (moyenne / écart-type ImageNet)
  function preprocess(rgba, mean, std) {
    const N = 224 * 224, x = new Float32Array(3 * N);
    for (let i = 0; i < N; i++) for (let c = 0; c < 3; c++) x[c * N + i] = (rgba[4 * i + c] / 255 - mean[c]) / std[c];
    return x;
  }

  root.CNN = { Model, softmax, preprocess, halfToFloat, conv2d };
  if (typeof module !== "undefined") module.exports = root.CNN;
})(typeof self !== "undefined" ? self : globalThis);
