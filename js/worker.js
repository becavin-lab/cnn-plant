/* worker.js — fait tourner le CNN en arrière-plan pour que la page reste fluide. */
importScripts("cnn.js");
let model = null;

// les n cartes les plus actives d'une couche, ramenées entre 0 et 255
function cartesActives(t, n) {
  const HW = t.H * t.W, moy = [];
  for (let c = 0; c < t.C; c++) { let s = 0; for (let i = 0; i < HW; i++) s += t.d[c * HW + i]; moy.push([s / HW, c]); }
  moy.sort((a, b) => b[0] - a[0]);
  return moy.slice(0, n).map(([m, c]) => {
    let mx = 1e-9; for (let i = 0; i < HW; i++) mx = Math.max(mx, t.d[c * HW + i]);
    const px = new Uint8ClampedArray(HW);
    for (let i = 0; i < HW; i++) px[i] = Math.round(255 * Math.max(0, t.d[c * HW + i]) / mx);
    return { c, px };
  });
}

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === "charger") {
      const meta = await (await fetch(m.base + "model.json")).json();
      const rep = await fetch(m.base + "weights.bin");
      if (!rep.ok) throw new Error("weights.bin introuvable (" + rep.status + ")");
      const total = +rep.headers.get("Content-Length") || 0, lecteur = rep.body.getReader(), morceaux = [];
      let recu = 0;
      for (;;) {
        const { done, value } = await lecteur.read(); if (done) break;
        morceaux.push(value); recu += value.length;
        self.postMessage({ type: "chargement", recu, total });
      }
      const buf = new Uint8Array(recu); let o = 0; for (const c of morceaux) { buf.set(c, o); o += c.length; }
      model = new CNN.Model(meta, buf.buffer);
      self.postMessage({ type: "pret", filtres: Array.from(model.t(meta.conv1_raw)), forme: meta.conv1_raw.shape });
    } else if (m.type === "predire") {
      const t0 = performance.now(), couches = [];
      const x = CNN.preprocess(m.rgba, model.meta.mean, model.meta.std);
      const z = model.forward(x,
        m.couches ? (nom, t) => couches.push({ nom, C: t.C, H: t.H, W: t.W, cartes: cartesActives(t, 8) }) : null,
        (p, nom) => self.postMessage({ type: "progression", id: m.id, p, nom }));
      self.postMessage({ type: "resultat", id: m.id, prob: CNN.softmax(z), couches, ms: performance.now() - t0 });
    }
  } catch (err) {
    self.postMessage({ type: "erreur", id: m.id, message: String((err && err.message) || err) });
  }
};
