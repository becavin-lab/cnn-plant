/* app.js — logique de la page « Identifier une plante avec un CNN ». */
"use strict";
const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));
const fmt = (x, d = 0) => x.toLocaleString("fr-FR", { minimumFractionDigits: d, maximumFractionDigits: d });

const S = {
  etape: 1, exemples: [], especes: [], meta: null, pret: false,
  photo: 0,             // photo choisie dans la partie cours (indice dans S.exemples)
  images: {},           // cache des images chargées
  rgba: null,           // pixels 224 x 224 de la photo du cours
  gris: null,           // niveaux de gris (Float32Array 224*224)
  carte: null,          // dernière carte de convolution (valeurs absolues normalisées)
  tp: null,             // { rgba, exemple } de la partie TP
  id: 0, attente: {},
};
const worker = new Worker("js/worker.js");

// ------------------------------------------------------------------ navigation (comme le site ECG)
function allerA(k) {
  const classer = k === "classer";
  if (!classer) S.etape = k;
  $$("main > section").forEach((s) => (s.hidden = s.id !== (classer ? "classer" : `etape-${k}`)));
  $$(".parties button").forEach((b) =>
    b.setAttribute("aria-current", (b.dataset.partie === "classer") === classer ? "page" : "false"));
  $("#etapes").hidden = classer;
  if (!classer) {
    $$("#etapes button").forEach((b) => b.setAttribute("aria-current", b.dataset.etape === String(k) ? "step" : "false"));
    const nav = $("#etapes"), actif = $(`#etapes button[data-etape="${k}"]`);
    nav.scrollLeft = actif.offsetLeft - (nav.clientWidth - actif.clientWidth) / 2;
  }
  history.replaceState(null, "", classer ? "#classer" : `#etape=${k}`);
  if (k === 2) dessinerConvolution();
  if (k === 3) dessinerPooling();
  window.scrollTo({ top: 0 });
}

// ------------------------------------------------------------------ images
function chargerImage(src) {
  if (S.images[src]) return Promise.resolve(S.images[src]);
  return new Promise((ok, ko) => {
    const im = new Image();
    im.onload = () => { S.images[src] = im; ok(im); };
    im.onerror = () => ko(new Error("image illisible"));
    im.src = src;
  });
}
// recadrage identique à l'entraînement : côté court à 256, puis carré central de 224
function vers224(im, canvas) {
  const s = Math.min(im.width, im.height), c = s * 224 / 256;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = "high";
  ctx.drawImage(im, (im.width - c) / 2, (im.height - c) / 2, c, c, 0, 0, 224, 224);
  return ctx.getImageData(0, 0, 224, 224).data;
}

// palette « magma » simplifiée pour les cartes
const STOPS = [[0, 0, 4], [59, 15, 112], [140, 41, 129], [222, 73, 104], [254, 159, 109], [252, 253, 191]];
const LUT = new Uint8ClampedArray(256 * 3);
for (let i = 0; i < 256; i++) {
  const t = i / 255 * (STOPS.length - 1), a = Math.floor(t), b = Math.min(a + 1, STOPS.length - 1), f = t - a;
  for (let c = 0; c < 3; c++) LUT[i * 3 + c] = STOPS[a][c] * (1 - f) + STOPS[b][c] * f;
}
function peindre(canvas, W, H, v255, couleur = true) {
  canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext("2d"), im = ctx.createImageData(W, H);
  for (let i = 0; i < W * H; i++) {
    const v = v255[i];
    if (couleur) { im.data[4 * i] = LUT[3 * v]; im.data[4 * i + 1] = LUT[3 * v + 1]; im.data[4 * i + 2] = LUT[3 * v + 2]; }
    else { im.data[4 * i] = im.data[4 * i + 1] = im.data[4 * i + 2] = v; }
    im.data[4 * i + 3] = 255;
  }
  ctx.putImageData(im, 0, 0);
}

// ------------------------------------------------------------------ choix de la photo (partie cours)
function initChoixPhotos() {
  for (const box of $$("[data-choix-photo]")) {
    box.innerHTML = S.exemples.map((e, i) =>
      `<button type="button" data-photo="${i}" title="${e.label}"><img src="examples/${e.file}" alt="${e.label}"></button>`).join("");
  }
  document.addEventListener("click", (e) => {
    const b = e.target.closest("[data-photo]"); if (!b) return;
    choisirPhoto(+b.dataset.photo);
  });
}
async function choisirPhoto(i) {
  S.photo = i;
  $$("[data-photo]").forEach((b) => b.setAttribute("aria-pressed", +b.dataset.photo === i ? "true" : "false"));
  const im = await chargerImage("examples/" + S.exemples[i].file);
  S.rgba = vers224(im, $("#c224"));
  S.gris = new Float32Array(224 * 224);
  for (let p = 0; p < 224 * 224; p++) S.gris[p] = 0.299 * S.rgba[4 * p] + 0.587 * S.rgba[4 * p + 1] + 0.114 * S.rgba[4 * p + 2];
  majZoom(112, 112);
  dessinerConvolution(); dessinerPooling();
  $("#couches").innerHTML = ""; $("#barre-capot").style.width = "0"; $("#etat-capot").textContent = "";
}

// ------------------------------------------------------------------ étape 1 : pixels
function majZoom(x, y) {
  if (!S.rgba) return;
  const p = 4 * (y * 224 + x), R = S.rgba[p], G = S.rgba[p + 1], B = S.rgba[p + 2];
  $("#pxinfo").innerHTML = `pixel (${x}, ${y}) : <span class="sw" style="background:rgb(${R},${G},${B})"></span> R = ${R} · V = ${G} · B = ${B}`;
  const x0 = Math.min(Math.max(x - 3, 0), 218), y0 = Math.min(Math.max(y - 3, 0), 218);
  let h = "";
  for (let j = y0; j < y0 + 6; j++) {
    h += "<tr>";
    for (let i = x0; i < x0 + 6; i++) {
      const q = 4 * (j * 224 + i), r = S.rgba[q], g = S.rgba[q + 1], b = S.rgba[q + 2];
      const clair = 0.299 * r + 0.587 * g + 0.114 * b > 140;
      h += `<td style="background:rgb(${r},${g},${b});color:${clair ? "#0f172a" : "#fff"}">${r}<br>${g}<br>${b}</td>`;
    }
    h += "</tr>";
  }
  $("#zoom").innerHTML = h;
}
function initPixels() {
  const c = $("#c224");
  const lire = (e) => {
    const r = c.getBoundingClientRect(), t = e.touches ? e.touches[0] : e;
    majZoom(Math.min(223, Math.max(0, Math.floor((t.clientX - r.left) / r.width * 224))),
            Math.min(223, Math.max(0, Math.floor((t.clientY - r.top) / r.height * 224))));
  };
  c.addEventListener("mousemove", lire);
  c.addEventListener("touchmove", (e) => { e.preventDefault(); lire(e); }, { passive: false });
}

// ------------------------------------------------------------------ étape 2 : convolution
const FILTRES = {
  vert: [1, 0, -1, 2, 0, -2, 1, 0, -1],
  horiz: [1, 2, 1, 0, 0, 0, -1, -2, -1],
  contour: [0, 1, 0, 1, -4, 1, 0, 1, 0],
  flou: [1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9],
  nettete: [0, -1, 0, -1, 5, -1, 0, -1, 0],
};
function initConvolution() {
  $("#filtre-grille").innerHTML = Array.from({ length: 9 }, () => `<input type="text" inputmode="decimal">`).join("");
  const remplir = (k) => FILTRES[k].forEach((v, i) => ($$("#filtre-grille input")[i].value = Number.isInteger(v) ? v : "1/9"));
  remplir("vert");
  $("#filtre-choix").onchange = (e) => { if (e.target.value !== "perso") remplir(e.target.value); dessinerConvolution(); };
  $("#filtre-grille").addEventListener("input", () => { $("#filtre-choix").value = "perso"; dessinerConvolution(); });
  $("#conv-sortie").addEventListener("click", (e) => {
    const r = e.target.getBoundingClientRect();
    detailCalcul(Math.floor((e.clientX - r.left) / r.width * 224), Math.floor((e.clientY - r.top) / r.height * 224));
  });
}
function valeurFiltre(s) { if (String(s).includes("/")) { const [a, b] = s.split("/"); return +a / +b; } return parseFloat(String(s).replace(",", ".")) || 0; }
function convolution(g, k) {
  const out = new Float32Array(224 * 224);
  for (let y = 0; y < 224; y++) for (let x = 0; x < 224; x++) {
    let s = 0;
    for (let j = -1; j <= 1; j++) for (let i = -1; i <= 1; i++) {
      const yy = Math.min(223, Math.max(0, y + j)), xx = Math.min(223, Math.max(0, x + i));
      s += g[yy * 224 + xx] * k[(j + 1) * 3 + i + 1];
    }
    out[y * 224 + x] = s;
  }
  return out;
}
function dessinerConvolution() {
  if (!S.gris) return;
  const k = $$("#filtre-grille input").map((i) => valeurFiltre(i.value));
  peindre($("#conv-entree"), 224, 224, Uint8ClampedArray.from(S.gris), false);
  const out = convolution(S.gris, k), somme = k.reduce((a, b) => a + b, 0);
  const v = new Uint8ClampedArray(224 * 224);
  if (Math.abs(somme) < 1e-6) {          // filtre « détecteur » : on affiche la force de la réponse
    const a = Array.from(out, Math.abs).sort((p, q) => p - q), hi = a[Math.floor(a.length * 0.995)] || 1;
    for (let i = 0; i < v.length; i++) v[i] = 255 * Math.min(1, Math.abs(out[i]) / hi);
  } else {
    for (let i = 0; i < v.length; i++) v[i] = out[i];
  }
  S.carte = v; S.k = k; S.sortie = out;
  peindre($("#conv-sortie"), 224, 224, v, Math.abs(somme) < 1e-6);
}
function detailCalcul(x, y) {
  const k = S.k, g = S.gris; let s = 0, cases = "", filtre = "", termes = [];
  for (let j = -1; j <= 1; j++) {
    cases += "<tr>"; filtre += "<tr>";
    for (let i = -1; i <= 1; i++) {
      const yy = Math.min(223, Math.max(0, y + j)), xx = Math.min(223, Math.max(0, x + i));
      const p = Math.round(g[yy * 224 + xx]), w = k[(j + 1) * 3 + i + 1];
      s += g[yy * 224 + xx] * w; cases += `<td>${p}</td>`; filtre += `<td class="f">${fmt(w, Number.isInteger(w) ? 0 : 2)}</td>`;
      if (w !== 0) termes.push(`${p}×${fmt(w, Number.isInteger(w) ? 0 : 2)}`);
    }
    cases += "</tr>"; filtre += "</tr>";
  }
  $("#calcul").innerHTML = `Au point (${x}, ${y}) : <table>${cases}</table> × <table>${filtre}</table> =
    <span class="mono">${termes.join(" + ")} = <strong>${fmt(s, 1)}</strong></span>`;
}

// ------------------------------------------------------------------ étape 3 : pooling
function maxPool(v, n) {
  const m = n / 2, o = new Uint8ClampedArray(m * m);
  for (let y = 0; y < m; y++) for (let x = 0; x < m; x++)
    o[y * m + x] = Math.max(v[2 * y * n + 2 * x], v[2 * y * n + 2 * x + 1], v[(2 * y + 1) * n + 2 * x], v[(2 * y + 1) * n + 2 * x + 1]);
  return o;
}
function dessinerPooling() {
  if (!S.carte) dessinerConvolution();
  if (!S.carte || $("#etape-3").hidden) return;
  const nom = $("#filtre-choix").selectedOptions[0].textContent;
  $("#pool-resume").innerHTML = `Carte de départ : le filtre <strong>${nom}</strong> de l'étape 2, appliqué à la photo choisie.`;
  let v = S.carte, n = 224, h = "";
  const box = $("#pool"); box.innerHTML = "";
  for (let k = 0; k < 4; k++) {
    const f = document.createElement("figure"), c = document.createElement("canvas");
    peindre(c, n, n, v, Math.abs(S.k.reduce((a, b) => a + b, 0)) < 1e-6);
    c.style.width = "200px"; c.style.height = "200px";
    f.appendChild(c);
    f.insertAdjacentHTML("beforeend", `<figcaption>${n} × ${n}${k ? ` (après ${k} max pooling)` : " (carte de départ)"} · ${fmt(n * n)} valeurs</figcaption>`);
    box.appendChild(f);
    if (k < 3) { v = maxPool(v, n); n /= 2; }
  }
}

// ------------------------------------------------------------------ étape 4 : le modèle
function initModele() {
  const L = S.meta.layers, taille = (r) => r.shape.reduce((a, b) => a * b, 1);
  const p = (l) => (l.w ? taille(l.w) + taille(l.b) : 0) +
    ["c1", "c2", "down"].reduce((a, k) => a + (l[k] ? taille(l[k].w) + taille(l[k].b) : 0), 0);
  const blocs = [
    ["entree", "Entrée", "photo 224 × 224 × 3", 0],
    ["conv", "Convolution 7 × 7", "64 filtres → 112 × 112 × 64", p(L[0])],
    ["", "Max pooling", "→ 56 × 56 × 64", 0],
    ["conv", "Bloc 1 (4 convol.)", "→ 56 × 56 × 64", p(L[2]) + p(L[3])],
    ["conv", "Bloc 2 (4 convol.)", "→ 28 × 28 × 128", p(L[4]) + p(L[5])],
    ["conv", "Bloc 3 (4 convol.)", "→ 14 × 14 × 256", p(L[6]) + p(L[7])],
    ["conv", "Bloc 4 (4 convol.)", "→ 7 × 7 × 512", p(L[8]) + p(L[9])],
    ["", "Moyenne", "→ 512 nombres", 0],
    ["dense", "Dense + softmax", `→ ${S.meta.n_classes} probabilités`, p(L[11])],
  ];
  const total = blocs.reduce((a, b) => a + b[3], 0);
  $("#schema").innerHTML = blocs.map(([c, t, d, n], i) =>
    `${i ? '<span class="fleche">→</span>' : ""}<div class="bloc ${c}"><strong>${t}</strong><span>${d}</span>${n ? `<small>${fmt(n)} poids</small>` : ""}</div>`).join("");
  $("#nb-params").textContent = fmt(total);
}
function dessinerFiltres(w, forme) {
  const [N, C, K] = forme, box = $("#filtres64"); box.innerHTML = "";
  for (let n = 0; n < N; n++) {
    const o = n * C * K * K; let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < C * K * K; i++) { lo = Math.min(lo, w[o + i]); hi = Math.max(hi, w[o + i]); }
    const c = document.createElement("canvas"); c.width = K; c.height = K;
    const ctx = c.getContext("2d"), im = ctx.createImageData(K, K);
    for (let q = 0; q < K * K; q++) {
      for (let ch = 0; ch < 3; ch++) im.data[4 * q + ch] = 255 * (w[o + ch * K * K + q] - lo) / (hi - lo + 1e-9);
      im.data[4 * q + 3] = 255;
    }
    ctx.putImageData(im, 0, 0); box.appendChild(c);
  }
}

// ------------------------------------------------------------------ calcul du réseau (via le worker)
function predire(rgba, couches, surProgression) {
  const id = ++S.id;
  return new Promise((ok, ko) => {
    S.attente[id] = { ok, ko, surProgression };
    worker.postMessage({ type: "predire", id, rgba, couches });
  });
}
worker.onmessage = (e) => {
  const m = e.data, a = S.attente[m.id];
  if (m.type === "chargement") {
    if (m.total) $("#barre-chargement").style.width = (100 * m.recu / m.total) + "%";
  } else if (m.type === "pret") {
    S.pret = true;
    $("#statut").textContent = `✔ Modèle ResNet18 chargé : 1 081 espèces`;
    dessinerFiltres(m.filtres, m.forme);
    $("#lancer-capot").disabled = false;
    $("#classer-attente").hidden = true; $("#classer-pret").hidden = false;
  } else if (m.type === "progression" && a && a.surProgression) {
    a.surProgression(m.p, m.nom);
  } else if (m.type === "resultat" && a) {
    delete S.attente[m.id]; a.ok(m);
  } else if (m.type === "erreur") {
    if (a) { delete S.attente[m.id]; a.ko(new Error(m.message)); }
    else $("#statut").textContent = "⚠ Erreur de chargement du modèle : " + m.message;
  }
};

// ------------------------------------------------------------------ étape 5 : sous le capot
const TEXTES = {
  conv1: "1re convolution : 64 filtres de 7 × 7 → bords, couleurs, contrastes",
  layer1: "Bloc 1 : combinaisons de bords → petites textures",
  layer2: "Bloc 2 : textures et motifs (nervures, contours de pétales…)",
  layer3: "Bloc 3 : motifs plus grands, parties d'organes",
  layer4: "Bloc 4 : représentations abstraites, utiles pour séparer les espèces",
};
async function sousLeCapot() {
  if (!S.rgba) return;
  $("#lancer-capot").disabled = true;
  try {
    const r = await predire(S.rgba, true, (p, nom) => {
      $("#barre-capot").style.width = (100 * p) + "%"; $("#etat-capot").textContent = "couche " + nom + "…";
    });
    $("#etat-capot").textContent = `calculé en ${fmt(r.ms / 1000, 1)} s · espèce proposée : ${S.especes[argmax(r.prob)].name} (${pct(r.prob[argmax(r.prob)])})`;
    const box = $("#couches"); box.innerHTML = "";
    for (const c of r.couches) {
      const d = document.createElement("div"); d.className = "couche";
      d.innerHTML = `<h3>${c.nom} — ${c.H} × ${c.W} × ${c.C}</h3><p class="note">${TEXTES[c.nom] || ""}</p><div class="cartes8"></div>`;
      for (const k of c.cartes) { const cv = document.createElement("canvas"); cv.title = "carte n° " + k.c; peindre(cv, c.W, c.H, k.px); d.querySelector(".cartes8").appendChild(cv); }
      box.appendChild(d);
    }
  } catch (err) { $("#etat-capot").textContent = "⚠ " + err.message; }
  $("#lancer-capot").disabled = false;
}

// ------------------------------------------------------------------ partie TP : identifier une plante
const argmax = (p) => { let k = 0; for (let i = 1; i < p.length; i++) if (p[i] > p[k]) k = i; return k; };
const pct = (p) => p >= 0.995 ? "> 99 %" : p < 0.005 ? "< 1 %" : fmt(100 * p) + " %";
const memeEspece = (nom, verite) => nom === verite || nom.startsWith(verite + " ");

function initTP() {
  $("#exemple").innerHTML = '<option value="">(choisir)</option>' +
    S.exemples.map((e, i) => `<option value="${i}">${e.label}</option>`).join("");
  $("#exemple").onchange = async (e) => {
    if (e.target.value === "") return;
    const ex = S.exemples[+e.target.value];
    identifier(await chargerImage("examples/" + ex.file), ex);
  };
  const depot = $("#depot");
  $("#fichier").onchange = (e) => e.target.files[0] && depuisFichier(e.target.files[0]);
  ["dragover", "dragenter"].forEach((t) => depot.addEventListener(t, (e) => { e.preventDefault(); depot.classList.add("survol"); }));
  ["dragleave", "drop"].forEach((t) => depot.addEventListener(t, (e) => { e.preventDefault(); depot.classList.remove("survol"); }));
  depot.addEventListener("drop", (e) => e.dataTransfer.files[0] && depuisFichier(e.dataTransfer.files[0]));
  $("#reveler").onchange = () => S.tp && S.tp.prob && afficherResultat();
  $("#recherche").oninput = (e) => listeEspeces(e.target.value);
  listeEspeces("");
}
async function depuisFichier(f) {
  $("#exemple").value = "";
  identifier(await chargerImage(URL.createObjectURL(f)), null);
}
async function identifier(im, exemple) {
  const cv = $("#apercu-tp"); cv.hidden = false;
  const rgba = vers224(im, cv);
  $("#credit").innerHTML = exemple ? `Photo : ${exemple.credit} · <a href="${exemple.source}" target="_blank" rel="noopener">source</a>` : "Votre photo.";
  S.tp = { exemple };
  $("#resultat-tp").innerHTML = `<p class="note">Calcul en cours dans votre navigateur…</p><div class="progression"><div id="barre-tp"></div></div>`;
  const r = await predire(rgba, false, (p) => { const b = $("#barre-tp"); if (b) b.style.width = (100 * p) + "%"; });
  S.tp.prob = r.prob; S.tp.ms = r.ms;
  afficherResultat();
}
function afficherResultat() {
  const { prob, exemple, ms } = S.tp, ordre = Array.from(prob.keys()).sort((a, b) => prob[b] - prob[a]);
  const top = ordre.slice(0, 5), p1 = prob[top[0]], p2 = prob[top[1]], nom = (i) => S.especes[i].name;
  let v;
  if (p1 >= 0.7) v = `<div class="verdict-p haut">Le modèle est <strong>confiant</strong> : ${pct(p1)} pour la première espèce. Un score élevé n'est pas une preuve : vérifiez.</div>`;
  else if (p1 >= 0.3) v = `<div class="verdict-p moyen">Le modèle <strong>hésite</strong> : ${pct(p1)} pour la première espèce, ${pct(p2)} pour la deuxième.</div>`;
  else v = `<div class="verdict-p bas">Le modèle est <strong>très incertain</strong> (${pct(p1)} au mieux) : l'espèce n'est peut-être pas parmi les 1 081 qu'il connaît.</div>`;
  const reveler = $("#reveler").checked && exemple;
  let h = `<h3>Les 5 espèces les plus probables</h3>${v}<table class="pred">` + top.map((i, r) => {
    const vrai = reveler && memeEspece(nom(i), exemple.truth);
    return `<tr class="${vrai ? "vrai" : ""}"><td class="rang">${r + 1}</td>
      <td class="nom">${nom(i)}${vrai ? " ✔" : ""}<small>genre ${S.especes[i].genus}</small></td>
      <td class="barre-p"><div class="pbar" style="width:${(100 * prob[i] / p1).toFixed(1)}%"></div></td>
      <td class="pval">${pct(prob[i])}</td></tr>`;
  }).join("") + `</table><p class="note">Calculé en ${fmt(ms / 1000, 1)} s dans votre navigateur. Somme des 1 081 probabilités = 100 %.</p>`;
  if (reveler) {
    if (exemple.in_base) {
      const rang = ordre.findIndex((i) => memeEspece(nom(i), exemple.truth)) + 1;
      h += `<div class="verite">Vraie espèce : <strong><em>${exemple.truth}</em></strong> (${exemple.common}).
        Rang dans la réponse du modèle : <strong>${rang}</strong>${rang === 1 ? " : bonne réponse." : " : le modèle s'est trompé."}</div>`;
    } else {
      h += exemple.truth === "—"
        ? `<div class="verite absente">Ce n'est <strong>pas une plante</strong> : c'est ${exemple.common}.
            Le modèle ne connaît que 1 081 espèces de plantes : il ne pouvait pas avoir raison, et il a pourtant répondu.</div>`
        : `<div class="verite absente">Vraie espèce : <strong>${exemple.common}</strong> (<em>${exemple.truth}</em>${exemple.note ? ", " + exemple.note : ""}).
            Elle <strong>ne fait pas partie</strong> des 1 081 espèces du modèle : il ne pouvait pas la trouver, et il a pourtant répondu.</div>`;
    }
  } else if ($("#reveler").checked && !exemple) {
    h += `<p class="note">Vraie espèce inconnue : c'est votre photo. À vous de vérifier avec une flore ou une clé de détermination.</p>`;
  }
  $("#resultat-tp").innerHTML = h;
}
function listeEspeces(q) {
  q = q.trim().toLowerCase();
  const l = S.especes.filter((s) => !q || s.name.toLowerCase().includes(q));
  $("#nb-trouves").textContent = q ? `${l.length} résultat(s)` : `${S.especes.length} espèces`;
  $("#liste-especes").innerHTML = l.map((s) => `<div>${s.name}</div>`).join("");
}

// ------------------------------------------------------------------ démarrage
async function demarrer() {
  document.addEventListener("click", (e) => {
    const c = e.target.closest("[data-aller], [data-partie], #etapes button");
    if (!c) return;
    e.preventDefault();
    const { aller, partie, etape } = c.dataset;
    const k = partie ? (partie === "classer" ? "classer" : S.etape) : (aller ?? etape);
    allerA(k === "classer" ? k : Number(k));
  });
  worker.postMessage({ type: "charger", base: new URL("model/", location.href).href });
  [S.exemples, S.especes, S.meta] = await Promise.all(
    ["examples/examples.json", "model/species.json", "model/model.json"].map((u) => fetch(u).then((r) => r.json())));
  initChoixPhotos(); initPixels(); initConvolution(); initModele(); initTP();
  $("#lancer-capot").onclick = sousLeCapot;
  const depart = location.hash === "#classer" ? "classer" : Number((location.hash.match(/etape[=-](\d)/) || [])[1]) || 1;
  allerA(depart === "classer" || (depart >= 1 && depart <= 6) ? depart : 1);
  await choisirPhoto(11);   // le dahlia, fil rouge des vidéos
}
demarrer();
