# Identifier une plante avec un réseau convolutif (CNN)

Site pédagogique du **chapitre 4 (BOE)** de la Compétence Transversale IA — L3 Sciences de la Vie, Université Côte d'Azur.
Même présentation que l'outil ECG-LSTM du chapitre 3.

- **🧠 Comprendre le modèle (cours)** : 1. Données (pixels) · 2. Convolution (filtres modifiables, calcul détaillé) ·
  3. Pooling · 4. Modèle ResNet18 (schéma, 64 filtres appris) · 5. Sous le capot (cartes des couches) · 6. Questions.
- **🌿 Identifier une plante (TP)** : déposer une photo ou choisir l'une des 13 photos d'exemple → 5 espèces les plus probables,
  case « Révéler la vraie espèce », liste des 1 081 espèces connues.

Tout le calcul se fait **dans le navigateur** : aucune installation, aucune photo envoyée sur internet, pas de serveur, pas de quota.

## Mettre en ligne sur GitHub Pages

1. Créer un dépôt GitHub (par ex. `cnn-plantes`) et y copier **tout le contenu de ce dossier** (le site est à la racine).
2. `git add . && git commit -m "Site CNN plantes" && git push`
3. Sur GitHub : *Settings → Pages → Build and deployment → Source : Deploy from a branch*, branche `main`, dossier `/ (root)`.
4. Le site est en ligne après 1 à 2 minutes : `https://<compte>.github.io/cnn-plantes/` ; lien direct vers le TP : `…/cnn-plantes/#classer`.

Le fichier `model/weights.bin` fait 23 Mo (limite GitHub : 100 Mo par fichier) : pas besoin de Git LFS.
Le fichier `.nojekyll` évite tout traitement Jekyll.

Tester en local : `python3 -m http.server 8000` dans ce dossier, puis ouvrir http://localhost:8000
(ouvrir `index.html` directement en double-cliquant ne marche pas : le navigateur bloque le chargement du modèle).

## Contenu

| Fichier | Rôle |
|---|---|
| `index.html`, `style.css` | page et style (style.css identique au site ECG) ; `plantes.css` : compléments |
| `js/app.js` | interface |
| `js/cnn.js` | **le réseau ResNet18 écrit à la main** (convolution, ReLU, max pooling, moyenne, dense), sans librairie |
| `js/worker.js` | exécute le réseau en arrière-plan |
| `model/` | poids (float16, BatchNorm fusionnée), description des couches, noms des 1 081 espèces |
| `examples/` | 13 photos d'exemple + `examples.json` (vraie espèce, licence, auteur, source) |
| `outils/` | scripts de conversion et de test (Python, sans PyTorch) |

## Le modèle

- ResNet18 pré-entraîné sur ImageNet puis ré-entraîné sur **Pl@ntNet-300K** (306 146 photos, 1 081 espèces) :
  [cpoisson/plantnet300k-resnet18](https://huggingface.co/cpoisson/plantnet300k-resnet18), licence Apache-2.0.
  Justesse annoncée : top-1 75,8 %, top-5 94,0 %. **Modèle indépendant**, non produit par l'équipe Pl@ntNet.
- Prétraitement identique à l'entraînement : côté court à 256 px, recadrage central 224 × 224, normalisation ImageNet.
- Vérification : le moteur JavaScript reproduit l'export ONNX officiel du modèle (corrélation des scores 0,9999999,
  écart maximal sur les probabilités < 0,1 %, même espèce prédite).
- Données : Garcin et al., « Pl@ntNet-300K: a plant image dataset with high label ambiguity and a long-tailed distribution », NeurIPS 2021.

Régénérer les poids : `python3 outils/convert.py plantnet_resnet18.pth plantnet300K_species_id_2_name.json model`
(fichiers téléchargeables sur la page Hugging Face ci-dessus).

## Photos d'exemple : réponses du site (corrigé)

Valeurs affichées par le site dans le navigateur (vérifiées sur la version en ligne).

| Photo | Vraie espèce | Dans la base ? | 1re proposition du modèle | Rang de la bonne réponse |
|---|---|---|---|---|
| 1 | *Papaver rhoeas* (coquelicot) | oui | *Papaver rhoeas*, > 99 % ✔ | 1 |
| 2 | *Papaver dubium* (pavot douteux) | oui | *Papaver dubium*, 97 % ✔ | 1 |
| 3 | *Lavandula stoechas* (lavande stéchade) | oui | *Lavandula stoechas*, > 99 % ✔ | 1 |
| 4 | *Cirsium vulgare* (cirse commun) | oui | *Cirsium vulgare*, > 99 % ✔ | 1 |
| 5 | *Fragaria vesca* (fraisier des bois, fruit) | oui | *Fragaria vesca*, 64 % ✔ | 1 |
| 6 | *Hypericum perforatum* (millepertuis perforé) | oui | *Hypericum maculatum*, 85 % ✘ | 2 |
| 7 | *Lamium purpureum* (lamier pourpre) | oui | *Lamium purpureum*, > 99 % ✔ | 1 |
| 8 | *Calendula officinalis* (souci officinal) | oui | *Calendula officinalis*, 98 % ✔ | 1 |
| 9 | *Tagetes patula* (œillet d'Inde) | oui | *Tagetes erecta*, > 99 % ✘ : confiant **et** faux | 2 |
| 10 | *Taraxacum officinale* (pissenlit) | **non** | *Calendula officinalis*, 56 % | — |
| 11 | *Bellis perennis* (pâquerette) | **non** | *Calendula officinalis*, 80 % | — |
| 12 | Dahlia (plante cultivée) | **non** | *Tagetes patula*, 89 % | — |
| 13 | un chat | **non** | *Lagenaria siceraria* (calebasse), 64 % | — |

Crédits photos : voir `examples/examples.json` (Wikimedia Commons, CC BY / CC BY-SA ; dahlia : vultilion, CC BY 2.0 ; chat : S. van der Walt, CC0).
