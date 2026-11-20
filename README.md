# TP 3 – Détection de contours (Sobel, Laplace, Canny)

Comparaison quantitative de trois détecteurs de contours avec OpenCV (C++), sur les 40 images de la base
de contours de l'université de Groningue (RUG), et **seuils automatiques** calculés à partir de chaque image.

## Contenu

```
TP_contours/
├── CMakeLists.txt
├── src/
│   ├── detectors.hpp / .cpp   Sobel, Laplace (passages par zéro), Canny + seuils automatiques
│   ├── scoring.hpp / .cpp     5 grandeurs, P, TFP, TFN, image de comparaison couleur
│   ├── analysis.hpp / .cpp    analyse statistique (corrélations, meilleur seuil, validation croisée)
│   └── main.cpp               mode interactif, batch, sweep, analyse
├── data/
│   ├── images/                40 images .pgm
│   └── reference/             contours de référence *_gt_binary.pgm
└── results/                   créé par le programme
```

## Compilation (Linux)

```bash
sudo apt install build-essential cmake libopencv-dev
cmake -B build
cmake --build build
```

## Utilisation

| Commande | Rôle |
|---|---|
| `./build/edge_eval` | Mode interactif : trackbars, mesures P / TFP / TFN en direct |
| `./build/edge_eval batch [manual\|global\|local\|all]` | Traite les 40 images avec les seuils manuels ou automatiques |
| `./build/edge_eval sweep` | Fait varier les paramètres et calcule les performances moyennes |
| `./build/edge_eval analyse` | Analyse statistique : texture des images, corrélations, meilleur seuil par image, règles automatiques |

**Touches du mode interactif** (fenêtre « Edges ») : `d` / `a` image suivante / précédente,
`o` carte binaire ou superposition couleur, `w` sauvegarde dans `results/snapshots/`, `Échap` quitter.
Couleurs : vert = contour correct, rouge = faux positif, bleu = contour manqué.

**Trackbar « mode »** de chaque détecteur :
- `0` manuel : seuil réglé à la main ;
- `1` automatique global : seuil = gain × gradient moyen de l'image ;
- `2` automatique local (Sobel, Canny) : la réponse est d'abord diminuée là où il y a beaucoup de
  gradient autour (texture), puis seuillée comme en mode global.

## Évaluation

Tolérance de voisinage 3×3 (dilatation). Les contours corrects sont comptés côté référence, donc
`faux_négatifs = contours_référence − contours_corrects`.

- P = corrects / (corrects + FP + FN)
- TFP = FP / (corrects + FP + FN)
- TFN = FN / (corrects + FP + FN)

## Résultats produits

| Dossier | Contenu |
|---|---|
| `results/manual/`, `results/global/`, `results/local/` | `scores.csv` (5 grandeurs + 3 mesures par image, ligne de moyennes), cartes de contours, images de comparaison |
| `results/sweep/` | Performances moyennes en fonction des paramètres |
| `results/analysis/` | `analysis.txt`, `features.csv`, `best_thresholds.csv` |
| `results/snapshots/` | Captures du mode interactif (touche `w`) |

## Données

Contour Detection Database, université de Groningue :
http://www.cs.rug.nl/~imaging/databases/contour_database/contour_database.html
