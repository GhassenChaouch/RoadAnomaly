# Rapport individuel : Yassine Hammouda
## BO3 : détecter des objets jamais vus, sans étiquettes d'anomalies

Branche : `yassine` · Comparaison détaillée : `comparaison_Yassine.md`

## 1. Objectif
Repérer les anomalies routières **sans étiquette d'anomalie à l'entraînement**. Critères : AUROC et précision moyenne (AP) au niveau pixel, avec 2 modèles techniquement différents. Le résultat attendu est plus faible que le supervisé, et c'est la discussion à mener.

## 2. Données et protocole
- RoadAnomaly, `split.csv` commun (36 train / 12 validation / 12 test), images en 512×288.
- Les 60 images contiennent toutes une anomalie : l'entraînement se fait sur des images contaminées, sans masque.
- Augmentation (train) : flip, rotation ±10°, zoom, luminosité / contraste / saturation.
- 12 runs : 2 modèles × avec/sans augmentation × 3 seeds.

## 3. Modèles
Le score d'un pixel est son erreur de reconstruction.

| Modèle | Description | Lien avec le cours |
|---|---|---|
| **M1** | Autoencodeur convolutif débruitant sur les pixels, appris depuis zéro | Chap. 5 |
| **M2** | Autoencodeur sur les features d'un ResNet18 ImageNet gelé | Assemblage de ResNet (chap. 4) et d'un autoencodeur profond (chap. 5) |

## 4. Résultats (test, moyenne de 3 seeds, avec augmentation)

| | M1 | M2 | Référence de position* |
|---|---|---|---|
| AUROC pixel | 0,679 | **0,781** | 0,816 |
| AP pixel | 0,129 | **0,192** | 0,217 |
| Dice | 0,187 | **0,297** | 0,325 |
| Inférence CPU | **30 ms** | 98 ms | |

*La référence de position utilise les masques du train : c'est un point de repère, pas un concurrent non supervisé.

![Courbes ROC et précision-rappel](outputs_bo3/courbes_roc_pr.png)

## 5. Conclusions
- **M2 est meilleur que M1** : +0,10 d'AUROC (IC 95 % : +0,058 à +0,157) et +0,06 d'AP (IC 95 % : +0,023 à +0,146).
- **L'augmentation n'apporte aucun gain mesurable** pour les deux modèles.
- **Aucun des deux modèles ne dépasse la référence de position** en score global.
- **Le supervisé reste bien meilleur** : Dice 0,297 pour M2 contre 0,832 pour le U-Net pré-entraîné d'Ahmed, sur le même test.

## 6. Limites
Seulement 12 images de test, aucune image normale pure, annotations partielles, et méthode plus simple que la resynthèse du papier de référence. Ce prototype n'est pas utilisable pour une alerte de conduite réelle.

## 7. Reproduire
```powershell
python BO3_train_eval.py --model feat --aug 1 --seed 42 --epochs 30 --out outputs_bo3/test_perso
```
`BO3_analysis.py` produit la comparaison après les 12 runs (`--model {feat,dcae} --aug {0,1} --seed {42,43,44} --epochs 30`).
