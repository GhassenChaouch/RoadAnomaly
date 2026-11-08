# Comparaison des modèles : détection d'anomalies routières (RoadAnomaly)

*Tous les chiffres proviennent de l'exécution du notebook `BO1_RoadAnomaly_VERSION_FINALE_AMELIOREE_1.ipynb` (CPU).*

## 1. Objectif et protocole

Comparer trois modèles de segmentation d'anomalies sur le dataset RoadAnomaly (Lis et al., ICCV 2019) : 60 images 1280×720 avec masques pixel-level (fond = 0, anomalie = 2).

| Élément | Choix |
|---|---|
| Découpage | 38 train / 10 validation / 12 test, séparation faite **avant** l'augmentation, seed 42 |
| Part de pixels anormaux | train 10,1 % · validation 8,4 % · test 10,2 % |
| Prétraitement | images redimensionnées en 224×224, masques en plus proche voisin |
| Augmentation (train seulement) | flip horizontal, rotation ±7°, luminosité / contraste |
| Loss (FCN) | entropie croisée pondérée (poids de la classe « anomalie » calculé sur le train, environ 9) |
| Sélection du modèle | meilleur Dice sur la validation ; le test sert uniquement à l'évaluation finale |
| Métriques | précision, rappel, Dice, IoU (pixel-level, agrégées sur tout le test), temps d'inférence |

## 2. Les trois modèles

| Modèle | Description | Initialisation | Entraînement réalisé |
|---|---|---|---|
| **FCN scratch** | FCN multi-échelles à 3 blocs convolutifs (58 156 paramètres) | aléatoire | 12 epochs (limite atteinte), AdamW lr 2e-4 |
| **Mask R-CNN scratch** | Mask R-CNN ResNet50-FPN, segmentation d'instances ; masques agrégés en carte binaire | aléatoire | 3 epochs, batch 1, AdamW lr 1e-4 |
| **FCN ResNet50 (ImageNet)** | FCN avec backbone ResNet50 pré-entraîné (transfer learning), backbone gelé | poids ImageNet | 7 epochs (early stopping, meilleur epoch : 3), AdamW lr 5e-5 |

## 3. Résultats sur le test (12 images)

| Modèle | Précision | Rappel | Dice (= F1) | IoU | Inférence (CPU) |
|---|---|---|---|---|---|
| **FCN ResNet50 (ImageNet)** | 0,398 | 0,740 | **0,517** | **0,349** | 210 ms |
| FCN scratch | 0,215 | 0,497 | 0,300 | 0,176 | 10 ms |
| Mask R-CNN scratch | 0,000 | 0,000 | 0,000 | 0,000 | 402 ms |

![Courbes](roadanomaly_results/courbes.png)
*Courbes d'apprentissage (extraites de la section 8 du notebook). Le Mask R-CNN reste à 0 en Dice et en rappel ; le FCN scratch progresse lentement ; le FCN ResNet50 apprend vite mais oscille.*

## 4. Analyse

### 4.1 Meilleur modèle : FCN ResNet50 pré-entraîné
- Il obtient le meilleur Dice (0,52) et le meilleur IoU (0,35), soit environ **+0,22 de Dice** par rapport au FCN scratch.
- Visuellement, il localise bien les animaux (par exemple le troupeau de moutons `animals06` : IoU 0,80, précision 0,89, rappel 0,89), même si ses contours sont grossiers et qu'il produit des fausses alertes sur la route ou les véhicules.
- Sa **précision reste basse (0,40)** : plus de la moitié des pixels signalés sont de fausses alertes.
- Pourquoi il gagne : le backbone ImageNet apporte des caractéristiques visuelles déjà apprises, ce qui est décisif quand on n'a que 38 images d'entraînement.

### 4.2 FCN scratch : modèle rapide mais encore sous-entraîné
- Il est **20 fois plus rapide** (10 ms contre 210 ms), mais deux fois moins précis.
- Il n'a pas convergé : sa loss part de 0,693 (le niveau du hasard) et n'atteint que 0,57, et son Dice de validation monte encore à la dernière epoch (0,16 → 0,26). Il s'est arrêté parce que la limite de 12 epochs était atteinte.
- Sur les 3 images affichées, il semble signaler les zones sombres (ombres, végétation) et rater les animaux clairs. **C'est une hypothèse issue de 3 images**, à confirmer.
- Images les plus difficiles : `animals06` (rappel 0,19), `obstacles12_rocks4` (Dice 0 : petits cailloux dispersés), `animals01` (Dice 0,05).

### 4.3 Mask R-CNN scratch : échec
- Il ne prédit **aucune région** au-dessus du seuil de score (0,35) : zéro pixel détecté, donc zéro faux positif.
- Sa loss d'entraînement baisse pourtant (4,10 → 1,84 → 1,51) : il apprend un peu, mais très loin de ce qu'il faut.
- Causes probables : réseau de plusieurs dizaines de millions de paramètres entraîné **sans pré-entraînement**, sur 38 images, avec seulement 3 epochs (environ 114 mises à jour) ; de plus 14 images sur 60 n'ont pas de masque d'instances (remplacé par des composantes connexes).
- **Précision méthodologique** : comme le Dice de validation est resté à 0, le seul checkpoint sauvegardé est celui de l'**epoch 1**. C'est donc ce modèle qui a été évalué au test.

### 4.4 Compromis vitesse / précision

| | FCN scratch | FCN ResNet50 | Mask R-CNN |
|---|---|---|---|
| Qualité (Dice) | 0,30 | **0,52** | 0 |
| Vitesse (CPU) | **10 ms** | 210 ms | 402 ms |

Pour une application embarquée temps réel, le FCN scratch est le plus léger, mais sa qualité actuelle est insuffisante. Le FCN ResNet50 offre le meilleur compromis dans cette étude.

## 5. Limites (à mentionner explicitement)

1. **Test de 12 images seulement**, sans intervalle d'incertitude. D'après le découpage utilisé, il ne contient **aucune image de la catégorie `lost`** (les plus petites anomalies), ce qui rend les scores plutôt optimistes.
2. **Validation bruitée** (10 images) : le Dice de validation du ResNet50 oscille entre 0,30 et 0,66 d'une epoch à l'autre. Son meilleur Dice de validation (0,66) dépasse son Dice de test (0,52), ce qui est typique d'une sélection faite sur très peu d'images.
3. **Comparaison non équitable** : les trois modèles diffèrent à la fois par l'initialisation, la taille (58 000 paramètres contre des dizaines de millions), le nombre d'epochs et le learning rate. On peut conclure que le pré-entraînement aide, mais **pas** que l'architecture scratch est mauvaise en soi.
4. **Résolution 224×224 carrée** : le ratio 16:9 est déformé et les petites anomalies deviennent très petites (la plus petite fait 188 pixels).
5. Temps d'inférence mesurés sur CPU, sur une seule machine.

## 6. Améliorations possibles

- Entraîner le FCN scratch plus longtemps (60 à 100 epochs) pour vérifier s'il rattrape son retard.
- Utiliser un U-Net léger comme second modèle from scratch, plus adapté aux petits jeux de données que Mask R-CNN.
- Normaliser les images avec les statistiques ImageNet pour le modèle pré-entraîné, et garder les BatchNorm du backbone gelé en mode évaluation.
- Utiliser une validation croisée (plusieurs plis) et rapporter moyenne ± écart-type, avec un découpage stratifié par catégorie.
- Choisir le seuil de décision sur la validation.
- Conserver le ratio 16:9 (par exemple 256×512).

## 7. Conclusion

Dans cette étude, le **FCN avec backbone ResNet50 pré-entraîné** est le meilleur modèle (Dice 0,52, IoU 0,35), grâce au transfer learning, au prix d'un temps d'inférence 20 fois supérieur au FCN scratch. Le FCN scratch, rapide mais sous-entraîné, n'atteint que 0,30 de Dice. Le Mask R-CNN entraîné de zéro sur 38 images ne produit aucune détection. Ces résultats sont **exploratoires** : avec 12 images de test et des budgets d'entraînement différents, ils montrent l'intérêt du pré-entraînement sans permettre de classer définitivement les architectures, et ils ne démontrent pas une aptitude à la conduite réelle.
