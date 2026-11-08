# Rapport individuel — BO1 : Sécurité et aide à la conduite
## Détection et segmentation des anomalies routières par Deep Learning

**Projet collectif :** RoadAnomaly  
**Périmètre de ce rapport :** contribution individuelle — trois architectures alternatives  
**Sources :** `BO1_RoadAnomaly_VERSION_FINALE_AMELIOREE(2).ipynb` et `comparaison_modeles (1).md`  
**Statut :** résultats expérimentaux obtenus sur CPU ; intégration finale de l'équipe à poursuivre.

---

## 1. Contexte et objectif métier

La perception de l'environnement est une composante essentielle des systèmes d'aide à la conduite. Des objets inhabituels présents sur la chaussée (animaux, obstacles ou autres éléments non attendus) peuvent représenter des situations à risque.

**Use case :** analyser une image issue d'une caméra routière, **localiser au niveau du pixel les régions anormales** et produire une **alerte visuelle indicative** lorsque la proportion de pixels détectés comme anormaux dépasse un seuil de démonstration.

Le système étudié ne calcule pas la distance aux obstacles, n'est pas un dispositif de freinage automatique et n'est pas validé pour un véhicule réel.

### Objectifs techniques
- Étudier le dataset RoadAnomaly et les annotations disponibles.
- Préparer les images et les masques de segmentation.
- Développer et entraîner trois architectures distinctes des U-Net déjà étudiés par un autre membre de l'équipe.
- Comparer précision, rappel, Dice, IoU et temps d'inférence.
- Examiner les faux négatifs et la pertinence d'une alerte visuelle.
- Documenter les limites expérimentales avant l'intégration au projet collectif.

## 2. Dataset et exploration

Le travail s'appuie sur **RoadAnomaly**, avec **60 images routières** de résolution native **1280 × 720 pixels** et leurs masques de segmentation sémantique.

| Élément | Description |
|---|---|
| Images | 60 fichiers `.webp` |
| Masques sémantiques | 60 fichiers `labels_semantic.png` |
| Valeur du fond | `0` |
| Valeur de l'anomalie | `2` |
| Masques d'instances disponibles | 46 |
| Masques d'instances absents | 14 |
| Tâche principale | Segmentation binaire : fond / anomalie |

L'exploration comprend le contrôle des dimensions, des valeurs de masques, des fichiers illisibles, des doublons exacts et des images visuellement proches. Le notebook illustre aussi les images, les masques et leurs superpositions.

## 3. Préparation des données

### 3.1 Découpage expérimental individuel

Le notebook fourni utilise une séparation par groupes d'images proches, **avant l'augmentation**, avec `seed = 42`.

| Ensemble | Images | Utilisation |
|---|---:|---|
| Entraînement | 38 | Apprentissage des paramètres |
| Validation | 10 | Choix des checkpoints |
| Test | 12 | Évaluation finale |
| **Total** | **60** | |

**Important pour l'équipe :** ce découpage **38/10/12** correspond aux résultats de ce rapport. Le projet d'Ahmed utilise un protocole **36/12/12**. Les métriques de ce rapport ne doivent **pas** être présentées comme une comparaison directe avec celles des autres membres. Une réévaluation sur un split commun est nécessaire.

### 3.2 Prétraitement

- Conversion des images en RGB.
- Redimensionnement à **224 × 224** pixels.
- Redimensionnement des masques par interpolation au **plus proche voisin**, afin de préserver les étiquettes.
- Transformation du masque sémantique en cible binaire : `anomalie = (masque == 2)`.
- Mise à l'échelle des valeurs RGB dans `[0, 1]` dans ce notebook.
- Utilisation d'une **entropie croisée pondérée** pour les FCN ; le poids de la classe anomalie est estimé exclusivement sur le train.

**Limite :** le passage d'une image 16:9 à 224 × 224 modifie les proportions et réduit la visibilité de petites anomalies.

### 3.3 Data Augmentation

Les transformations sont appliquées **uniquement à l'entraînement**, à la volée :

| Transformation | Application |
|---|---|
| Retournement horizontal | Probabilité de 50 % |
| Rotation | Jusqu'à ±7°, selon une probabilité de 50 % |
| Luminosité / contraste | Variations aléatoires modérées |

Les transformations géométriques sont appliquées conjointement à l'image et au masque. Le notebook présente plusieurs versions augmentées d'une même image et vérifie le déséquilibre des classes.

La comparaison quantitative **avec/sans augmentation** est prévue dans le notebook, mais **n'a pas été exécutée** (`RUN_AUGMENTATION_ABLATION = False`). On ne peut donc pas encore chiffrer le gain apporté par l'augmentation.

## 4. Modèles développés

### 4.1 Modèle 1 — FCN Multi-Scale From Scratch

Un réseau **Fully Convolutional Network (FCN)** léger, entraîné à partir de poids aléatoires. Il comprend trois blocs convolutionnels et combine des cartes de scores à plusieurs échelles.

- **Paramètres :** 58 156.
- **Initialisation :** aléatoire.
- **Optimiseur :** AdamW, learning rate `2e-4`.
- **Entraînement :** 12 époques.
- **Intérêt :** faible coût de calcul et inférence rapide.

### 4.2 Modèle 2 — Mask R-CNN From Scratch

Architecture de **segmentation d'instances** avec backbone ResNet50-FPN, entraînée sans poids pré-entraînés. Les masques d'instances prédits sont réunis pour produire une carte binaire d'anomalies.

- **Initialisation :** aléatoire.
- **Optimiseur :** AdamW, learning rate `1e-4`.
- **Entraînement :** 3 époques, batch de 1.
- **Particularité :** les masques d'instances manquants sont remplacés par des composantes connexes du masque sémantique.

**Réserve méthodologique :** ces pseudo-instances ne sont pas équivalentes à des annotations manuelles. L'association entre les instances fournies et la classe « anomalie » doit également être contrôlée. Le résultat de Mask R-CNN ne peut pas être interprété comme une validation de cette méthode d'annotation.

### 4.3 Modèle 3 — FCN ResNet50 pré-entraîné sur ImageNet

Architecture FCN avec backbone **ResNet50 initialisé par des poids ImageNet**. Le backbone est gelé pendant l'entraînement présenté ; la tête de segmentation est adaptée à deux classes.

- **Initialisation :** transfert d'apprentissage ImageNet.
- **Optimiseur :** AdamW, learning rate `5e-5`.
- **Entraînement :** 7 époques, arrêt anticipé ; meilleur checkpoint à l'époque 3.
- **Intérêt :** exploiter des représentations visuelles préalablement apprises malgré un dataset limité.

## 5. Entraînement et protocole d'évaluation

Les checkpoints FCN sont sélectionnés selon le **meilleur Dice de validation**. Les modèles sont évalués sur le test à l'aide des métriques suivantes :

- **Précision :** `TP / (TP + FP)` — part des pixels prédits comme anomalies qui sont corrects.
- **Rappel (Recall) :** `TP / (TP + FN)` — part des pixels anormaux réellement retrouvés.
- **Dice :** `2TP / (2TP + FP + FN)`.
- **IoU :** `TP / (TP + FP + FN)`.
- **Temps d'inférence :** durée moyenne de prédiction par image, mesurée sur CPU.

Les métriques ci-dessous sont calculées **au niveau du pixel, agrégées sur les 12 images de test**. Elles ne sont pas des moyennes de scores par image.

## 6. Résultats expérimentaux

| Modèle | Précision | Recall | Dice | IoU | Inférence CPU |
|---|---:|---:|---:|---:|---:|
| **FCN ResNet50 (ImageNet)** | **0,398** | **0,740** | **0,517** | **0,349** | 210 ms/image |
| FCN Multi-Scale Scratch | 0,215 | 0,497 | 0,300 | 0,176 | **10 ms/image** |
| Mask R-CNN Scratch | 0,000 | 0,000 | 0,000 | 0,000 | 402 ms/image |

*Source des chiffres : `comparaison_modeles (1).md`, issu de l'exécution du notebook sur CPU. Ces chiffres appartiennent au split individuel 38/10/12 et ne sont pas des résultats du protocole collectif 36/12/12.*

### 6.1 Analyse du FCN ResNet50

Le **FCN ResNet50 pré-entraîné** est le meilleur modèle de cette expérimentation pour la qualité de segmentation : **Dice 0,517**, **IoU 0,349** et **Recall 0,740**.

Son rappel signifie qu'environ 74 % des pixels anormaux du test ont été retrouvés. En revanche, sa précision de 0,398 traduit un nombre important de faux positifs. Le modèle peut donc générer des alertes inutiles.

Le transfert d'apprentissage apparaît bénéfique dans ce contexte de données limitées, sans permettre de conclure que cette architecture serait systématiquement supérieure sur d'autres données.

### 6.2 Analyse du FCN Scratch

Le FCN léger est le plus rapide : **10 ms/image** contre **210 ms/image** pour le FCN ResNet50 sur le CPU testé.

Sa qualité reste inférieure (**Dice 0,300**). Sa perte d'entraînement diminue au cours des 12 époques, tandis que son Dice de validation progresse encore en fin d'entraînement. Un entraînement plus long pourrait être étudié, mais aucun gain supplémentaire n'a été mesuré ici.

### 6.3 Analyse du Mask R-CNN Scratch

Le Mask R-CNN obtient **0 sur les quatre métriques** avec le seuil de détection utilisé. Il ne produit pas de région retenue au-dessus du seuil de score de **0,35**.

La perte d'entraînement diminue pourtant de **4,10 à 1,51** sur trois époques. Cela indique une optimisation de la fonction de perte, **pas** une détection réussie.

Plusieurs facteurs peuvent contribuer à cet échec : modèle complexe initialisé aléatoirement, seulement 38 images d'entraînement, trois époques et qualité limitée des cibles d'instances. Ces explications restent des hypothèses ; un diagnostic des scores et des prédictions serait nécessaire.

## 7. Visualisations, interprétabilité et analyse d'erreurs

Le notebook inclut :

- Des exemples **image originale / masque réel / masque prédit**.
- Des superpositions colorées pour localiser les anomalies.
- Des courbes de **loss d'entraînement, Dice et Recall de validation**.
- Des tableaux d'erreurs par image, avec **faux positifs et faux négatifs**.
- Une fonction **Grad-CAM** pour analyser les zones visuelles contribuant à la réponse des FCN.

La présence de ces outils dans le code ne signifie pas que chaque illustration XAI a été exécutée ou validée sur toutes les images.

### Démonstration d'alerte visuelle

La fonction `show_alert(...)` affiche en couleur les pixels prédits comme anomalies. Une alerte indicative est déclenchée si la fraction de pixels prédits dépasse un seuil fixé pour la démonstration (par défaut **1 %**).

Ce seuil **n'est pas un seuil de danger validé**. Il ne renseigne ni sur la distance à l'objet, ni sur sa vitesse, ni sur le risque de collision.

**Aucune application de déploiement ni alerte sonore n'est revendiquée comme livrable de ce travail individuel.** L'intégration applicative pourra être réalisée ultérieurement par l'équipe.

## 8. Limites de l'étude

1. **Échantillon restreint :** seulement 60 images, dont 12 pour le test.
2. **Généralisation non démontrée :** les performances sur d'autres routes, caméras ou conditions météo sont inconnues.
3. **Comparaison non contrôlée :** modèles, nombres d'époques, initialisations et tailles diffèrent.
4. **Résolution carrée :** les proportions originales sont déformées et les petites anomalies peuvent disparaître.
5. **Déséquilibre des classes :** les pixels d'anomalie sont minoritaires.
6. **Mask R-CNN :** les annotations d'instances et les seuils nécessitent une vérification approfondie.
7. **Test limité :** le document de comparaison indique que certaines catégories d'anomalies, notamment `lost`, ne sont pas représentées dans ce test.
8. **Mesure CPU :** les temps d'inférence ne prouvent pas une compatibilité temps réel sur matériel embarqué.
9. **Ablation non exécutée :** aucun bénéfice quantifié de la Data Augmentation n'est encore disponible.
10. **Absence de validation de sécurité :** ce prototype ne peut pas être utilisé pour prendre des décisions de conduite.

## 9. Intégration dans le projet collectif

Ce rapport concerne **uniquement ma contribution BO1** : les trois modèles FCN Multi-Scale Scratch, Mask R-CNN Scratch et FCN ResNet50 ImageNet, ainsi que leur exploration, entraînement, évaluation et visualisation.

Les autres membres du groupe développent leurs propres modules. Les notebooks et résultats existants doivent rester inchangés lors de l'ajout de ce travail.

Pour une comparaison finale cohérente, l'équipe devra :

1. Définir un **split commun** (le projet d'Ahmed utilise actuellement **36 train / 12 validation / 12 test**).
2. Réentraîner ou réévaluer les modèles avec les **mêmes images de test**.
3. Employer les **mêmes définitions de métriques** et préciser si elles sont micro- ou macro-agrégées.
4. Déclarer les différences de résolution, normalisation, initialisation et budget d'entraînement.
5. Ajouter les résultats individuels au tableau collectif **sans réutiliser les scores 38/10/12 comme s'ils provenaient du split commun**.

L'intégration de ce module ne signifie pas que le projet global est terminé.

## 10. Perspectives

- Diagnostiquer les scores de confiance de Mask R-CNN et contrôler les pseudo-instances.
- Entraîner le FCN Scratch plus longtemps et mesurer l'évolution de la validation.
- Exécuter l'ablation **avec / sans augmentation**.
- Tester un prétraitement qui conserve le rapport d'aspect.
- Étudier plusieurs découpages ou une validation croisée.
- Définir les seuils d'alerte sur la validation et analyser particulièrement les faux négatifs.
- Réaliser la comparaison collective une fois les autres modules terminés.

## 11. Conclusion

Cette contribution BO1 a permis d'implémenter et de comparer **trois architectures de Deep Learning** pour la segmentation des anomalies routières sur RoadAnomaly.

Dans l'expérience individuelle réalisée, le **FCN ResNet50 pré-entraîné sur ImageNet** obtient les meilleurs résultats (**Recall 0,740 ; Dice 0,517 ; IoU 0,349**). Le **FCN Multi-Scale Scratch** est beaucoup plus rapide mais moins performant, tandis que le **Mask R-CNN Scratch** n'obtient aucune détection au seuil utilisé.

Ces résultats montrent l'intérêt potentiel du **Transfer Learning** sur un petit jeu de données, tout en soulignant l'importance de la qualité des annotations, de l'analyse des erreurs et d'un protocole de comparaison commun. Ils constituent une **contribution expérimentale au projet collectif**, et non un système de conduite autonome opérationnel.

---

## Références et fichiers du travail

- **Dataset :** RoadAnomaly, cité dans les documents du projet comme *Lis et al., ICCV 2019*.
- **Notebook individuel :** `BO1_RoadAnomaly_VERSION_FINALE_AMELIOREE(2).ipynb`.
- **Analyse des résultats :** `comparaison_modeles (1).md`.
- **Sorties prévues par le notebook :** `roadanomaly_results/` (checkpoints, historiques, métriques et analyses d'erreurs).

> **Note de traçabilité :** ce rapport documente les résultats déjà présents dans les fichiers fournis. Les performances doivent être recalculées après toute modification du découpage, du prétraitement ou de l'entraînement.
