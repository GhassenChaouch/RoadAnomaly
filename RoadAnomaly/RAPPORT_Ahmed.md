# Rapport individuel : Ahmed
## Projet Deep Learning : détection d'anomalies routières (RoadAnomaly)

Branche : `ahmed/unet` · Dépôt : `GhassenChaouch/RoadAnomaly` · Rendu : 11/11/2026

> **Pour les coéquipiers.** Ce document décrit ma partie en suivant les phases CRISP-DM demandées. Les sections 2 et 3 (données et protocole) sont communes : si vous gardez le même découpage, la même résolution et les mêmes métriques, nos résultats seront directement comparables dans le rapport final. Les sections 4 à 6 sont propres à mon BO. Les tableaux sont prêts à être repris. Les passages marqués **[à compléter]** attendent un résultat qui n'est pas encore là.

---

## 1. Business understanding

### 1.1 Objectif métier (BO)
Aider un **système d'aide à la conduite** à repérer et localiser tout obstacle inattendu sur la chaussée (animal, rocher, cône, engin), pour **déclencher une alerte**.

### 1.2 Ce que cela change pour l'évaluation
- Rater un obstacle est plus grave qu'une fausse alerte : le **rappel** est le critère principal.
- Trop de fausses alertes rend le système inutilisable : la **précision au rappel visé** mesure ce coût.
- Un système de conduite doit répondre vite : on mesure aussi la **taille** du modèle et son **temps d'inférence**.

### 1.3 Critères de succès retenus
| Critère | Mesure |
|---|---|
| Ne pas rater d'obstacle | rappel pixel sur le test, et nombre d'images dont au moins 50 % de l'obstacle est retrouvé |
| Peu de fausses alertes | précision au rappel visé, et part de la surface de l'image signalée à tort |
| Réactivité | temps d'inférence par image sur CPU |
| Légèreté | nombre de paramètres, taille du fichier de poids |

---

## 2. Data understanding et préparation

Le détail de l'exploration est dans le notebook d'exploration (`poject.ipynb`, travail d'Arije). Ce qui est utilisé ici :

- **Jeu de données :** 60 images routières avec un masque pixel par pixel (0 = route normale, 1 = anomalie), résolution d'origine 1280×720. Le fond représente environ 90 % des pixels.
- **Découpage commun** (`split.csv`) : **36 images d'entraînement, 12 de validation, 12 de test**, stratifié par groupe d'obstacle (animaux, obstacles, cônes, autres) et par taille d'anomalie, graine 42. Aucune image n'est dans deux ensembles.
- **Prétraitement :** redimensionnement à **512×288** (ratio 16:9 conservé, multiple de 32) ; normalisation avec la moyenne et l'écart-type ImageNet, pour que tous les modèles reçoivent les mêmes entrées.
- **Augmentation (entraînement uniquement) :** zoom aléatoire (recadrage de 50 à 100 % de l'image, contenant de l'anomalie), symétrie horizontale, variations de luminosité et de contraste. La rotation n'est pas utilisée.
- **Matériel :** aucun GPU disponible, tout est entraîné et mesuré sur CPU.

---

## 3. Protocole commun d'entraînement et d'évaluation

| Élément | Choix |
|---|---|
| Perte | BCE + Dice (poids 0,5 / 0,5), car le fond écrase l'anomalie |
| Optimiseur | AdamW, weight decay 1e-4, planning cosinus du learning rate |
| Epochs | 60 au maximum, arrêt anticipé si le Dice de validation ne progresse plus pendant 15 epochs |
| Batch | 4 |
| Sélection du modèle | epoch au meilleur Dice de validation |
| Seuil de décision | choisi **sur la validation**, jamais sur le test |
| Métriques | IoU, Dice (= F1), précision, rappel, calculées sur l'ensemble des pixels (globales) et par image (moyennes) |
| Références | « tout fond » (Dice 0) et **a priori de position** (Dice 0,314, IoU 0,186 sur le test), qui prédit seulement où les anomalies apparaissent en général |

Le notebook `Comparaison` refait toutes ces mesures pour mes trois modèles avec exactement le même code.

---

## 4. Modeling : trois modèles, un seul BO

Les trois modèles servent le même BO. Chaque comparaison ne change qu'un seul élément à la fois.

| | A. U-Net ResNet34 pré-entraîné | B. U-Net ResNet34 from scratch | C. Encodeur-décodeur CNN simple |
|---|---|---|---|
| Notebook | `01_unet_pretrained.ipynb` | `02_unet_scratch.ipynb` | `03_cnn_encdec_scratch.ipynb` |
| Architecture | U-Net, encodeur ResNet34 | identique à A | 4 blocs Conv2D + MaxPooling, goulot avec Dropout, 4 blocs de convolutions transposées, **sans connexions de saut** |
| Initialisation | poids ImageNet | aléatoire | aléatoire |
| Learning rate | 1e-4 (encodeur), 1e-3 (décodeur) | identique à A | 1e-3 (un seul) |
| Paramètres | 24,44 M | 24,44 M | 1,75 M |
| Fichier de poids | 97,9 Mo | 97,9 Mo | 7,0 Mo |
| Epoch retenue | 35 | 39 (arrêt anticipé à l'epoch 54) | 20 (arrêt anticipé à l'epoch 35, 5,6 min) |

**Ce que mesure chaque comparaison**
- **A contre B :** même architecture, même hyperparamètres, seule l'initialisation change. Cela mesure l'apport du **transfert d'apprentissage**.
- **B contre C :** même initialisation aléatoire, seule l'architecture change. Cela mesure l'apport de **ResNet et des connexions de saut**.

**Choix à justifier à l'oral**
- Le **U-Net n'est pas dans la liste des chapitres du cours**. Il se présente comme un encodeur ResNet (chapitre 4) avec un décodeur à connexions de saut qui produit un masque pixel par pixel.
- Le modèle C n'utilise que des briques du cours (Conv2D, MaxPooling, ReLU, Dropout, convolutions transposées comme dans l'autoencodeur convolutif du chapitre 5). Il ajoute de la **Batch Normalization**, absente de la liste du cours, activable avec `USE_BN`.
- Pour B, j'ai gardé les hyperparamètres de A (learning rate de l'encodeur 1e-4) pour que seule l'initialisation diffère. Un learning rate plus élevé n'a pas été testé et pourrait améliorer B.

---

## 5. Evaluation

### 5.1 Résultats sur le test au seuil du meilleur Dice de validation
| Modèle | Seuil | IoU | Dice | Précision | Rappel |
|---|---|---|---|---|---|
| A. U-Net pré-entraîné | 0,45 | **0,713** | **0,832** | **0,880** | **0,790** |
| B. U-Net from scratch | 0,75 | 0,339 | 0,507 | 0,580 | 0,450 |
| C. CNN encodeur-décodeur | 0,50 | 0,267 | 0,421 | 0,289 | 0,776 |
| Référence : a priori de position | 0,18 | 0,186 | 0,314 | 0,205 | 0,673 |

Dice sur la validation : A 0,900 · B 0,744 · C 0,611. Les trois modèles dépassent la référence, mais C de seulement 0,11 point de Dice.

### 5.2 Résultats orientés BO : seuil choisi pour atteindre un rappel visé
Pour chaque rappel visé, le seuil est le plus élevé qui atteint la cible **sur la validation**. Les métriques sont lues sur le **test**. « Fausses alertes » = part moyenne de la surface de l'image signalée à tort.

| Modèle | Mode | Seuil | Rappel val | Rappel test | Précision test | Dice test | Obstacles détectés (≥ 50 %) | Fausses alertes |
|---|---|---|---|---|---|---|---|---|
| A | Dice maximal | 0,45 | 0,879 | 0,790 | 0,880 | 0,832 | 11/12 | 0,8 % |
| A | Rappel visé 80 % | 0,87 | 0,804 | 0,701 | 0,939 | 0,802 | 11/12 | 0,3 % |
| A | **Rappel visé 90 %** | **0,27** | 0,900 | **0,811** | **0,849** | 0,830 | 11/12 | **1,1 %** |
| B | Dice maximal | 0,75 | 0,684 | 0,450 | 0,580 | 0,507 | 5/12 | 2,5 % |
| B | Rappel visé 80 % | 0,54 | 0,802 | 0,577 | 0,486 | 0,528 | 7/12 | 4,6 % |
| B | Rappel visé 90 % | 0,26 | 0,902 | 0,695 | 0,369 | 0,482 | 11/12 | 9,0 % |
| C | Dice maximal | 0,50 | 0,786 | 0,776 | 0,289 | 0,421 | 12/12 | 14,5 % |
| C | Rappel visé 80 % | 0,48 | 0,804 | 0,793 | 0,283 | 0,417 | 12/12 | 15,2 % |
| C | Rappel visé 90 % | 0,32 | 0,903 | 0,891 | 0,238 | 0,376 | 12/12 | 21,6 % |

**Lecture**
- **Le modèle A est le seul exploitable pour ce BO.** Au rappel visé de 90 %, il garde 85 % de précision et signale environ 1 % de la surface de l'image à tort.
- **Le modèle C obtient « 12/12 » en marquant plus d'un cinquième de l'image.** Un système qui alerte en permanence est inutilisable : cet indicateur ne se lit jamais sans la colonne « fausses alertes ».
- **Le modèle B** ne retrouve 11 obstacles sur 12 qu'avec 9 % de la surface en fausses alertes.
- **La cible de rappel de la validation n'est pas atteinte telle quelle sur le test** (A : 0,90 en validation, 0,81 en test). Seuils et rappels sont estimés sur 12 images, il faut annoncer le chiffre du test.

### 5.3 Taille et temps d'inférence (CPU, 1 image 512×288)
| Modèle | Paramètres | Fichier | Inférence |
|---|---|---|---|
| A et B (même architecture) | 24,44 M | 97,9 Mo | environ 120 ms |
| C | 1,75 M | 7,0 Mo | environ 50 ms |

Les mesures varient avec la charge du PC : entre 119 et 225 ms pour les deux U-Net, entre 50 et 96 ms pour C. Les deux U-Net ont la même architecture, leurs temps ne diffèrent pas réellement. Les valeurs ci-dessus sont la meilleure estimation, obtenue en médiane sur 50 passages après échauffement. Même C (environ 20 images par seconde) reste juste pour un flux vidéo à 30 images par seconde.

### 5.4 Analyse critique
1. **Un seul obstacle presque raté (A, seuil 0,27).** `vehicle02_Broadmoor_Cones_Skidloader` : rappel 0,076, précision 0,728. Les 11 autres images ont un rappel de 0,76 à 0,98. Cela explique le plafond de rappel de A sur le test, qui n'atteint pas 0,9 même au seuil le plus bas.
2. **Transfert d'apprentissage.** +0,33 de Dice entre A et B. Avec 36 images d'entraînement, apprendre les filtres visuels depuis zéro est très difficile.
3. **Connexions de saut.** B dépasse C de 0,09 de Dice, et surtout de 0,29 de précision. Les cartes de C sont floues et en blocs : sans connexions de saut, les contours se perdent.
4. **Validation bruitée.** Le Dice de validation varie de 0,3 à 0,7 d'une epoch à l'autre pour B et C (12 images). L'epoch retenue correspond souvent à un pic : le Dice de validation est optimiste, le test est plus honnête (B : 0,744 en validation, 0,507 en test).
5. **Petit jeu de test.** 12 images : un écart de quelques points entre deux modèles n'est pas significatif.
6. **Mesure du temps.** CPU de PC portable, résultats instables d'un passage à l'autre, non représentatifs d'un système embarqué.
7. **Les tests rapides** (petite résolution, 12 epochs) servent uniquement à valider la chaîne, leurs chiffres ne sont pas rapportés.

---

## 6. XAI : test d'occlusion sur le modèle A
Notebook : `05_xai_occlusion.ipynb`. On masque tour à tour chaque zone de l'image (carré gris 64×64, pas de 32) et on mesure la baisse de confiance du modèle. Une zone rouge est une zone dont dépend la décision.
**Résultats (modèle A, seuil 0,27)**
- **Cas bien détecté (`animals12`)** : la sensibilité recouvre le corps de l'animal. Masquer cette zone fait chuter la confiance, le modèle s'appuie donc sur l'obstacle lui-même.
- **Cas raté (`vehicle02`, rappel 0,08)** : la détection et la sensibilité se concentrent sur les trois cônes du premier plan, pas sur l'engin de chantier. Le modèle a appris que les cônes sont des anomalies mais ne reconnaît pas cet engin. Dans ce masque, seul l'engin est annoté : les cônes comptent comme fausses alertes, ce qui explique la précision de 0,73 sur cette image.
- **Sur les 12 images du test**, en moyenne 36 % de la sensibilité tombe sur l'obstacle, qui occupe 7,6 % de l'image en moyenne (environ 4,7 fois plus que le hasard). 11 images sur 12 ont un rapport part / surface de 2,9 à 17,4 (médiane 5,7). Seule `vehicle02` est au niveau du hasard (0,95), et c'est aussi la seule image dont le rappel est faible.
- **Conclusion** : le modèle fonde en général sa décision sur l'obstacle, pas sur le décor. Son échec principal vient d'un type d'obstacle absent de l'apprentissage (un engin), pas d'une confusion avec le contexte de la route.

**Précautions de lecture** : le rapport part / surface est mécaniquement plus élevé pour les petits objets (cônes, veau) car le carré masqué de 64 px déborde de l'objet ; l'échelle de couleur est normalisée par image ; le test d'occlusion montre une dépendance, pas une causalité ; le carré gris est lui-même hors distribution ; un seul cas d'échec ne permet pas de généraliser. Grad-CAM donnerait une carte plus fine, au prix d'un accès aux couches internes.

**[à compléter après exécution du notebook 05]**
- Cas bien détecté (`animals12`) : la sensibilité recouvre-t-elle l'animal ? …
- Cas raté (`vehicle02`) : la détection dépend-elle des cônes plutôt que de l'engin ? …
- Part moyenne de la sensibilité située sur l'obstacle, comparée à la surface de l'obstacle (rapport part / surface) : …
- Images où le modèle regarde surtout le contexte (rapport proche de 1 ou inférieur) : …

Limites : le test d'occlusion montre une dépendance, pas une causalité ; le carré de 64 px est grossier pour les petits objets ; le carré gris est lui-même hors distribution.

---

## 7. Déploiement (web app « vitrine »)
- Modèle déployé pour mon BO : **A (U-Net ResNet34 pré-entraîné)**.
- Seuil par défaut : **0,27** (mode rappel visé 90 %), avec un **curseur de seuil** pour montrer le compromis rappel / précision.
- Affichage prévu : image envoyée, masque prédit, carte de probabilité.
- Fichier de poids `unet_resnet34_pretrained_best.pt` (98 Mo) : non versionné sur GitHub (`*.pt` est ignoré). Le déploiement est fait à partir de mon poste ; le fichier peut être partagé par Drive sur demande.
- Temps de réponse attendu sur CPU : environ 120 ms par image pour le modèle, hors chargement de l'image.

---

## 8. Liens avec le cours
| Notion du cours | Où dans ce travail |
|---|---|
| Binary Cross-Entropy (chap. 2) | perte BCE + Dice |
| Optimiseurs Adam, weight decay L2, Dropout, early stopping (chap. 3) | AdamW, weight decay 1e-4, Dropout au goulot de C, arrêt anticipé |
| Conv2D, MaxPooling, ResNet (chap. 4) | encodeur ResNet34 (A et B), blocs de C |
| Autoencodeur convolutif (chap. 5) | structure encodeur-décodeur de C |
| Recherche manuelle d'hyperparamètres (chap. 3) | learning rates, seuil de décision |

---

## 9. Fichiers et reproduction
Tous les notebooks s'ouvrent depuis `RoadAnomaly\RoadAnomaly` avec le kernel du venv du dépôt et ne contiennent aucun chemin absolu.

| Fichier | Rôle |
|---|---|
| `01_unet_pretrained.ipynb` | modèle A |
| `02_unet_scratch.ipynb` | modèle B (`ENCODER_WEIGHTS = None`) |
| `03_cnn_encdec_scratch.ipynb` | modèle C (`QUICK_TEST = False` pour le vrai run) |
| `Comparaison_AHMED.ipynb` | comparaison orientée rappel, taille et temps (à exécuter seul sur la machine) |
| `05_xai_occlusion.ipynb` | test d'occlusion sur le modèle A |
| `outputs_unet_resnet34_pretrained`, `outputs_unet_resnet34_scratch`, `outputs_cnn_encdec_scratch` | courbes, prédictions, `resultats_unet.json`, tableaux (poids `.pt` exclus de Git) |
| `outputs_comparaison_bo`, `outputs_xai` | tableaux et figures de la comparaison et du XAI |

---

## 10. Ce qui est proposé au groupe pour que les résultats soient comparables
1. Garder le **même `split.csv`**, la même résolution 512×288 et la même normalisation.
2. Rapporter les **mêmes métriques** (IoU, Dice, précision, rappel, globales et par image), avec un seuil choisi sur la validation.
3. Enregistrer les résultats dans un dossier `outputs_<modele>` avec un `resultats_unet.json` de même format, que le notebook de comparaison sait lire.
4. Indiquer pour chaque modèle l'initialisation (pré-entraîné ou from scratch), les hyperparamètres et le temps d'inférence.
5. Décider ensemble qui construit l'app (une seule app, un onglet par BO), la vidéo et le rapport.
