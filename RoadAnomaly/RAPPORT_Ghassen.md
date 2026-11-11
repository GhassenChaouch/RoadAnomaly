# Rapport individuel : BO2 - classification du type d'anomalie

## 1. Objectif métier

Le deuxième objectif métier (BO2) consiste à adapter la réaction du système au type d'objet détecté. Après avoir localisé une anomalie routière, le système doit la classer dans l'une des catégories suivantes :

- animal ;
- obstacle ;
- cône ;
- véhicule.

La réaction attendue n'est pas la même pour un animal, un rocher ou un cône de chantier. Le critère principal retenu est donc le **F1-score par classe**, complété par la macro-F1, la matrice de confusion, la précision et le rappel.

## 2. Données utilisées

Le dataset RoadAnomaly contient des images routières et des masques sémantiques pixel par pixel dans `frames/`. Le fichier `split.csv` fournit le découpage commun entre les ensembles d'entraînement, de validation et de test.

Les catégories sont déduites du nom des images uniquement pour créer les labels d'entraînement :

| Préfixe du fichier | Classe BO2 |
|---|---|
| `animals*` | animal |
| `cones*` | cone |
| `obstacles*` | obstacle |
| `lost_tires*` | obstacle |
| `vehicle*` | vehicle |

Le nom du fichier n'est jamais fourni au réseau comme donnée d'entrée. Il sert uniquement à construire la cible attendue pendant l'entraînement.

La distribution est déséquilibrée :

- 31 images d'animaux ;
- 15 images d'obstacles ;
- 8 images de cônes ;
- 3 images de véhicules ;
- 3 images de pneus perdus, regroupés dans `obstacle`.

La classe `vehicle` est donc trop petite pour fournir une estimation statistique robuste. Son F1 doit être interprété avec prudence et l'ajout d'images de véhicules est recommandé.

## 3. Architecture retenue

La solution retenue est une architecture en deux étapes.

### Étape 1 : localisation

Le modèle U-Net avec encodeur ResNet34 pré-entraîné, développé pour le premier objectif métier, prédit le masque de l'anomalie dans l'image.

### Étape 2 : classification

Le classifieur reçoit un crop centré sur la région d'anomalie et prédit son type :

```text
image routière
    ↓
U-Net : prédiction du masque d'anomalie
    ↓
crop de la région détectée
    ↓
classifieur : animal / obstacle / cone / vehicle
```

Cette approche évite de faire dépendre la classification du décor général de la route et correspond au fonctionnement attendu en situation réelle.

## 4. Modèles comparés

Deux modèles sont prévus afin de mesurer l'apport du transfert d'apprentissage :

### ResNet18 pré-entraîné

ResNet18 utilise des poids ImageNet et possède une dernière couche adaptée aux quatre classes BO2. Il constitue le modèle principal, car le dataset est très petit.

### CNN entraîné from scratch

Un CNN compact est entraîné avec une initialisation aléatoire. Il sert de baseline et permet de comparer directement les performances avec le modèle pré-entraîné.

Un échantillonnage équilibré est utilisé pendant l'entraînement pour limiter l'effet du déséquilibre entre les classes.

## 5. Implémentation réalisée

Les fichiers suivants ont été créés :

- [`bo2_classification.py`](./bo2_classification.py) : implémentation du dataset, des modèles, de l'entraînement, des métriques et de l'inférence ;
- [`BO2_classification.ipynb`](./BO2_classification.ipynb) : notebook pédagogique avec des cellules indépendantes et des descriptions en français.

Le notebook présente successivement :

1. la préparation de l'environnement ;
2. la création et la vérification des labels ;
3. l'affichage d'une image, de son masque et du crop ;
4. la construction des ensembles train/validation/test ;
5. la sélection du modèle ;
6. l'entraînement avec sélection par macro-F1 de validation ;
7. l'évaluation sur le test ;
8. la sauvegarde du checkpoint ;
9. le test sur une image uploadée.

## 6. Inférence sur une image uploadée

Pour une image nouvelle, aucun nom particulier et aucun masque annoté ne sont nécessaires. La fonction `classify_image` :

1. reçoit le chemin de l'image ;
2. demande au U-Net de prédire le masque d'anomalie ;
3. extrait automatiquement le crop correspondant ;
4. applique le classifieur ;
5. retourne la classe prédite et sa confiance.

Ainsi, une image possédant un nom quelconque peut être testée. Le filename n'intervient pas dans la prédiction.

## 7. Évaluation et limites

Les sorties prévues sont :

- F1-score pour chaque classe ;
- macro-F1 ;
- précision et rappel par classe ;
- matrice de confusion ;
- nombre de paramètres et temps d'inférence, si mesurés.

Les principales limites sont :

1. seulement trois images de véhicules sont disponibles ;
2. le test contient une seule image de véhicule ;
3. les résultats peuvent varier fortement avec un dataset aussi petit ;
4. la qualité de la classification dépend de la qualité du masque prédit par le U-Net ;
5. une validation croisée stratifiée serait préférable pour confirmer les résultats.

## 8. État de validation

La structure du notebook a été validée :

- le fichier notebook est un JSON valide ;
- les cellules Python sont syntaxiquement valides ;
- le script Python se compile correctement ;
- les exports `load_unet`, `predict_anomaly_crop` et `classify_image` sont disponibles ;
- la correction du label `lost_tires` a été intégrée.

L'exécution complète de PyTorch reste dépendante de l'environnement Windows local : lors d'une vérification précédente, une DLL native de PyTorch a été bloquée par la politique de contrôle des applications. Le code est toutefois prêt à être exécuté dans un environnement où PyTorch peut charger ses bibliothèques natives.

