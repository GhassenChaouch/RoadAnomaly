# Processus de réalisation de BO2

## 1. Demande initiale

L'objectif demandé était de réaliser BO2 :

> Une fois l'anomalie localisée, la classer en animal, obstacle, cône ou véhicule, afin d'adapter la réaction du système. Le critère principal est le F1 par classe.

Il fallait également déterminer si les données présentes dans `frames/` étaient suffisantes et proposer des modèles permettant de valoriser le travail réalisé.

## 2. Analyse du projet existant

Le projet contenait déjà :

- un dataset RoadAnomaly ;
- des images dans `frames/` ;
- des masques sémantiques dans les dossiers `.labels` ;
- un fichier `split.csv` ;
- un modèle U-Net ResNet34 pour localiser les anomalies ;
- plusieurs notebooks d'entraînement et d'évaluation.

L'analyse du rapport existant a montré que le U-Net pré-entraîné était le modèle principal pour la localisation.

## 3. Analyse des catégories

L'information de catégorie est présente dans les noms de fichiers. L'analyse de `split.csv` a identifié :

- `animals*` ;
- `cones*` ;
- `obstacles*` ;
- `vehicle*` ;
- `lost_tires*`.

Le choix a été fait de regrouper `lost_tires` avec `obstacle`, afin de conserver les quatre classes demandées par BO2.

Un problème de déséquilibre a été identifié : la classe véhicule ne contient que trois images. Il a donc été conclu que les données sont suffisantes pour un prototype et une comparaison académique, mais insuffisantes pour une conclusion robuste sur les véhicules.

## 4. Première proposition technique

La première proposition consistait à :

1. réutiliser le U-Net pour localiser l'anomalie ;
2. extraire un crop avec le masque ;
3. entraîner un classifieur ;
4. comparer un modèle pré-entraîné à un CNN from scratch ;
5. mesurer le F1 par classe et la macro-F1.

Les modèles proposés étaient ResNet18 pré-entraîné, MobileNetV3 éventuellement, et un petit CNN from scratch comme baseline.

## 5. Première implémentation

Le fichier [`bo2_classification.py`](./bo2_classification.py) a été créé avec :

- les quatre classes BO2 ;
- la création des labels depuis les noms ;
- le chargement des images et des masques ;
- l'extraction des crops ;
- un CNN from scratch ;
- un ResNet18 pré-entraîné ;
- un sampler équilibré ;
- l'entraînement ;
- le rapport de classification ;
- la matrice de confusion ;
- la sauvegarde du meilleur checkpoint.

Un notebook pédagogique [`BO2_classification.ipynb`](./BO2_classification.ipynb) a ensuite été créé pour permettre l'exécution cellule par cellule.

## 6. Problème identifié par l'utilisateur

L'utilisateur a remarqué que le code utilisait le nom de fichier pour déterminer la classe et a demandé si le modèle continuerait à utiliser ce nom lors d'une future prédiction sur une image uploadée.

Cette remarque a conduit à distinguer clairement deux usages :

- pendant l'entraînement, le nom sert à fournir le label attendu ;
- pendant l'inférence, le modèle doit recevoir uniquement les pixels de l'image.

Le nom ne doit donc jamais être une caractéristique d'entrée du classifieur.

## 7. Décision validée

Après clarification, l'approche suivante a été approuvée :

- architecture en deux étapes ;
- U-Net pour prédire le masque sur l'image uploadée ;
- classifieur pour prédire la catégorie depuis le crop prédit ;
- comparaison ResNet18 pré-entraîné contre CNN from scratch.

Cette décision évite d'utiliser un masque annoté, qui n'existe pas pour une image uploadée.

## 8. Corrections apportées

### Correction du mapping des labels

Le premier mapping séparait le nom au premier `_`. Cela transformait `lost_tires03` en `lost`, ce qui provoquait une erreur.

Le mapping a été corrigé pour tester le début complet du nom :

- `animals` ;
- `cones` ;
- `obstacles` ;
- `lost_tires` ;
- `vehicle`.

### Ajout de l'inférence U-Net

Les fonctions suivantes ont été ajoutées à `bo2_classification.py` :

- `crop_from_mask` : extrait un crop à partir d'un masque en mémoire ;
- `load_unet` : charge le checkpoint U-Net ;
- `predict_anomaly_crop` : prédit le masque et extrait le crop ;
- `classify_image` : réalise la chaîne complète sur une image nouvelle.

### Mise à jour du notebook

Le notebook a été enrichi avec :

- le rechargement du module Python pour éviter les versions mises en cache par Jupyter ;
- la correction du mapping des catégories ;
- une cellule de chargement du U-Net et du classifieur ;
- une cellule `UPLOAD_PATH` pour tester une image avec n'importe quel nom ;
- l'affichage de la classe et de la confiance prédites.

Les sorties d'erreur précédentes ont été supprimées afin que le notebook puisse être relancé proprement.

## 9. Fonctionnement final

Le processus final est :

```text
label du nom de fichier
    → utilisé uniquement pendant l'entraînement

image d'entraînement + masque annoté
    → crop d'anomalie
    → apprentissage du classifieur

image uploadée, nom quelconque
    → U-Net
    → masque prédit
    → crop prédit
    → classifieur
    → animal / obstacle / cone / vehicle
```

## 10. Vérifications réalisées

Les vérifications suivantes ont été réalisées :

- validation JSON du notebook ;
- analyse syntaxique de toutes les cellules Python ;
- compilation du script Python ;
- vérification des exports du module ;
- vérification de la correction `lost_tires` ;
- nettoyage des anciennes sorties d'erreur du notebook.

La vérification complète du réseau dépend toutefois du chargement des DLL natives de PyTorch dans l'environnement Windows. Une politique de contrôle des applications avait précédemment empêché le chargement de PyTorch dans cet environnement.

## 11. Améliorations futures

Les améliorations recommandées sont :

1. ajouter davantage d'images de véhicules ;
2. ajouter davantage d'images de cônes ;
3. utiliser une validation croisée stratifiée ;
4. mesurer le temps d'inférence du pipeline complet ;
5. tester plusieurs seuils du masque U-Net ;
6. créer une interface d'upload ;
7. évaluer séparément l'erreur de localisation et l'erreur de classification.
