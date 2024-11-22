
## Qu'est-ce que j'ai fait cette semaine?

### Veille bibliographique :
- Lecture E. GRINSTEIN 2023 - *The Neural-SRP method for universal robust multi-source tracking*
- Lecture M. SAAD AYUB 2023 - *Disambiguation of measurements for multiple acoustic source localization using deep multi-dimensional assignments*
- Lecture S. ADAVANNE 2021 - *Differentiable Tracking-Based Training of Deep Learning Sound Source Localizers*
- Lecture Y. XU 2020 - *How To Train Your Deep Multi-Object Tracker*

### Analyse des modèles :
Le modèle **HungarianNet** conçu par ADAVANNE est une architecture simplifiée - et je pense qu'ils ont mené une étude d'ablation - du modèle **Deep Hungarian Net (DHN)** de XU.

**HunNet** est une architecture à trois couches cachées :
- Une couche GRU à 128 neurones
- Une couche d'auto-attention à une seule tête de 128 neurones cachés
- Une couche fully-connected de taille F

Trois outputs possèdent une fonction de coût BCE loss. Une combinaison linéaire de ces trois fonctions de coût est rétropropagée pour la descente de gradients des poids de ce réseau. La tâche apprise est une tâche de classification binaire.

Alors que le modèle **DHN** fonctionne avec des réseaux Bi-RNNS et une fonction de coût focal loss. La tâche apprise est une tâche de classification binaire 2D.

### Questions et observations :
Comment a été réalisée l'étude d'ablation de ADAVANNE pour concevoir son modèle si différent de celui de XU dont il prétend s'inspirer?

XU prétend que son modèle possède des performances équilibrées pour les deux classes (0 et 1).

### Prise en main HungarianNet (version ADAVANNE) :
- Analyse et compréhension des étapes de génération des données, de l'entraînement du réseau, des résultats
- Analyse du fonctionnement et de la conception de HungarianNet
- Passage du fonctionnement de 2 sources à 10 sources (+ est possible)

### Génération des données :
Comment fonctionne la génération des données? Voir le document `generate_hnet_training_data.md`.

Pourquoi générer des combinaisons (`('nb_ref', 'nb_pred') = [(0, 0), (0, 1), (1, 0) ... (max_doas, max_doas)]`) pour l'entraînement supervisé de HunNet?

J'ai constaté également que l'algorithme de génération des données de ADAVANNE génère des labels A* avec plus d'occurrences de la classe 0 que de la classe 1 lorsque je veux l'entraîner dans un contexte de 10 DoAs.
- C'est une class imbalance. Mais pour 10 DoAs ground-truth correctement prédits, cela veut dire qu'il y a 10 uns dans une matrice 10x10 donc 90 zéros.
- Et puisque HunNet est entraîné à reproduire la tâche de la résolution du problème d'assignation de l'algorithme Hungarian.
- Alors dans une très grande majorité des cas (99%), HunNet résout le problème d'assignation (puisqu'il n'a jamais vu de matrice avec 100 uns).

### Entraînement de HunNet :
Comment fonctionne l'entraînement de HunNet? Voir le document `train_hnet.md`.

Quelle a été l'intuition de ADAVANNE pour concevoir son HunNet?

### Prochaine étape :
La prochaine étape à réaliser est l'étude d'ablation de HunNet. C'est surtout un problème d'organisation, de gestion et de suivi de l'étude car :
- Il est nécessaire de faire un suivi des expériences.
- Il est nécessaire de faire un versioning des modèles.
- Il est nécessaire de garantir la reproductibilité des expériences.

Pour cela, nous pourrions tirer profit des outils MLOps.
- Leverager l'outil MLflow quasi-identique à l'outil de visualisation TensorBoard (projet programme réentraînement Vibravox).
- Sont disponibles les outils MLflow, Weights & Biases, TensorBoard, Kubeflow, CometML.
- Il pourrait être envisagé d'intégrer du CI/CD dans l'étude d'ablation et dans la conception de BeamLearning v2 (comme ce qu'a fait Julien) (workflow d'automatisation des tests, de garantie de la consistance du modèle et de l'intégrité du code, créer une pipeline développement-à-déploiement pour une meilleure efficacité).
- Conteneuriser l'environnement d'entraînement avec Docker pour meilleure consistance de l'environnement, reproductibilité, isolation des dépendances, intégration avec les outils MLOps, collaboration en partageant les images Docker.
- leverager les frameworks fast.ai, TorchScript et Pytorch Lightning - DataParallel with Pytorch.Lightning
- leverager des outils comme SigOpt, Ray Tune, W&B pour la recherche d'hyperparamètres

### Prévision utilisation d'un bras de levier : MLOps
Voir le document `ablation_study.md` et `ci_cd_ablation_study.md`.
**Fil de pensée :**
1. Mener une étude d'ablation sur HungarianNet, qui est un petit modèle.
2. Une étude d'ablation nécessite de retirer des composants d'un modèle afin d'en comprendre son fonctionnement dans le but d'optimiser ses performances.
3. Ainsi, il est nécessaire d'implémenter, d'entraîner et de tester plusieurs modèles.
4. Il est donc aussi nécessaire de s'organiser, de log, d'enregistrer afin de ne pas se perdre i.e. enregistrer les modèles et leurs métadonnées dans une base de données (comment ont été entraînés ces modèles?) (F-score, epochs entraînés, temps d'inférence sur CPU...).
5. Il est nécessaire de rédiger des tests unitaires et d'intégration pour garantir la robustesse du ou des modèles.
6. Le MLOps (ou DevOps pour Machine Learning) peut être un outil très puissant dans cette tâche et pour la conception de BeamLearning v2.