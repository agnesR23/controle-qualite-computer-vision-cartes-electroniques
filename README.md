# Contrôle qualité de cartes électroniques par Computer Vision

## Objectif métier
Passer d’un contrôle visuel manuel à une détection automatique des défauts sur cartes électroniques.

---

## Démarche du projet

1. **Comparer les modèles de Computer Vision**  
   Benchmark entre Faster R-CNN, YOLOv8s baseline et plusieurs variantes YOLOv8s.

2. **Sélectionner le meilleur modèle**  
   Choix du modèle final à partir des métriques de détection et du temps d’inférence.

3. **Exposer l’inférence**  
   Création d’une API FastAPI avec documentation Swagger.

4. **Créer une application métier**
   Dashboard Streamlit permettant de contrôler une image, de visualiser les défauts détectés et de générer un rapport qualité automatique.

---

## Choix et évaluation du modèle

![Choix du modèle](docs/screenshots/choix-modele.png)

Modèle retenu : `yolov8s_epochs_20`

Cette configuration a été sélectionnée sur le jeu de validation, où elle obtient la meilleure mAP50-95 : **0,599**. Les expérimentations ont été suivies avec MLflow.

### Résultats finaux sur le jeu de test

Évaluation réalisée une seule fois sur **266 images indépendantes**, après la sélection du modèle.

| Métrique | Valeur |
|---|---:|
| mAP50-95 | 0,608 |
| mAP50 | 0,985 |
| Précision | 0,964 |
| Rappel | 0,991 |
| Temps de calcul local | 0,004 s/image |

Le modèle détecte la quasi-totalité des défauts réels. L’écart entre la mAP50 et la mAP50-95 montre que la précision de localisation des boîtes reste le principal axe d’amélioration.

---

## Déploiement et fonctionnement

Le dashboard et l’API sont déployés séparément :

- **Dashboard Streamlit** : [ouvrir l’application](https://controle-qualite-computer-vision-cartes-electroniques.streamlit.app/) ;
- **API d’inférence** : [ouvrir le Space Hugging Face](https://huggingface.co/spaces/agnesR23/pcb-defect-detection-api).

### Fonctionnement du dashboard Streamlit

**Plateforme : Streamlit Community Cloud**

![Démo Streamlit](docs/screenshots/demo-streamlit.png)

Le dashboard permet de :

- sélectionner une image à contrôler ;
- l’envoyer à l’API pour lancer l’inférence ;
- afficher l’image annotée et les résultats retournés ;
- générer automatiquement un rapport qualité.

> [!WARNING]
> Si l’API est indisponible, le dashboard le signale et n’affiche aucune prédiction statique.

### Fonctionnement de l’API FastAPI

**Plateforme : Hugging Face Spaces**

![API Swagger](docs/screenshots/api-swagger.png)

L’API permet de :

- recevoir une image à contrôler ;
- exécuter l’inférence avec le modèle ;
- détecter et localiser les défauts ;
- retourner le type de chaque défaut et son score de confiance ;
- retourner l’image annotée.

Endpoints principaux :

| Endpoint | Rôle |
|---|---|
| `/health` | Vérifier que l’API est disponible |
| `/model-info` | Afficher les informations du modèle |
| `/predict` | Envoyer une image et récupérer les prédictions |
| `/docs` | Accéder à la documentation Swagger |

---

## Organisation du projet

```text
notebooks/        benchmark, entraînement et sélection du modèle
src/              code d’inférence
api/              API FastAPI
app/              dashboard Streamlit
tests/            tests automatisés
docs/screenshots/ captures du README
```

Le modèle entraîné n’est pas versionné dans GitHub. Il est chargé localement s’il est présent, sinon récupéré depuis Hugging Face.

---

## Données

Le projet utilise le [PCB Defect Dataset — Norbert Elter](https://www.kaggle.com/datasets/norbertelter/pcb-defect-dataset), basé sur le dataset PCB Defect de Peking University.

Le jeu fourni contenait **10 668 images**, déjà réparties entre entraînement, validation et test. L’analyse a révélé :

- des versions d’une même image — rotations à 90° et 270°, et variation de luminosité — réparties dans plusieurs jeux ;
- **10 doublons**, regroupés en quatre groupes.

Les données ont été dédupliquées puis réparties à nouveau sans fuite entre les jeux :

- **8 500 images d’entraînement**, augmentations comprises ;
- **266 images de validation** ;
- **266 images de test**.

Les 12 images incluses dans l’application proviennent du nouveau jeu de test.

---

## Installation

```bash
conda env create -f environment.yml
conda activate pcb-defect-detection_env
```

---

## Lancer le projet en local

Lancer l’API :

```bash
uvicorn api.main:app --reload --port 8000
```

Swagger :

```text
http://127.0.0.1:8000/docs
```

Lancer Streamlit dans un deuxième terminal :

```bash
streamlit run app/streamlit_app.py
```

---

## Lancer l’API avec Docker

```bash
docker build -f api/Dockerfile -t controle-qualite-cv-cartes-electroniques-api .
docker run --rm -p 8000:7860 controle-qualite-cv-cartes-electroniques-api
```

Swagger :

```text
http://127.0.0.1:8000/docs
```

---

## Tests

```bash
pytest
```

Tests validés localement :

```text
test_metrics.py      OK
test_app_assets.py   OK
test_inference.py    OK
test_api.py          OK
test_data.py         OK
```

---

## Stack technique

- **Modélisation :** Python 3.11 · YOLOv8 · PyTorch · Torchvision
- **Expérimentation :** MLflow · Pandas · Matplotlib
- **Application et API :** Streamlit · FastAPI
- **Déploiement :** Docker · Hugging Face Spaces · Streamlit Community Cloud
- **Qualité du code :** Pytest · Ruff
- **Automatisation :** GitHub Actions · tests CI et requêtes planifiées de réactivation des services

---

## Licence

Copyright © 2026 Agnès REGAUD.

Ce projet est distribué sous licence [GNU Affero General Public License v3.0](LICENSE). Il utilise [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) sous licence AGPL-3.0.

---

## Auteur

Projet réalisé par Agnès REGAUD.

LinkedIn : https://www.linkedin.com/in/agnes-regaud/
