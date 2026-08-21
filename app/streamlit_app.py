# Copyright (C) 2026 Agnès REGAUD
# SPDX-License-Identifier: AGPL-3.0-only

# -----------------------------
# Imports
# -----------------------------
import base64
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from api_client import check_api_health, predict_image_with_api
from settings import get_api_url

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(page_title="PCB Defect Detection", layout="wide")


# -----------------------------
# Paths
# -----------------------------
APP_DIR = Path(__file__).resolve().parent

ASSETS_DIR = APP_DIR / "assets"
ORIGINALS_DIR = APP_DIR / "sample_images" / "originals"

BENCHMARK_RESULTS_PATH = ASSETS_DIR / "benchmark_results.csv"
YOLO_TUNING_RESULTS_PATH = ASSETS_DIR / "yolo_tuning_results.csv"
FINAL_TEST_METRICS_PATH = ASSETS_DIR / "final_test_metrics.csv"
FINAL_TEST_PER_CLASS_METRICS_PATH = (
    ASSETS_DIR / "final_test_per_class_metrics.csv"
)
TEST_DIAGNOSTICS_PATH = ASSETS_DIR / "test_diagnostics.png"
RESULTS_PATH = ASSETS_DIR / "results.csv"

# -----------------------------
# API configuration
# -----------------------------
API_URL = get_api_url()


# -----------------------------
# Constants
# -----------------------------
CLASS_NAMES = [
    "mouse_bite",
    "spur",
    "missing_hole",
    "short",
    "open_circuit",
    "spurious_copper",
]

CLASS_LABELS = {
    "mouse_bite": "Mouse bite (bord grignoté)",
    "spur": "Spur (excroissance de cuivre)",
    "missing_hole": "Missing hole (trou manquant)",
    "short": "Short (court-circuit)",
    "open_circuit": "Open circuit (circuit ouvert)",
    "spurious_copper": "Spurious copper (cuivre parasite)",
}

PLOT_TITLE_SIZE = 9
PLOT_LABEL_SIZE = 8
PLOT_TICK_SIZE = 7
PLOT_VALUE_SIZE = 7


# -----------------------------
# Helpers
# -----------------------------
@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def format_metric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = df[col].round(3)
    return df


def add_experiment_parameters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add readable training parameters for each benchmark or tuning run.
    """
    df = df.copy()

    parameter_mapping = {
        "yolov8s": (
            "YOLOv8s baseline — epochs=10, batch=16, lr=0.001, "
            "image_size=640, device=mps"
        ),
        "faster_rcnn_resnet50_fpn": (
            "Faster R-CNN baseline — epochs=1, batch=2, lr=0.005, "
            "image_size=640, device=cpu, workers=0"
        ),
        "yolov8s_epochs_20": (
            "YOLOv8s tuning — epochs=20, batch=16, lr=0.001, "
            "image_size=640, device=mps"
        ),
        "yolov8s_lr_0.0005": (
            "YOLOv8s tuning — epochs=10, batch=16, lr=0.0005, "
            "image_size=640, device=mps"
        ),
        "yolov8s_batch_8": (
            "YOLOv8s tuning — epochs=10, batch=8, lr=0.001, "
            "image_size=640, device=mps"
        ),
    }

    df["parameters"] = df["model_name"].map(parameter_mapping).fillna("Non renseigné")

    return df


def plot_horizontal_metric(
    df: pd.DataFrame,
    label_col: str,
    value_col: str,
    title: str,
    xlabel: str,
    figsize: tuple[float, float] = (6, 3),
    value_format: str = ".3f",
):
    plot_df = df.sort_values(value_col, ascending=True)

    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.barh(
        plot_df[label_col],
        plot_df[value_col],
        height=0.45,
    )

    ax.set_title(title, fontsize=PLOT_TITLE_SIZE)
    ax.set_xlabel(xlabel, fontsize=PLOT_LABEL_SIZE)
    ax.set_ylabel("")
    ax.set_xlim(0, 1)
    ax.grid(axis="x", alpha=0.3)
    ax.tick_params(axis="both", labelsize=PLOT_TICK_SIZE)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar in bars:
        value = bar.get_width()
        ax.text(
            value + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:{value_format}}",
            va="center",
            fontsize=PLOT_VALUE_SIZE,
        )

    fig.tight_layout()
    return fig


def load_demo_images() -> dict[str, list[dict[str, Path]]]:
    """
    Load the original demonstration images.
    """
    demo_images = {}

    for class_name in CLASS_NAMES:
        class_images = []

        for idx in [1, 2]:
            base_name = f"{class_name}_{idx:02d}"
            original_path = ORIGINALS_DIR / f"{base_name}.jpg"

            if original_path.exists():
                class_images.append(
                    {
                        "name": base_name,
                        "original": original_path,
                    }
                )

        if class_images:
            demo_images[class_name] = class_images

    return demo_images


def render_detection_summary(
    detected_classes: list[str],
    api_available: bool,
) -> None:
    """
    Display detected classes, inference service availability and confidence information.
    """
    for class_name in CLASS_NAMES:
        label = CLASS_LABELS.get(class_name, class_name)
        checked = "✅" if class_name in detected_classes else "☐"
        st.markdown(f"{checked} {label}")

    st.markdown("#### État du système")

    if api_available:
        st.markdown("✅ Système de détection opérationnel (API disponible)")
    else:
        st.markdown("❌ Système de détection indisponible (API indisponible)")

    st.markdown("#### Score de confiance")
    st.markdown(
        "Sur l’image, chaque défaut détecté est accompagné de son score "
    "de confiance. Il mesure la correspondance entre chaque défaut détecté "
    "et les caractéristiques apprises par le modèle pendant l’entraînement. "
    "Si plusieurs défauts sont détectés, le rapport final par image affiche le score le plus élevé."
    )


def render_quality_report(
    image_name: str,
    detected_classes: list[str],
    detections: list[dict],
    prediction_mode: str | None,
    api_available: bool,
) -> None:
    """
    Display the automatic quality report below the images.
    """
    detected_labels = [
        CLASS_LABELS.get(class_name, class_name)
        for class_name in detected_classes
    ]

    confidence_scores = [
        detection.get("confidence")
        for detection in detections
        if detection.get("confidence") is not None
    ]

    if prediction_mode != "live":
        confidence_display = "Prédiction non effectuée"
    elif confidence_scores:
        confidence_display = f"{max(confidence_scores):.2f}"
    else:
        confidence_display = "Aucun score disponible"

    system_status = (
        "Système de détection opérationnel (API disponible)"
        if api_available
        else "Système de détection indisponible (API indisponible)"
    )

    if prediction_mode == "live":
        defect_count_display = len(detections)
        defect_type_display = (
            ", ".join(detected_labels)
            if detected_labels
            else "Aucun défaut détecté"
        )
    else:
        defect_count_display = "Prédiction non effectuée"
        defect_type_display = "Prédiction non effectuée"

    report_df = pd.DataFrame(
        [
            {
                "Information": "Image contrôlée",
                "Valeur": image_name,
            },
            {
                "Information": "Nombre de défauts détectés",
                "Valeur": defect_count_display,
            },
            {
                "Information": "Type de défaut détecté",
                "Valeur": defect_type_display,
            },
            {"Information": "État du système", "Valeur": system_status},
            {"Information": "Score de confiance le plus élevé de l’image", "Valeur": confidence_display},
        ]
    )
    st.markdown("---")
    st.markdown("#### Rapport qualité automatique")

    col_report, col_action = st.columns([1.4, 1])

    with col_report:
        st.markdown(
            """
    <style>
    table {
        font-size: 17px !important;
    }
    thead tr th {
        font-size: 17px !important;
        font-weight: 700 !important;
    }
    tbody tr td {
        font-size: 17px !important;
        padding: 8px 10px !important;
    }
    </style>
    """,
            unsafe_allow_html=True,
        )

        st.markdown(
            report_df.to_html(index=False, escape=False),
            unsafe_allow_html=True,
        )

    with col_action:
        st.markdown(
            """
    <div style="
        padding: 16px 18px;
        border-radius: 10px;
        border: 1px solid #d1d5db;
        background-color: #f8fafc;
        font-size: 18px;
        line-height: 1.6;
    ">
        <strong>Action possible</strong><br><br>
        Ce rapport peut être connecté à un envoi
        <strong>email</strong>, <strong>Slack</strong>, <strong>Teams</strong> ou <strong>SMS</strong>
        pour alerter l’équipe qualité.
    </div>
    """,
            unsafe_allow_html=True,
        )

# -----------------------------
# Load data
# -----------------------------
benchmark_df = load_csv(BENCHMARK_RESULTS_PATH)
tuning_df = load_csv(YOLO_TUNING_RESULTS_PATH)
final_test_metrics_df = load_csv(FINAL_TEST_METRICS_PATH)
per_class_df = load_csv(FINAL_TEST_PER_CLASS_METRICS_PATH)
results_df = load_csv(RESULTS_PATH)

comparison_df = (
    pd.concat([benchmark_df, tuning_df], ignore_index=True)
    .drop_duplicates(subset="model_name")
    .sort_values("map50_95", ascending=False)
    .reset_index(drop=True)
)

best_model = comparison_df.iloc[0]

final_test_metrics = final_test_metrics_df.iloc[0]

demo_images = load_demo_images()

per_class_display_df = per_class_df.copy()
per_class_display_df["display_class_name"] = per_class_display_df["class_name"].map(
    lambda x: CLASS_LABELS.get(x, x)
)


# -----------------------------
# Header
# -----------------------------
st.title("Contrôle qualité automatisé des cartes électroniques")

st.markdown(
    """
##### Application de Computer Vision — Agnès REGAUD  
##### Du contrôle visuel manuel à une détection automatique des défauts sur cartes électroniques pour réduire les retours client et refidéliser.  
[LinkedIn](https://www.linkedin.com/in/agnes-regaud/) • [GitHub](https://github.com/agnesR23/controle-qualite-computer-vision-cartes-electroniques)
"""
)
st.markdown(
    """
    Application développée à partir du
    [PCB Defect Dataset disponible sur Kaggle]
    (https://www.kaggle.com/datasets/norbertelter/pcb-defect-dataset).
    """
)

st.success(
    """
### 🎯 Contexte métier

##### Un défaut qui échappe au contrôle qualité ne coûte pas seulement une carte à remplacer : il mobilise le SAV, relance la logistique et grignote la marge.
##### Cette application de Computer Vision permet de contrôler automatiquement et rapidement une carte électronique avant expédition :

- **inférence locale mesurée à environ 4 ms par image sur le jeu de test** ;
- **détection automatique de 6 types de défauts visibles** ;
- **localisation du défaut sur l’image** ;
- **rapport qualité automatique** : image contrôlée, défaut détecté, score de confiance et état du système.

##### À partir de ce rapport, l’entreprise peut déclencher une action qualité : alerte équipe, carte isolée, contrôle historisé, défauts récurrents suivis.
"""
)

st.info(
    """
### 🧭 Mode d’emploi

1. **Choisir ci-dessous une image de carte électronique parmi les 12 exemples.**
2. **Cliquer sur le bouton rouge « Cliquer ici pour détecter ».**
3. **Consulter le défaut localisé, le score de confiance et le rapport qualité.**
4. **Ouvrir la partie technique pour en savoir plus sur les données, les modèles et leurs performances.**
"""
)

st.markdown(
    """
<div style="
    padding: 14px 18px;
    border-radius: 10px;
    background-color: #f8fafc;
    border: 1px solid #d1d5db;
    font-size: 16px;
    line-height: 1.6;
">
    <strong>Rapport connectable à :</strong>
    📧 <span style="color:#2563eb; font-weight:600;">Email</span> ·
    📱 <span style="color:#16a34a; font-weight:600;">SMS</span> ·
    💬 <span style="color:#7c3aed; font-weight:600;">Slack</span> ·
    👥 <span style="color:#dc2626; font-weight:600;">Teams</span>
</div>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# 1. Dynamic or Static demonstration
# -----------------------------
st.divider()

api_available = check_api_health(API_URL)


demo_examples = []

for class_images in demo_images.values():
    for image in class_images:
        image = image.copy()
        image["display_name"] = f"Image {len(demo_examples) + 1}"
        demo_examples.append(image)

if not demo_examples:
    st.warning("Aucune image de démonstration disponible dans app/sample_images.")
else:
    st.markdown(
        """
<div style="
    display: flex;
    align-items: baseline;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 16px;
">
    <span style="font-size: 28px; font-weight: 700;">
        Tester l’application
    </span>
    <span style="font-size: 18px;">
        — À chaque clic, l’image est analysée par le modèle et le résultat est renvoyé dans l’application.
    </span>
</div>
""",
        unsafe_allow_html=True,
    )

    col_choice, _, col_result_title, col_defects_title = st.columns(
        [1, 0.35, 1, 0.75]
    )

    with col_choice:
        st.markdown("#### Choisir une image")

        selected_image = st.selectbox(
            "Choisir une image",
            options=demo_examples,
            format_func=lambda image: image["display_name"],
            label_visibility="collapsed",
        )

    with col_result_title:
        st.markdown("#### Résultat de détection")

    with col_defects_title:
        st.markdown("#### Défauts détectés")

    prediction_key = f"prediction_{selected_image['name']}"
    button_key = f"inference_button_{selected_image['name']}"

    if prediction_key not in st.session_state:
        st.session_state[prediction_key] = {
            "mode": None,
            "image_bytes": None,
            "status": None,
            "detected_classes": [],
            "detections": [],
            "response_time": None,
        }

    col_visuals, col_summary = st.columns([2.35, 0.75])

    with col_visuals:
        col_original, col_button, col_prediction = st.columns(
            [1, 0.35, 1],
            vertical_alignment="center",
        )

        with col_original:
            st.image(
                str(selected_image["original"]),
                use_container_width=True,
            )

        with col_button:
            run_inference = st.button(
                "Cliquer ici pour détecter",
                key=button_key,
                type="primary",
                use_container_width=True,
            )

        if run_inference:
            if api_available:
                try:
                    with st.spinner("Détection en cours..."):
                        response_start_time = time.perf_counter()

                        api_response = predict_image_with_api(
                            api_url=API_URL,
                            image_path=selected_image["original"],
                            timeout=15,
                        )

                        response_time = time.perf_counter() - response_start_time

                    annotated_image_base64 = api_response.get(
                        "annotated_image_base64"
                    )
                    detections = api_response.get("detections", [])
                    detected_classes = sorted(
                        {
                            detection["class_name"]
                            for detection in detections
                        }
                    )

                    if annotated_image_base64:
                        st.session_state[prediction_key] = {
                            "mode": "live",
                            "image_bytes": base64.b64decode(
                                annotated_image_base64
                            ),
                            "status": "Résultat généré.",
                            "detected_classes": detected_classes,
                            "detections": detections,
                            "response_time": response_time,
                        }
                    else:
                        st.session_state[prediction_key] = {
                            "mode": "unavailable",
                            "image_bytes": None,
                            "status": (
                                "Réponse incomplète. "
                                "Réessayez dans quelques instants."
                            ),
                            "detected_classes": [],
                            "detections": [],
                            "response_time": None,
                        }

                except RuntimeError as error:
                    st.session_state[prediction_key] = {
                        "mode": "unavailable",
                        "image_bytes": None,
                        "status": (
                            "Erreur pendant la détection. "
                            "Réessayez dans quelques instants."
                        ),
                        "detected_classes": [],
                        "detections": [],
                        "response_time": None,
                    }

                    st.markdown(f"Détail technique : {error}")

            else:
                st.session_state[prediction_key] = {
                    "mode": "unavailable",
                    "image_bytes": None,
                    "status": (
                        "Système de détection indisponible. "
                        "Réessayez dans quelques instants."
                    ),
                    "detected_classes": [],
                    "detections": [],
                    "response_time": None,
                }

        prediction = st.session_state[prediction_key]

        with col_prediction:
            if (
                prediction["mode"] == "live"
                and prediction["image_bytes"] is not None
            ):
                st.image(
                    prediction["image_bytes"],
                    use_container_width=True,
                )

            elif prediction["mode"] == "unavailable":
                st.warning(prediction["status"])
            else:
                st.markdown(
                    """
                    <div style="
                        width: 100%;
                        aspect-ratio: 1 / 1;
                    ">
                        <div style="
                            box-sizing: border-box;
                            padding: 16px;
                            border: 1px solid #c7ddf4;
                            border-radius: 8px;
                            background-color: #eaf3fc;
                            color: #155a96;
                        ">
                            Le résultat apparaîtra ici après avoir lancé la détection.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        if prediction["response_time"] is not None:
            _, _, col_response_time = st.columns([1, 0.35, 1])

            with col_response_time:
                response_time_display = (
                    f'{prediction["response_time"]:.2f}'.replace(".", ",")
                )

                st.info(
                    f"""
            ### ⏱️ Temps de réponse : {response_time_display} s

            Durée entre l’envoi de l’image par Streamlit et la réception du résultat : transfert vers l’API, calcul du modèle et retour de la réponse.
            """
                )
    with col_summary:
        render_detection_summary(
            detected_classes=prediction["detected_classes"],
            api_available=api_available,
        )

    render_quality_report(
        image_name=selected_image["display_name"],
        detected_classes=prediction["detected_classes"],
        detections=prediction["detections"],
        prediction_mode=prediction["mode"],
        api_available=api_available,
    )

# -----------------------------
# 2. Technical verification
# -----------------------------
st.divider()

st.markdown(
    """
<style>
[data-testid="stExpander"] summary p {
    font-size: 22px !important;
    font-weight: 700 !important;
}
</style>
""",
    unsafe_allow_html=True,
)

with st.expander("Partie technique du projet", expanded=False):

    # -----------------------------
    # A. Model comparison
    # -----------------------------
    st.markdown(
        """
    <style>
    .data-preparation-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 17px;
    }
    .data-preparation-table th {
        font-size: 18px;
        font-weight: 700;
        background-color: #f1f5f9;
    }
    .data-preparation-table th,
    .data-preparation-table td {
        padding: 12px 14px;
        border: 1px solid #d1d5db;
        text-align: left;
        vertical-align: top;
    }
    .data-preparation-table td:first-child {
        width: 25%;
        font-weight: 700;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )
    st.subheader("Préparation des données")

    st.markdown(
        """
    #### Ce qui a été observé

    - Le dataset Kaggle contenait **10 668 images** : **2 667 images sources** et trois copies transformées de chacune — rotations à **90°** et **270°**, et variation de **luminosité**.
    - Il était déjà réparti en **8 534 images d’entraînement**, **1 066 de validation** et **1 068 de test**.
    - Mais pour **1 556 images sources sur 2 667**, soit **58 %**, les copies d’une même image étaient réparties dans plusieurs jeux, créant une fuite de données.
    - Et **10 doublons**, répartis en **4 groupes**, ont également été détectés, avec parfois des annotations différentes.
    """
    )

    st.markdown(
        "**Pour corriger ces problèmes, les jeux de données ont été reconstruits avant l’entraînement.**"
    )
    st.markdown("#### Ce qui a été fait")

    data_processing_df = pd.DataFrame(
        [
            {
                "Traitement": "Doublons",
                "Réalisation": (
                    "Suppression des doublons : "
                    "2 657 images sources uniques conservées."
                ),
            },
            {
                "Traitement": "Fuite de données",
                "Réalisation": (
                    "Regroupement de chaque image source avec ses "
                    "copies transformées."
                ),
            },
            {
                "Traitement": "Nouvelle répartition",
                "Réalisation": (
                    "Séparation stratifiée en entraînement, validation et test."
                ),
            },
            {
                "Traitement": "Copies transformées",
                "Réalisation": (
                    "Conservation uniquement dans le jeu d’entraînement."
                ),
            },
            {
                "Traitement": "Vérification finale",
                "Réalisation": (
                    "Aucune image source commune entre les trois jeux."
                ),
            },
        ]
    )

    st.markdown(
        data_processing_df.to_html(
            index=False,
            classes="data-preparation-table",
            escape=False,
        ),
        unsafe_allow_html=True,
    )
    st.markdown("#### Répartition avant et après préparation")

    data_split_df = pd.DataFrame(
        [
            {
                "Jeu": "Entraînement",
                "Données fournies": "8 534 images",
                "Après préparation": (
                    "8 500 images : 2 125 images sources "
                    "et leurs copies transformées"
                ),
            },
            {
                "Jeu": "Validation",
                "Données fournies": "1 066 images",
                "Après préparation": "266 images sources uniquement",
            },
            {
                "Jeu": "Test",
                "Données fournies": "1 068 images",
                "Après préparation": "266 images sources uniquement",
            },
            {
                "Jeu": "Total",
                "Données fournies": "10 668 images",
                "Après préparation": "9 032 images",
            },
        ]
    )

    st.markdown(
        data_split_df.to_html(
            index=False,
            classes="data-preparation-table",
            escape=False,
        ),
        unsafe_allow_html=True,
    )

    st.divider()

    st.subheader("Comparaison des modèles")

    st.markdown(
        """
    ##### Comparaison des modèles et variantes testés avec les mêmes métriques, puis classement des résultats par mAP50-95.
    """
    )


    st.markdown(
        """
    ##### Conditions du benchmark

    - **Faster R-CNN** : entraîné sur 3 000 images stratifiées pendant 1 époque sur CPU, afin de limiter le temps d’exécution.
    - **YOLOv8s baseline et variantes** : entraînés sur les 8 500 images d’entraînement avec accélération MPS.
    - Tous les modèles sont comparés avec les mêmes métriques sur les 266 images du jeu de validation.

    **Les temps d’inférence restent indicatifs : Faster R-CNN a été mesuré sur CPU et YOLOv8s sur MPS.**
    """
    )

    st.markdown(
        """
        ##### Les résultats sont classés par mAP50-95, la métrique principale retenue pour comparer la qualité globale de détection.
        """
    )

    st.markdown("### Résultats classés")

    comparison_display_df = add_experiment_parameters(comparison_df)

    comparison_display_df = comparison_display_df[
        [
            "model_name",
            "parameters",
            "map50_95",
            "map50",
            "training_time",
            "inference_time",
        ]
    ].copy()

    comparison_display_df = comparison_display_df.rename(
        columns={
            "model_name": "Modèle / configuration",
            "parameters": "Paramètres testés",
            "map50_95": "mAP50-95",
            "map50": "mAP50",
            "training_time": "Temps entraînement (s)",
            "inference_time": "Temps inférence — validation (s)",
        }
    )

    comparison_display_df = format_metric_columns(
        comparison_display_df,
        [
        "mAP50-95",
        "mAP50",
        "Temps entraînement (s)",
        "Temps inférence — validation (s)",
        ],
    )

    best_model_name = str(best_model["model_name"])

    comparison_styler = comparison_display_df.style.apply(
        lambda row: [
            "background-color: #dcfce7; font-weight: 600;"
            if row["Modèle / configuration"] == best_model_name
            else ""
            for _ in row
        ],
        axis=1,
    )
    comparison_styler = comparison_styler.format(
        {
            "mAP50-95": "{:.3f}",
            "mAP50": "{:.3f}",
            "Temps entraînement (s)": "{:.1f}",
            "Temps inférence — validation (s)": "{:.3f}",
        }
    )

    st.dataframe(
        comparison_styler,
        use_container_width=True,
        hide_index=True,
    )


    st.markdown(
        """
    ##### Les entraînements ont été suivis avec MLflow pour tracer les métriques et conserver les artefacts.
    """
    )

    # -----------------------------
    # B. Model selection
    # -----------------------------
    st.divider()
    st.subheader("Sélection du modèle")

    st.success(
    """
    ### Décision

    La configuration surlignée, **YOLOv8s entraîné pendant 20 époques**, est retenue car elle obtient la meilleure mAP50-95 sur le jeu de validation.

    Elle améliore la qualité globale de détection par rapport à la baseline YOLOv8s, tout en conservant un temps d’inférence adapté à une application interactive.

    Le jeu de test n’a pas participé à la sélection. Il est utilisé une seule fois pour mesurer les performances finales du modèle retenu.
    """
    )
    st.markdown("### Résultats finaux sur le jeu de test")
    st.markdown(
        """
    #### Résultats mesurés une seule fois sur 266 images indépendantes, après la sélection du modèle.
    """
    )

    response_time = (
        prediction.get("response_time")
        if demo_examples
        else None
    )

    response_time_display = (
        f"{response_time:.2f} s".replace(".", ",")
        if response_time is not None
        else "À mesurer lors d’une détection"
    )

    final_results_df = pd.DataFrame(
        [
            {
                "Indicateur": "mAP50-95",
                "Valeur": f'{final_test_metrics["map50_95"]:.3f}',
                "Signification": (
                    "Moyenne des performances sur les six catégories, "
                    "avec des seuils de recouvrement entre la boîte prédite "
                    "et la zone réelle allant de 50 % à 95 %. Le score de 0,608 sur 1, "
                    "inférieur au mAP50 de 0,985, montre que le modèle détecte très "
                    "bien les défauts, mais perd en performance lorsque la boîte doit "
                    "épouser plus précisément leur emplacement."
                ),
            },
            {
                "Indicateur": "mAP50",
                "Valeur": f'{final_test_metrics["map50"]:.3f}',
                "Signification": (
                    "Performance moyenne de détection sur les six catégories "
                    "avec un seuil de recouvrement de 50 %. Le score de 0,985 "
                    "indique une détection très performante avec ce critère."
                ),
            },
            {
                "Indicateur": "Précision",
                "Valeur": f'{final_test_metrics["precision"]:.3f}',
                "Signification": (
                    "96,4 % des zones signalées par le modèle correspondent "
                    "à un défaut réel : les fausses alertes restent limitées."
                ),
            },
            {
                "Indicateur": "Rappel",
                "Valeur": f'{final_test_metrics["recall"]:.3f}',
                "Signification": (
                    "99,1 % des défauts réels sont détectés : très peu de défauts "
                    "risquent d’échapper au contrôle avant expédition."
                ),
            },
            {
                "Indicateur": "Temps de calcul local",
                "Valeur": (
                    f'{final_test_metrics["inference_time"] * 1000:.1f} ms'
                ),
                "Signification": (
                    "Temps moyen nécessaire au modèle seul pour analyser une image "
                    "dans l’environnement local de test."
                ),
            },
            {
                "Indicateur": "Temps de réponse de l’application",
                "Valeur": response_time_display,
                "Signification": (
                    "Temps réellement attendu par l’utilisateur, comprenant "
                    "le transfert de l’image, le calcul du modèle et le retour "
                    "du résultat par l’API."
                ),
            },
        ]
    )

    st.markdown(
        """
    <style>
    .final-results-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 17px;
    }
    .final-results-table th {
        font-size: 18px;
        font-weight: 700;
        text-align: left;
        background-color: #f1f5f9;
    }
    .final-results-table th,
    .final-results-table td {
        padding: 12px 14px;
        border: 1px solid #d1d5db;
        vertical-align: top;
    }
    .final-results-table td:first-child {
        font-weight: 700;
    }
    .final-results-table td:nth-child(2) {
        font-weight: 700;
        white-space: nowrap;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        final_results_df.to_html(
            index=False,
            classes="final-results-table",
            escape=False,
        ),
        unsafe_allow_html=True,
    )
    # -----------------------------
    # C. Technical details
    # -----------------------------

    st.markdown("### Analyse des performances du modèle retenu")

    st.markdown("#### Détection, classification et erreurs — jeu de test")
    st.image(TEST_DIAGNOSTICS_PATH, width=1500)

    col_class, col_training = st.columns(2)

    with col_class:
        st.markdown("#### Performance par type de défaut — jeu de test")

        fig_class = plot_horizontal_metric(
            df=per_class_display_df,
            label_col="display_class_name",
            value_col="ap50_95",
            title="AP50-95 par classe — jeu de test",
            xlabel="AP50-95",
            figsize=(5, 3),
        )
        st.pyplot(fig_class, use_container_width=True)
        plt.close(fig_class)

        st.info(
            """
        #### Lecture du graphique des performances par type de défaut

        L’AP50-95 par classe évalue :

        - la détection des défauts réels ;
        - la limitation des zones signalées à tort ;
        - la précision de localisation des défauts.

        Les résultats sont relativement homogènes : l’AP50-95 varie de **0,570 à 0,645** selon les catégories. Aucun type de défaut ne présente donc de décrochage important.
        """
        )

    with col_training:
        st.markdown("#### Évolution de l’entraînement")

        st.markdown(
            "##### **Courbes de validation enregistrées par Ultralytics pendant "
            "les 20 époques d’entraînement du modèle YOLOv8s retenu.**"
        )

        fig_train, ax_train = plt.subplots(figsize=(5, 3))

        ax_train.plot(
            results_df["metrics/mAP50-95(B)"],
            label="mAP50-95",
        )
        ax_train.plot(
            results_df["metrics/mAP50(B)"],
            label="mAP50",
        )

        ax_train.set_title(
            "mAP par epoch — YOLOv8s epochs 20",
            fontsize=10,
        )
        ax_train.set_xlabel("Epoch", fontsize=9)
        ax_train.set_ylabel("Score", fontsize=9)
        ax_train.set_ylim(0, 1)
        ax_train.tick_params(axis="both", labelsize=8)
        ax_train.legend(fontsize=8)
        ax_train.grid(alpha=0.3)

        ax_train.spines["top"].set_visible(False)
        ax_train.spines["right"].set_visible(False)

        fig_train.tight_layout()

        st.pyplot(fig_train, use_container_width=True)
        plt.close(fig_train)

    st.divider()

    st.markdown("### Stack technique")

    st.markdown(
        """
- **Modélisation :** Python 3.11 · YOLOv8 · PyTorch · Torchvision
- **Expérimentation :** MLflow · Pandas · Matplotlib
- **Application et API :** Streamlit · FastAPI
- **Déploiement :** Docker · Hugging Face Spaces · Streamlit Community Cloud
- **Qualité du code :** Pytest · Ruff
- **Automatisation :** GitHub Actions · tests CI et requêtes planifiées pour réactiver l’API et le dashboard après leur mise en veille
"""
    )