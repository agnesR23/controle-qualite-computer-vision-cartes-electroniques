# Copyright (C) 2026 Agnès REGAUD
# SPDX-License-Identifier: AGPL-3.0-only

# -----------------------------
# Imports
# -----------------------------
from pathlib import Path
import base64

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
PER_CLASS_METRICS_PATH = ASSETS_DIR / "per_class_metrics.csv"
RESULTS_PATH = ASSETS_DIR / "results.csv"
CONFUSION_MATRIX_PATH = ASSETS_DIR / "confusion_matrix_normalized.png"

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
    prediction_mode: str | None,
) -> None:
    """
    Display detected classes, prediction mode and automatic quality report.
    """
    st.markdown("#### Défauts détectés")

    for class_name in CLASS_NAMES:
        label = CLASS_LABELS.get(class_name, class_name)
        checked = "✅" if class_name in detected_classes else "☐"
        st.markdown(f"{checked} {label}")

    st.markdown("#### Mode de prédiction")

    live_checked = "✅" if prediction_mode == "live" else "☐"
    unavailable_checked = "✅" if prediction_mode == "unavailable" else "☐"

    st.markdown(f"{live_checked} API d’inférence active")
    st.markdown(f"{unavailable_checked} API indisponible")

    st.markdown("#### **Score de confiance**")
    st.caption(
        "Le score affiché sur l’image annotée correspond au niveau de confiance du modèle pour le défaut détecté."
    )


def render_quality_report(
    image_name: str,
    detected_classes: list[str],
    detections: list[dict],
    prediction_mode: str | None,
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

    prediction_mode_label = {
        "live": "API d’inférence active",
        "unavailable": "API indisponible",
        None: "En attente de prédiction",
    }.get(prediction_mode, "En attente de prédiction")

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
            {"Information": "Mode de prédiction", "Valeur": prediction_mode_label},
            {"Information": "Score de confiance", "Valeur": confidence_display},
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
per_class_df = load_csv(PER_CLASS_METRICS_PATH)
results_df = load_csv(RESULTS_PATH)

comparison_df = (
    pd.concat([benchmark_df, tuning_df], ignore_index=True)
    .drop_duplicates(subset="model_name")
    .sort_values("map50_95", ascending=False)
    .reset_index(drop=True)
)

best_model = comparison_df.iloc[0]

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
[LinkedIn](https://www.linkedin.com/in/agnes-regaud/) • [GitHub](https://github.com/agnesR23?tab=repositories&q=&type=&language=&sort=name)
"""
)

st.success(
    """
### 🎯 Contexte métier

##### Un défaut qui échappe au contrôle qualité ne coûte pas seulement une carte à remplacer : il mobilise le SAV, relance la logistique et grignote la marge.
##### Cette application de Computer Vision permet de contrôler automatiquement et rapidement une carte électronique avant expédition :

- **analyse en 0,004 seconde par image** ;
- **détection automatique de 6 types de défauts visibles** ;
- **localisation du défaut sur l’image** ;
- **rapport qualité automatique** : image contrôlée, défaut détecté, score de confiance et mode de prédiction.

##### À partir de ce rapport, l’entreprise peut déclencher une action qualité : alerte équipe, carte isolée, contrôle historisé, défauts récurrents suivis.
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
    st.sidebar.header("Tester l’application")

    st.sidebar.markdown("### Image à contrôler")
    selected_image = st.sidebar.selectbox(
        "Sélectionner une image :",
        options=demo_examples,
        format_func=lambda image: image["display_name"],
    )

    st.sidebar.caption(
        "Cette image sera envoyée au modèle, qui va prédire la localisation et le type de défaut."
    )

    st.sidebar.markdown("### Détection de défaut")

    prediction_key = f"prediction_{selected_image['name']}"
    button_key = f"inference_button_{selected_image['name']}"

    if prediction_key not in st.session_state:
        st.session_state[prediction_key] = {
            "mode": None,
            "image_bytes": None,
            "status": None,
            "detected_classes": [],
            "detections": [],
        }

    run_inference = st.sidebar.button(
        "Cliquer ici pour détecter le défaut",
        key=button_key,
        type="primary",
        use_container_width=True,
    )

    if api_available:
        st.sidebar.success("API live disponible")
    else:
        st.sidebar.warning(
            "API indisponible ou en cours de démarrage. Réessayez dans quelques instants."
        )

    if run_inference:
        if api_available:
            try:
                with st.spinner("Détection live en cours..."):
                    api_response = predict_image_with_api(
                        api_url=API_URL,
                        image_path=selected_image["original"],
                        timeout=15,
                    )

                annotated_image_base64 = api_response.get("annotated_image_base64")
                detections = api_response.get("detections", [])
                detected_classes = sorted(
                    {detection["class_name"] for detection in detections}
                )

                if annotated_image_base64:
                    st.session_state[prediction_key] = {
                        "mode": "live",
                        "image_bytes": base64.b64decode(annotated_image_base64),
                        "status": "Résultat généré en live via l’API.",
                        "detected_classes": detected_classes,
                        "detections": detections,
                    }
                else:
                    st.session_state[prediction_key] = {
                        "mode": "unavailable",
                        "image_bytes": None,
                        "status": "Réponse API incomplète. Réessayez dans quelques instants.",
                        "detected_classes": [],
                        "detections": [],
                    }

            except RuntimeError as error:
                st.session_state[prediction_key] = {
                    "mode": "unavailable",
                    "image_bytes": None,
                    "status": "Erreur pendant l’appel API. Réessayez dans quelques instants.",
                    "detected_classes": [],
                    "detections": [],
                }
                
                st.sidebar.caption(f"Détail technique : {error}")

        else:
            st.session_state[prediction_key] = {
                "mode": "unavailable",
                "image_bytes": None,
                "status": "API indisponible ou en cours de démarrage. Réessayez dans quelques instants.",
                "detected_classes": [],
                "detections": [],
            }

    prediction = st.session_state[prediction_key]


    col_original, col_prediction, col_summary = st.columns([1, 1, 0.75])

    with col_original:
        st.markdown("#### Visualisation image à contrôler :")
        st.image(str(selected_image["original"]), width=500)

    with col_prediction:
        st.markdown("#### Résultat de détection de défaut :")

        if prediction["mode"] == "live" and prediction["image_bytes"] is not None:
            st.image(prediction["image_bytes"], width=500)

        elif prediction["mode"] == "unavailable":
            st.warning(prediction["status"])

        else:
            st.info("Le résultat apparaîtra ici après avoir lancé la détection.")

    with col_summary:
        render_detection_summary(
            detected_classes=prediction["detected_classes"],
            prediction_mode=prediction["mode"],
        )

    render_quality_report(
        image_name=selected_image["display_name"],
        detected_classes=prediction["detected_classes"],
        detections=prediction["detections"],
        prediction_mode=prediction["mode"],
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
    st.subheader("Comparaison des modèles")

    st.markdown(
        """
    ##### Comparaison des modèles et variantes testés avec les mêmes métriques, puis classement des résultats par mAP50-95.
    """
    )


    st.markdown(
        """
        ##### Le benchmark compare le modèle de référence Faster R-CNN, une baseline YOLOv8s et plusieurs configurations YOLO optimisées.

        - **Faster R-CNN** : modèle de référence, testé en CPU avec une configuration allégée.
        - **YOLOv8s baseline** : premier entraînement YOLO, utilisé comme point de départ.
        - **YOLOv8s optimisé** : variantes testées sur les epochs, le learning rate et la batch size.
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
            "inference_time": "Temps inférence (s)",
        }
    )

    comparison_display_df = format_metric_columns(
        comparison_display_df,
        ["mAP50-95", "mAP50", "Temps entraînement (s)", "Temps inférence (s)"],
    )

    st.dataframe(
        comparison_display_df,
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

    col_best_1, col_best_2, col_best_3 = st.columns(3)

    with col_best_1:
        st.metric(
            label="Modèle retenu",
            value=str(best_model["model_name"]),
        )

    with col_best_2:
        st.metric(
            label="mAP50-95",
            value=f'{best_model["map50_95"]:.3f}',
        )

    with col_best_3:
        st.metric(
            label="Temps d’inférence (s)",
            value=f'{best_model["inference_time"]:.3f}',
        )

    st.success(
        """
    ### Décision
    Le modèle retenu présente le meilleur compromis entre performance globale et rapidité d’inférence.
    """
    )

    # -----------------------------
    # C. Technical details
    # -----------------------------

    st.markdown("### Analyse des performances du modèle retenu")

    col_confusion, col_right = st.columns([1.5, 1])

    with col_confusion:
        st.markdown("#### Matrice de confusion")
        st.image(CONFUSION_MATRIX_PATH, use_container_width=True)

    with col_right:
        st.markdown("#### Performance par type de défaut")

        fig_class = plot_horizontal_metric(
            df=per_class_display_df,
            label_col="display_class_name",
            value_col="ap50_95",
            title="AP50-95 par classe",
            xlabel="AP50-95",
            figsize=(5, 3),
        )
        st.pyplot(fig_class)

        st.markdown("#### Évolution de l’entraînement")
        st.caption(
            "Courbe issue du fichier results.csv généré automatiquement par Ultralytics."
        )

        fig_train, ax_train = plt.subplots(figsize=(4.2, 2.2))

        ax_train.plot(results_df["metrics/mAP50-95(B)"], label="mAP50-95")
        ax_train.plot(results_df["metrics/mAP50(B)"], label="mAP50")

        ax_train.set_title("mAP par epoch — YOLOv8s epochs 20", fontsize=10)
        ax_train.set_xlabel("Epoch", fontsize=9)
        ax_train.set_ylabel("Score", fontsize=9)
        ax_train.set_ylim(0, 1)
        ax_train.tick_params(axis="both", labelsize=8)
        ax_train.legend(fontsize=8)
        ax_train.grid(alpha=0.3)

        ax_train.spines["top"].set_visible(False)
        ax_train.spines["right"].set_visible(False)

        fig_train.tight_layout()

        st.pyplot(fig_train, use_container_width=False)



