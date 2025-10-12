"""Détection d'émotions sur les images synchronisées."""

from __future__ import annotations

import io
import json
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Détection optionnelle avec le modèle FER (Facial Emotion Recognition).
_FER_IMPORT_ERROR = ""

try:  # pragma: no cover - dépendance optionnelle
    from fer import FER  # type: ignore

    _FER_DISPONIBLE = True
except Exception as exc:  # pragma: no cover - dépendance optionnelle
    FER = None  # type: ignore
    _FER_DISPONIBLE = False
    _FER_IMPORT_ERROR = str(exc)


@st.cache_resource(show_spinner=False)
def charger_modele_emotions() -> tuple[Any | None, str]:
    """Instancie le détecteur FER si disponible."""
    if not _FER_DISPONIBLE:
        details = f" Détail de l'erreur : {_FER_IMPORT_ERROR}" if _FER_IMPORT_ERROR else ""
        return None, (
            "Le paquet `fer` n'est pas installé ou a échoué au chargement. Installez-le avec"
            " `pip install fer` puis redémarrez l'application pour activer la détection d'émotions." + details
        )
    try:
        detector = FER()
        return detector, "Modèle FER initialisé (architecture CNN pré-entraînée sur FER2013)."
    except Exception as exc:  # pragma: no cover - dépendance optionnelle
        return None, f"Échec de l'initialisation du modèle FER : {exc}"


def recommander_modele_emotions() -> str:
    """Retourne une recommandation de modèle à afficher dans l'interface."""
    return (
        "Nous recommandons d'utiliser le modèle **FER** (Facial Emotion Recognition) reposant sur"
        " un réseau de neurones convolutif pré-entraîné sur la base FER2013. Il fonctionne sur CPU,"
        " ne nécessite qu'une dépendance Python (`pip install fer`) et offre un bon compromis"
        " précision/performance pour des images extraites à 1 fps."
    )


def _analyser_image(detector: Any, image_bytes: bytes) -> list[dict[str, Any]]:
    """Applique le détecteur sur une image et renvoie une liste de résultats par visage."""
    if detector is None:
        return []

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return []

    arr = np.array(img)
    try:
        detections = detector.detect_emotions(arr)
    except Exception:  # pragma: no cover - dépendance optionnelle
        return []

    sorties: list[dict[str, Any]] = []
    for face_idx, resultat in enumerate(detections or []):
        emotions = resultat.get("emotions", {})
        if not emotions:
            continue
        emotion_predite, score = max(emotions.items(), key=lambda item: item[1])
        sorties.append(
            {
                "face_id": face_idx,
                "emotion_predite": str(emotion_predite),
                "score": float(score),
                "scores_emotions": emotions,
                "bbox": resultat.get("box", []),
            }
        )
    return sorties


def ui_emotions_images(df_images: pd.DataFrame | None) -> None:
    """Interface Streamlit pour lancer la détection d'émotions sur les images importées."""
    st.subheader("Détection d'émotions (images synchronisées)")
    st.markdown(recommander_modele_emotions())

    detector, message_modele = charger_modele_emotions()
    st.caption(message_modele)
    st.write(
        "Cette analyse s'appuie exclusivement sur les images importées dans l'onglet « 1. Données »."
        " Chaque fichier sélectionné est transmis au modèle FER qui détecte les visages et associe"
        " une émotion dominante à chacun."
    )

    images_store = st.session_state.get("images_store", []) or []
    if not images_store:
        st.info("Importez d'abord des images (onglet 1. Données) pour lancer l'analyse.")
        return

    if detector is None:
        st.warning(
            "Le modèle de détection n'est pas disponible. Installez `fer` puis relancez l'application"
            " pour activer cette section."
        )
        return

    noms_images = [it.get("name") for it in images_store if isinstance(it, dict) and it.get("name")]
    if not noms_images:
        st.info("Aucune image valide n'est disponible dans la session.")
        return

    default_selection = noms_images[: min(12, len(noms_images))]
    selection = st.multiselect(
        "Images à analyser",
        options=noms_images,
        default=default_selection,
        help="Sélectionnez un sous-ensemble si vous disposez de nombreuses images.",
    )

    if not selection:
        st.info("Sélectionnez au moins une image pour lancer l'analyse.")
        return

    lancer = st.button("Analyser les émotions sur la sélection")
    if not lancer:
        return

    resultats: list[dict[str, Any]] = []
    df_images = df_images if isinstance(df_images, pd.DataFrame) else pd.DataFrame()

    for nom in selection:
        bytes_img = None
        for item in images_store:
            if isinstance(item, dict) and item.get("name") == nom:
                bytes_img = item.get("bytes")
                break
        if bytes_img is None:
            continue

        detections = _analyser_image(detector, bytes_img)
        if not detections:
            resultats.append(
                {
                    "fichier_image": nom,
                    "t_image": _recuperer_timestamp(df_images, nom),
                    "face_id": np.nan,
                    "emotion_predite": "aucune",
                    "score": 0.0,
                    "scores_emotions": {},
                    "bbox": [],
                }
            )
            continue

        for det in detections:
            ligne = {
                "fichier_image": nom,
                "t_image": _recuperer_timestamp(df_images, nom),
                "face_id": det.get("face_id"),
                "emotion_predite": det.get("emotion_predite", ""),
                "score": float(det.get("score", 0.0)),
                "scores_emotions": det.get("scores_emotions", {}),
                "bbox": det.get("bbox", []),
            }
            resultats.append(ligne)

    if not resultats:
        st.warning("Aucun visage ni émotion détectés sur la sélection.")
        return

    df_res = pd.DataFrame(resultats)
    df_res["score"] = df_res["score"].astype(float)
    df_res["scores_emotions_json"] = df_res["scores_emotions"].apply(
        lambda d: json.dumps(d, ensure_ascii=False) if isinstance(d, dict) else "{}"
    )

    st.session_state["df_emotions"] = df_res.copy()

    st.markdown("#### Résultats détaillés")
    st.dataframe(
        df_res[[
            "fichier_image",
            "t_image",
            "face_id",
            "emotion_predite",
            "score",
            "bbox",
            "scores_emotions_json",
        ]],
        use_container_width=True,
    )

    st.markdown("#### Distribution des émotions détectées")
    distrib = (
        df_res[df_res["emotion_predite"] != "aucune"]["emotion_predite"].value_counts().reset_index()
    )
    distrib.columns = ["emotion", "occurrences"]
    if distrib.empty:
        st.info("Aucune émotion dominante n'a été détectée.")
    else:
        st.bar_chart(data=distrib.set_index("emotion"))

    st.markdown("#### Export")
    csv = df_res.drop(columns=["scores_emotions"]).to_csv(index=False).encode("utf-8")
    st.download_button(
        "Télécharger les résultats (CSV)",
        data=csv,
        file_name="emotions_images.csv",
        mime="text/csv",
    )


def _recuperer_timestamp(df_images: pd.DataFrame, nom: str) -> float | np.nan:
    """Récupère la valeur t_image correspondante si disponible."""
    if df_images is None or df_images.empty:
        return np.nan
    try:
        filt = df_images[df_images["fichier_image"] == nom]
        if filt.empty:
            return np.nan
        val = filt.iloc[0].get("t_image", np.nan)
        return float(val) if pd.notna(val) else np.nan
    except Exception:
        return np.nan
