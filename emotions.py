"""Détection d'émotions sur les images synchronisées."""

from __future__ import annotations

import io
import json
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from PIL import Image, ImageDraw, ImageFont

# Détection optionnelle avec le modèle FER (Facial Emotion Recognition).
_FER_IMPORT_ERROR = ""
_CV2_IMPORT_ERROR = ""
_MOVIEPY_IMPORT_ERROR = ""

try:  # pragma: no cover - dépendance optionnelle
    import cv2  # type: ignore

    _CV2_DISPONIBLE = True
except Exception as exc:  # pragma: no cover - dépendance optionnelle
    cv2 = None  # type: ignore
    _CV2_DISPONIBLE = False
    _CV2_IMPORT_ERROR = str(exc)

try:  # pragma: no cover - dépendance optionnelle
    from fer import FER  # type: ignore

    _FER_DISPONIBLE = True
except Exception as exc:  # pragma: no cover - dépendance optionnelle
    FER = None  # type: ignore
    _FER_DISPONIBLE = False
    _FER_IMPORT_ERROR = str(exc)

try:  # pragma: no cover - dépendance optionnelle
    import importlib.util as _importlib_util

    _MOVIEPY_DISPONIBLE = _importlib_util.find_spec("moviepy.editor") is not None
except Exception as exc:  # pragma: no cover - dépendance optionnelle
    _MOVIEPY_DISPONIBLE = False
    _MOVIEPY_IMPORT_ERROR = str(exc)


class _Cv2EmotionDetector:
    """Détecteur d'émotions de secours basé sur OpenCV.

    Cette implémentation s'appuie sur des cascades de Haar pour détecter les visages
    et les sourires. Elle fournit une estimation grossière de l'émotion dominante :
    « heureux » si un sourire est détecté, sinon « neutre ». Ce détecteur permet de
    proposer une analyse minimale lorsque `fer` n'est pas disponible.
    """

    def __init__(self) -> None:
        if cv2 is None:
            raise RuntimeError("OpenCV n'est pas disponible dans l'environnement courant.")

        base_path = getattr(cv2.data, "haarcascades", "")
        face_path = base_path + "haarcascade_frontalface_default.xml"
        smile_path = base_path + "haarcascade_smile.xml"

        self._face_cascade = cv2.CascadeClassifier(face_path)
        self._smile_cascade = cv2.CascadeClassifier(smile_path)

        if self._face_cascade.empty():
            raise RuntimeError(
                "Impossible de charger le classifieur de visages OpenCV (haarcascade_frontalface_default)."
            )
        if self._smile_cascade.empty():
            raise RuntimeError(
                "Impossible de charger le classifieur de sourires OpenCV (haarcascade_smile)."
            )

    def detect_emotions(self, image: np.ndarray) -> list[dict[str, Any]]:  # pragma: no cover - dépendance optionnelle
        if image is None or image.size == 0:
            return []

        if image.ndim == 2:
            gray = image
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        gray = cv2.equalizeHist(gray)

        faces = self._face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=6,
            minSize=(32, 32),
        )

        resultats: list[dict[str, Any]] = []
        for (x, y, w, h) in faces:
            roi_gray = gray[y : y + h, x : x + w]
            smiles = self._smile_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.7,
                minNeighbors=22,
                minSize=(15, 15),
            )

            a_sourire = len(smiles) > 0
            if a_sourire:
                scores = {"heureux": 0.9, "neutre": 0.1, "triste": 0.0}
            else:
                scores = {"heureux": 0.1, "neutre": 0.7, "triste": 0.2}

            resultats.append(
                {
                    "box": [int(x), int(y), int(w), int(h)],
                    "emotions": scores,
                }
            )

        return resultats


def _message_erreur_fer() -> str:
    """Construit un message d'aide en cas d'échec de l'import du paquet FER."""

    message = (
        "Le paquet `fer` n'est pas installé ou a échoué au chargement. Installez-le avec"
        " `pip install fer` puis redémarrez l'application pour activer la détection d'émotions."
    )
    if not _FER_IMPORT_ERROR:
        return message

    if "moviepy" in _FER_IMPORT_ERROR.lower():
        message += (
            " Le paquet `moviepy` est également requis par `fer`. Installez-le avec"
            " `pip install moviepy`."
        )
        if not _MOVIEPY_DISPONIBLE:
            details = " (aucun module `moviepy.editor` détecté dans l'environnement courant.)"
        else:
            details = " (module `moviepy` détecté mais `fer` ne parvient toujours pas à l'utiliser.)"
        message += details
        message += (
            " Vérifiez que vous utilisez le même environnement Python pour Streamlit et la"
            " commande d'installation (`python -m pip install moviepy`). Vous pouvez"
            " contrôler cela avec `python -m pip show moviepy` ou `pip show moviepy`."
            " En dernier recours, réinstallez avec `python -m pip install --upgrade --force-reinstall"
            " moviepy` puis redémarrez l'application."
        )

    message += f" Détail de l'erreur : {_FER_IMPORT_ERROR}"
    return message


@st.cache_resource(show_spinner=False)
def charger_modele_emotions() -> tuple[Any | None, str]:
    """Instancie un détecteur d'émotions."""

    messages: list[str] = []

    if _FER_DISPONIBLE:
        try:
            detector = FER()
        except Exception as exc:  # pragma: no cover - dépendance optionnelle
            messages.append(f"Échec de l'initialisation du modèle FER : {exc}")
        else:
            return detector, "Modèle FER initialisé (architecture CNN pré-entraînée sur FER2013)."
    else:
        messages.append(_message_erreur_fer())

    if _CV2_DISPONIBLE:
        try:
            detector_cv2 = _Cv2EmotionDetector()
        except Exception as exc:  # pragma: no cover - dépendance optionnelle
            messages.append(f"Échec de l'initialisation du détecteur OpenCV : {exc}")
        else:
            resume = (
                "Détecteur simplifié basé sur OpenCV (détection de sourires avec cascades de Haar)."
            )
            if messages:
                resume += " " + " ".join(messages)
            return detector_cv2, resume
    else:
        details = f" Détail de l'erreur : {_CV2_IMPORT_ERROR}" if _CV2_IMPORT_ERROR else ""
        messages.append(
            "Le paquet `opencv-python` est requis pour la détection de visages. Installez-le"
            " avec `pip install opencv-python` puis redémarrez l'application." + details
        )

    if not messages:
        messages.append("Aucun détecteur d'émotions n'est disponible dans l'environnement courant.")

    return None, " ".join(messages)


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


def _mesurer_texte(draw: ImageDraw.ImageDraw, texte: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    """Calcule la largeur et la hauteur d'un texte en s'adaptant à Pillow."""

    try:
        bbox = draw.textbbox((0, 0), texte, font=font)
    except AttributeError:  # Pillow < 8.0 ou primitives limitées
        try:
            bbox = font.getbbox(texte)  # type: ignore[attr-defined]
        except AttributeError:
            try:
                return font.getsize(texte)  # type: ignore[attr-defined]
            except AttributeError:  # pragma: no cover - scénario très improbable
                return len(texte) * 6, 11

    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _annoter_image(image_bytes: bytes, detections: list[dict[str, Any]]) -> Image.Image:
    """Dessine un cadre vert et un label sur les visages détectés."""

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return Image.new("RGB", (200, 200), color="black")

    if not detections:
        return img

    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    for det in detections:
        bbox = det.get("bbox", [])
        if not bbox or len(bbox) != 4:
            continue
        x, y, w, h = [int(float(v)) for v in bbox]
        draw.rectangle([(x, y), (x + w, y + h)], outline="green", width=3)

        emotion = det.get("emotion_predite", "")
        score = float(det.get("score", 0.0))
        texte = f"{emotion} ({score:.2f})"
        text_width, text_height = _mesurer_texte(draw, texte, font)
        text_x = x
        text_y = max(0, y - text_height - 6)
        draw.rectangle(
            [(text_x, text_y), (text_x + text_width + 6, text_y + text_height + 4)],
            fill=(0, 128, 0),
        )
        draw.text((text_x + 3, text_y + 2), texte, fill="white", font=font)

    return img


def ui_emotions_images(df_images: pd.DataFrame | None) -> None:
    """Interface Streamlit pour lancer la détection d'émotions sur les images importées."""
    st.subheader("Détection d'émotions (images synchronisées)")
    st.markdown(recommander_modele_emotions())

    detector, message_modele = charger_modele_emotions()
    st.caption(message_modele)
    st.write(
        "Cette analyse s'appuie exclusivement sur les images importées dans l'onglet « 1. Données »."
        " Chaque fichier sélectionné est transmis au détecteur disponible (FER ou module OpenCV) qui"
        " identifie les visages et attribue une émotion dominante à chacun."
    )

    images_store = st.session_state.get("images_store", []) or []
    if not images_store:
        st.info("Importez d'abord des images (onglet 1. Données) pour lancer l'analyse.")
        return

    if detector is None:
        st.warning(
            "Le modèle de détection n'est pas disponible. Installez `fer` ou assurez-vous que"
            " `opencv-python` est présent puis relancez l'application pour activer cette section."
        )
        return

    noms_images = [it.get("name") for it in images_store if isinstance(it, dict) and it.get("name")]
    if not noms_images:
        st.info("Aucune image valide n'est disponible dans la session.")
        return

    st.info(
        "Toutes les images importées sont analysées automatiquement afin d'assurer une couverture"
        " complète des détections."
    )

    resultats: list[dict[str, Any]] = []
    images_annotes: list[dict[str, Any]] = []
    df_images = df_images if isinstance(df_images, pd.DataFrame) else pd.DataFrame()
    ordre_images = {nom: idx for idx, nom in enumerate(noms_images)}

    with st.spinner("Analyse des émotions en cours..."):
        for nom in noms_images:
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
                images_annotes.append(
                    {
                        "fichier_image": nom,
                        "image": _annoter_image(bytes_img, []),
                        "detections": [],
                    }
                )
                continue

            images_annotes.append(
                {
                    "fichier_image": nom,
                    "image": _annoter_image(bytes_img, detections),
                    "detections": detections,
                }
            )

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
        st.warning("Aucun visage ni émotion détectés sur les images importées.")
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

    st.markdown("#### Streamgraph des émotions")
    df_stream = df_res[df_res["emotion_predite"] != "aucune"].copy()
    if df_stream.empty:
        st.info("Aucune émotion détectée pour générer le streamgraph.")
    else:
        df_stream["ordre_image"] = df_stream["fichier_image"].map(ordre_images).astype(float)
        if df_stream["t_image"].notna().any():
            df_stream["axe"] = df_stream["t_image"].fillna(df_stream["ordre_image"])
            axe_label = "Temps (s)"
        else:
            df_stream["axe"] = df_stream["ordre_image"]
            axe_label = "Ordre des images"

        stream_chart = (
            alt.Chart(df_stream)
            .mark_area()
            .encode(
                x=alt.X("axe:Q", title=axe_label),
                y=alt.Y("score:Q", stack="center", title="Score"),
                color=alt.Color("emotion_predite:N", title="Émotion"),
                tooltip=[
                    alt.Tooltip("fichier_image:N", title="Image"),
                    alt.Tooltip("emotion_predite:N", title="Émotion"),
                    alt.Tooltip("score:Q", title="Score", format=".2f"),
                ],
            )
            .properties(width="container", height=300)
        )
        st.altair_chart(stream_chart, use_container_width=True)

    st.markdown("#### Visualisation des images annotées")
    for entree in images_annotes:
        st.image(entree["image"], caption=entree["fichier_image"], use_column_width=True)
        detections = entree["detections"]
        if detections:
            lignes = []
            for det in detections:
                face_id = det.get("face_id")
                face_txt = f"Visage {face_id}" if face_id is not None else "Visage"
                emotion = det.get("emotion_predite", "")
                score = float(det.get("score", 0.0))
                lignes.append(f"- {face_txt} : **{emotion}** (score {score:.2f})")
            st.markdown("\n".join(lignes))
        else:
            st.markdown("- Aucun visage détecté sur cette image.")

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
