"""Détection d'émotions sur les images synchronisées."""

from __future__ import annotations

import io
import json
import os
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from PIL import Image, ImageDraw, ImageFont, ImageOps

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


_FACE_CASCADE: "cv2.CascadeClassifier | None" = None


def _charger_cascade_visage() -> "cv2.CascadeClassifier | None":
    """Charge et met en cache le détecteur de visages Haar d'OpenCV."""

    global _FACE_CASCADE

    if not _CV2_DISPONIBLE:
        return None

    if _FACE_CASCADE is not None:
        return _FACE_CASCADE

    base_path = getattr(cv2.data, "haarcascades", "")
    face_path = os.path.join(base_path, "haarcascade_frontalface_default.xml")

    cascade = cv2.CascadeClassifier(face_path)
    if cascade.empty():
        return None

    _FACE_CASCADE = cascade
    return _FACE_CASCADE


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
        face_path = os.path.join(base_path, "haarcascade_frontalface_default.xml")
        smile_path = os.path.join(base_path, "haarcascade_smile.xml")

        self._face_cascade = cv2.CascadeClassifier(face_path)
        self._smile_cascade = cv2.CascadeClassifier(smile_path)
        self._orientation: str | None = None

        if self._face_cascade.empty():
            raise RuntimeError(
                "Impossible de charger le classifieur de visages OpenCV (haarcascade_frontalface_default)."
            )
        if self._smile_cascade.empty():
            raise RuntimeError(
                "Impossible de charger le classifieur de sourires OpenCV (haarcascade_smile)."
            )

    def set_orientation(self, orientation: str | None) -> None:
        """Permet d'ajuster dynamiquement les paramètres de détection."""

        self._orientation = orientation

    def detect_emotions(self, image: np.ndarray) -> list[dict[str, Any]]:  # pragma: no cover - dépendance optionnelle
        if image is None or image.size == 0:
            return []

        if image.ndim == 2:
            gray = image
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        gray = cv2.equalizeHist(gray)

        orientation = (self._orientation or "").lower()
        if orientation in {"portrait", "9:16"}:
            scale_factor = 1.08
            min_neighbors = 5
            min_size = (30, 30)
        else:
            scale_factor = 1.15
            min_neighbors = 6
            min_size = (40, 40)

        faces = self._face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
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
                scores = {"happy": 0.9, "neutral": 0.1, "sad": 0.0}
            else:
                scores = {"happy": 0.1, "neutral": 0.7, "sad": 0.2}

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
        "Nous recommandons l'utilisation du modèle **FER** (Facial Emotion Recognition) reposant sur"
        " un réseau de neurones convolutif pré-entraîné sur le jeu de données FER2013. Il fonctionne"
        " sur CPU, nécessite uniquement le paquet `fer` (`pip install fer`) et offre un excellent"
        " compromis précision/performance pour une analyse d'images à 1 fps."
    )


def _nettoyer_bbox(
    bbox: Any, largeur: int, hauteur: int
) -> tuple[int, int, int, int]:
    """Normalise une boîte englobante en coordonnées absolues (x1, y1, x2, y2)."""

    if isinstance(bbox, dict):
        x = bbox.get("x") or bbox.get("left") or 0
        y = bbox.get("y") or bbox.get("top") or 0
        w = bbox.get("w") or bbox.get("width") or 0
        h = bbox.get("h") or bbox.get("height") or 0
        x2 = bbox.get("x2") or bbox.get("right")
        y2 = bbox.get("y2") or bbox.get("bottom")
        if x2 is None:
            x2 = x + w
        if y2 is None:
            y2 = y + h
    elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        x, y, third, fourth = bbox[:4]
        if third > 0 and fourth > 0 and third <= largeur and fourth <= hauteur:
            x2 = x + third
            y2 = y + fourth
        else:
            x2 = third
            y2 = fourth
    else:
        return 0, 0, 0, 0

    try:
        x1_i = int(round(float(x)))
        y1_i = int(round(float(y)))
        x2_i = int(round(float(x2)))
        y2_i = int(round(float(y2)))
    except Exception:
        return 0, 0, 0, 0

    x1_i = max(0, min(largeur, x1_i))
    y1_i = max(0, min(hauteur, y1_i))
    x2_i = max(x1_i, min(largeur, x2_i))
    y2_i = max(y1_i, min(hauteur, y2_i))
    return x1_i, y1_i, x2_i, y2_i


def _ponderer_detection(det: dict[str, Any], taille_image: tuple[int, int]) -> float:
    """Calcule une métrique pour prioriser les visages."""

    largeur, hauteur = taille_image
    x1, y1, x2, y2 = det.get("bbox", (0, 0, 0, 0))
    largeur_face = max(1, x2 - x1)
    hauteur_face = max(1, y2 - y1)
    aire = largeur_face * hauteur_face
    score = float(det.get("score", 0.0))
    centre_x = x1 + largeur_face / 2
    centre_y = y1 + hauteur_face / 2
    centre_image_x = largeur / 2
    centre_image_y = hauteur / 2
    distance_centre = np.hypot(centre_x - centre_image_x, centre_y - centre_image_y)
    distance_normalisee = distance_centre / max(1.0, np.hypot(largeur, hauteur))
    return (score + 0.05) * aire * (1.0 - distance_normalisee)


def _centrer_carre(
    bbox: tuple[int, int, int, int], largeur: int, hauteur: int
) -> tuple[int, int, int, int]:
    """Recentre la boîte sur un carré centré sur le visage détecté."""

    x1, y1, x2, y2 = bbox
    largeur_face = max(1, x2 - x1)
    hauteur_face = max(1, y2 - y1)
    cote = max(largeur_face, hauteur_face)
    centre_x = x1 + largeur_face / 2
    centre_y = y1 + hauteur_face / 2

    demi_cote = cote / 2
    nouveau_x1 = int(round(centre_x - demi_cote))
    nouveau_y1 = int(round(centre_y - demi_cote))
    nouveau_x2 = int(round(centre_x + demi_cote))
    nouveau_y2 = int(round(centre_y + demi_cote))

    nouveau_x1 = max(0, min(largeur - 1, nouveau_x1))
    nouveau_y1 = max(0, min(hauteur - 1, nouveau_y1))
    nouveau_x2 = max(nouveau_x1 + 1, min(largeur, nouveau_x2))
    nouveau_y2 = max(nouveau_y1 + 1, min(hauteur, nouveau_y2))

    return nouveau_x1, nouveau_y1, nouveau_x2, nouveau_y2


def _calculer_iou(b1: tuple[int, int, int, int], b2: tuple[int, int, int, int]) -> float:
    """Calcule l'Intersection over Union entre deux boîtes englobantes."""

    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])

    inter_largeur = max(0, x2 - x1)
    inter_hauteur = max(0, y2 - y1)
    inter = inter_largeur * inter_hauteur

    if inter == 0:
        return 0.0

    aire_b1 = max(0, b1[2] - b1[0]) * max(0, b1[3] - b1[1])
    aire_b2 = max(0, b2[2] - b2[0]) * max(0, b2[3] - b2[1])
    union = aire_b1 + aire_b2 - inter
    if union <= 0:
        return 0.0
    return inter / union


def _detect_faces_cv2(image: np.ndarray, orientation: str | None) -> list[tuple[int, int, int, int]]:
    """Détecte les visages à l'aide d'OpenCV pour fiabiliser les boîtes."""

    cascade = _charger_cascade_visage()
    if cascade is None:
        return []

    if image.ndim == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    gray = cv2.equalizeHist(gray)

    orientation_norm = (orientation or "").lower()
    if orientation_norm in {"portrait", "9:16"}:
        scale_factor = 1.08
        min_neighbors = 4
        min_size = (28, 28)
    else:
        scale_factor = 1.15
        min_neighbors = 5
        min_size = (36, 36)

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
    )

    return [
        (int(x), int(y), int(x + w), int(y + h))
        for (x, y, w, h) in faces
        if w > 0 and h > 0
    ]


def _evaluer_emotions_sur_face(detector: Any, image: np.ndarray) -> dict[str, float]:
    """Déduit les scores d'émotions sur un extrait de visage."""

    if detector is None:
        return {}

    try:
        predictions = detector.detect_emotions(image)
    except Exception:  # pragma: no cover - dépendance optionnelle
        return {}

    if not predictions:
        return {}

    if isinstance(predictions, list) and predictions:
        premier = predictions[0]
        emotions = premier.get("emotions") if isinstance(premier, dict) else None
        return emotions if isinstance(emotions, dict) else {}

    return {}


def _analyser_image(detector: Any, image_bytes: bytes, orientation: str | None = None) -> list[dict[str, Any]]:
    """Applique le détecteur sur une image et renvoie une liste de résultats par visage."""
    if detector is None:
        return []

    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = ImageOps.exif_transpose(img).convert("RGB")
    except Exception:
        return []

    largeur, hauteur = img.size
    arr = np.array(img)

    if hasattr(detector, "set_orientation"):
        try:
            detector.set_orientation(orientation)
        except Exception:
            pass

    try:
        detections = detector.detect_emotions(arr)
    except Exception:  # pragma: no cover - dépendance optionnelle
        detections = []

    faces_cv2 = _detect_faces_cv2(arr, orientation)

    sorties: list[dict[str, Any]] = []
    for resultat in detections or []:
        emotions = resultat.get("emotions", {})
        if not emotions:
            continue
        emotion_predite, score = max(emotions.items(), key=lambda item: item[1])
        x1, y1, x2, y2 = _nettoyer_bbox(resultat.get("box", []), largeur, hauteur)
        if x2 <= x1 or y2 <= y1:
            continue

        meilleure_face_idx = None
        meilleure_iou = 0.0
        for idx, face in enumerate(faces_cv2):
            iou = _calculer_iou((x1, y1, x2, y2), face)
            if iou > meilleure_iou:
                meilleure_iou = iou
                meilleure_face_idx = idx

        if meilleure_face_idx is not None and meilleure_iou >= 0.1:
            x1, y1, x2, y2 = faces_cv2.pop(meilleure_face_idx)

        x1, y1, x2, y2 = _centrer_carre((x1, y1, x2, y2), largeur, hauteur)
        sorties.append(
            {
                "predicted_emotion": str(emotion_predite),
                "score": float(score),
                "emotion_scores": emotions,
                "bbox": (x1, y1, x2, y2),
            }
        )

    if not sorties and faces_cv2:
        # Aucune émotion n'a été retournée mais des visages ont été repérés via OpenCV.
        for idx, face in enumerate(faces_cv2):
            x1, y1, x2, y2 = _centrer_carre(face, largeur, hauteur)
            visage = arr[y1:y2, x1:x2]
            emotions = _evaluer_emotions_sur_face(detector, visage) if visage.size else {}
            if emotions:
                emotion_predite, score = max(emotions.items(), key=lambda item: item[1])
            else:
                emotion_predite, score = "none", 0.0
            sorties.append(
                {
                    "predicted_emotion": str(emotion_predite),
                    "score": float(score),
                    "emotion_scores": emotions or {},
                    "bbox": (x1, y1, x2, y2),
                }
            )

    sorties.sort(key=lambda det: _ponderer_detection(det, (largeur, hauteur)), reverse=True)
    for idx, det in enumerate(sorties):
        det["face_id"] = idx
        det["is_primary_face"] = idx == 0

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
        img = Image.open(io.BytesIO(image_bytes))
        img = ImageOps.exif_transpose(img).convert("RGB")
    except Exception:
        return Image.new("RGB", (200, 200), color="black")

    if not detections:
        return img

    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    for det in detections:
        bbox = det.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = bbox
        draw.rectangle([(x1, y1), (x2, y2)], outline="green", width=3)

        emotion = det.get("predicted_emotion", "")
        score = float(det.get("score", 0.0))
        prefix = "Visage principal" if det.get("is_primary_face") else "Visage"
        emotion_display = emotion or "unknown"
        texte = f"{prefix} : {emotion_display} ({score:.2f})"
        text_width, text_height = _mesurer_texte(draw, texte, font)
        text_x = x1
        text_y = max(0, y1 - text_height - 6)
        draw.rectangle(
            [(text_x, text_y), (text_x + text_width + 6, text_y + text_height + 4)],
            fill=(0, 128, 0),
        )
        draw.text((text_x + 3, text_y + 2), texte, fill="white", font=font)

    return img


def ui_emotions_images(df_images: pd.DataFrame | None, orientation_images: str | None = None) -> None:
    """Streamlit interface that runs emotion analysis on imported still images."""

    st.subheader("Analyse des émotions (images synchronisées)")
    st.markdown(recommander_modele_emotions())

    detector, message_modele = charger_modele_emotions()
    st.caption(message_modele)
    st.write(
        "L'analyse repose exclusivement sur les images importées via l'onglet « 1. Données »."
        " Chaque fichier sélectionné est soumis au détecteur disponible (FER ou solution de"
        " repli OpenCV) afin d'identifier les visages et d'attribuer l'émotion dominante"
        " parmi : angry, disgust, fear, happy, neutral, sad, surprise."
    )

    images_store = st.session_state.get("images_store", []) or []
    if not images_store:
        st.info("Importez d'abord des images (onglet 1. Données) pour lancer l'analyse.")
        return

    if detector is None:
        st.warning(
            "Aucun détecteur n'est disponible. Installez `fer` ou vérifiez la présence de"
            " `opencv-python`, puis redémarrez l'application pour activer cette section."
        )
        return

    noms_images = [it.get("name") for it in images_store if isinstance(it, dict) and it.get("name")]
    if not noms_images:
        st.info("Aucune image valide n'a été trouvée dans la session en cours.")
        return

    st.info(
        "Toutes les images importées sont analysées afin d'assurer une couverture complète"
        " des détections."
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

            detections = _analyser_image(detector, bytes_img, orientation=orientation_images)
            if not detections:
                resultats.append(
                    {
                        "fichier_image": nom,
                        "t_image": _recuperer_timestamp(df_images, nom),
                        "face_id": np.nan,
                        "emotion_predite": "none",
                        "predicted_emotion": "none",
                        "score": 0.0,
                        "scores_emotions": {},
                        "emotion_scores": {},
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
                    "emotion_predite": det.get("predicted_emotion", ""),
                    "predicted_emotion": det.get("predicted_emotion", ""),
                    "score": float(det.get("score", 0.0)),
                    "scores_emotions": det.get("emotion_scores", {}),
                    "emotion_scores": det.get("emotion_scores", {}),
                    "bbox": list(det.get("bbox", [])),
                    "is_primary_face": bool(det.get("is_primary_face", False)),
                }
                resultats.append(ligne)

    if not resultats:
        st.warning("Aucun visage ou émotion n'a été détecté sur les images importées.")
        return

    df_res = pd.DataFrame(resultats)
    df_res["score"] = df_res["score"].astype(float)
    if "emotion_scores" not in df_res.columns:
        df_res["emotion_scores"] = [{}] * len(df_res)
    df_res["scores_emotions_json"] = df_res["emotion_scores"].apply(
        lambda d: json.dumps(d, ensure_ascii=False) if isinstance(d, dict) else "{}"
    )

    st.session_state["df_emotions"] = df_res.copy()

    st.markdown("#### Résultats détaillés")
    colonnes_affichage = {
        "fichier_image": "Fichier image",
        "t_image": "Timestamp (s)",
        "face_id": "Indice du visage",
        "predicted_emotion": "Émotion prédite",
        "score": "Confiance",
        "bbox": "Boîte englobante (x1, y1, x2, y2)",
        "scores_emotions_json": "Scores complets",
    }
    st.dataframe(
        df_res[list(colonnes_affichage.keys())].rename(columns=colonnes_affichage),
        use_container_width=True,
    )

    st.markdown("#### Répartition des émotions détectées")
    distrib = df_res[df_res["predicted_emotion"] != "none"]["predicted_emotion"].value_counts().reset_index()
    distrib.columns = ["emotion", "occurrences"]
    if distrib.empty:
        st.info("Aucune émotion dominante n'a été détectée.")
    else:
        st.bar_chart(data=distrib.set_index("emotion"))

    st.markdown("#### Évolution temporelle des émotions")
    df_stream = df_res[df_res["predicted_emotion"] != "none"].copy()
    if df_stream.empty:
        st.info("Aucune émotion détectée n'est disponible pour construire le graphique.")
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
                y=alt.Y("score:Q", stack="center", title="Confiance"),
                color=alt.Color("predicted_emotion:N", title="Emotion"),
                tooltip=[
                    alt.Tooltip("fichier_image:N", title="Image"),
                    alt.Tooltip("predicted_emotion:N", title="Emotion"),
                    alt.Tooltip("score:Q", title="Confiance", format=".2f"),
                ],
            )
            .properties(width="container", height=300)
        )
        st.altair_chart(stream_chart, use_container_width=True)

    st.markdown("#### Vignettes annotées")
    if images_annotes:
        taille_vignettes = st.slider(
            "Largeur des vignettes (px)", min_value=200, max_value=800, value=320, step=20
        )
        n_cols = 3
        rows = [images_annotes[i : i + n_cols] for i in range(0, len(images_annotes), n_cols)]
        for row in rows:
            cols = st.columns(len(row))
            for col, entree in zip(cols, row):
                col.image(entree["image"], caption=entree["fichier_image"], width=taille_vignettes)
                detections = entree["detections"]
                if detections:
                    lignes = []
                    for det in detections:
                        face_id = det.get("face_id")
                        face_txt = (
                            "Visage principal"
                            if det.get("is_primary_face")
                            else f"Visage {face_id}" if face_id is not None else "Visage"
                        )
                        emotion = det.get("predicted_emotion", "") or "unknown"
                        score = float(det.get("score", 0.0))
                        lignes.append(f"- {face_txt} : **{emotion}** (confiance {score:.2f})")
                    col.markdown("\n".join(lignes))
                else:
                    col.markdown("- Aucun visage détecté sur cette image.")

                with col.expander("Afficher en grand"):
                    st.image(entree["image"], caption=entree["fichier_image"], use_container_width=True)
    else:
        st.info("Aucune vignette annotée n'est disponible pour le moment.")

    st.markdown("#### Export")
    export_df = df_res.drop(columns=["emotion_scores"], errors="ignore")
    csv = export_df.to_csv(index=False).encode("utf-8")
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
