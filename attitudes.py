# attitudes.py
# Extraction d'attitudes non verbales à partir d'IMAGES (pas de vidéo)
# Si MediaPipe est disponible, on calcule quelques indicateurs simples :
# - ouverture de la bouche (bouche_ouverture)
# - ouverture des épaules (ouverture_epaules) via repères Pose
# - orientation approximative de la tête (orientation_tete) via repères du visage
# - nombre de mains visibles (nb_mains)
# Les résultats sont renvoyés par image + une agrégation temporelle (seconde).

import io
import math
import os
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import altair as alt

# Forcer MediaPipe à fonctionner en mode CPU : certaines plateformes (macOS
# sans contexte graphique, environnements headless, etc.) ne peuvent pas
# initialiser l'OpenGL requis par les graphes GPU. L'environnement
# MEDIAPIPE_DISABLE_GPU désactive la dépendance au service GPU.
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")

# chargement optionnel de mediapipe
try:
    import mediapipe as mp
    _mp_ok = True
except Exception:
    _mp_ok = False


# =========================
# utilitaires internes
# =========================

def _pil_from_bytes(b: bytes):
    """ouvrir une image PIL depuis des octets."""
    try:
        return Image.open(io.BytesIO(b)).convert("RGB")
    except Exception:
        return None

def _to_np(img_pil):
    """convertir PIL -> np.array float32 [0..1] en RGB."""
    arr = np.asarray(img_pil)
    if arr.dtype != np.uint8:
        arr = (255 * (arr / np.max(arr))).astype(np.uint8)
    return arr

def _dist(p1, p2):
    """distance euclidienne 2D."""
    return float(math.hypot(p1[0] - p2[0], p1[1] - p2[1]))

def _safe_get(ls, idx):
    """accès sécurisé dans une liste de landmarks."""
    if ls is None or idx is None:
        return None
    if idx < 0 or idx >= len(ls):
        return None
    return ls[idx]

def _norm_xy(lm, w, h):
    """convertir un landmark normalisé mediapipe (x,y) en pixels."""
    return (float(lm.x * w), float(lm.y * h))


# =========================
# calcul indicateurs (MediaPipe si dispo)
# =========================

def _analyser_image_mediapipe(arr_rgb):
    """
    calculer des indicateurs non verbaux simples à partir d'une image numpy RGB.
    retourne un dict avec:
      - bouche_ouverture (ratio)
      - orientation_tete (degrés approximatifs, signe = gauche/droite)
      - ouverture_epaules (distance normalisée)
      - nb_mains (0,1,2)
    si aucun visage/corps/mains détecté: NaN/0.
    """
    if not _mp_ok:
        return {"bouche_ouverture": np.nan,
                "orientation_tete": np.nan,
                "ouverture_epaules": np.nan,
                "nb_mains": 0}

    h, w, _ = arr_rgb.shape

    # face mesh
    fm = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        refine_landmarks=True,
        max_num_faces=1,
        min_detection_confidence=0.5
    )
    # pose
    pose = mp.solutions.pose.Pose(static_image_mode=True)
    # hands
    hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2)

    bouche_ouverture = np.nan
    orientation_tete = np.nan
    ouverture_epaules = np.nan
    nb_mains = 0

    # exécution
    res_face = fm.process(arr_rgb)
    res_pose = pose.process(arr_rgb)
    res_hands = hands.process(arr_rgb)

    # mains
    if res_hands and res_hands.multi_hand_landmarks:
        nb_mains = len(res_hands.multi_hand_landmarks)
    else:
        nb_mains = 0

    # visage
    if res_face and res_face.multi_face_landmarks:
        fl = res_face.multi_face_landmarks[0].landmark

        # indices utiles (FaceMesh) :
        # lèvres sup/inf approximatives : 13 (upper lip), 14 (lower lip)
        # coins des yeux : 33 (œil droit extérieur), 263 (œil gauche extérieur)
        idx_upper = 13
        idx_lower = 14
        idx_eye_r = 33
        idx_eye_l = 263

        p_up = _safe_get(fl, idx_upper)
        p_lo = _safe_get(fl, idx_lower)
        p_er = _safe_get(fl, idx_eye_r)
        p_el = _safe_get(fl, idx_eye_l)

        if p_up and p_lo and p_er and p_el:
            p_up = _norm_xy(p_up, w, h)
            p_lo = _norm_xy(p_lo, w, h)
            p_er = _norm_xy(p_er, w, h)
            p_el = _norm_xy(p_el, w, h)

            bouche = _dist(p_up, p_lo)
            inter_oc = _dist(p_er, p_el)
            if inter_oc > 1e-6:
                bouche_ouverture = bouche / inter_oc

            # orientation approximative : angle de la droite (œil gauche -> œil droit)
            dx = (p_er[0] - p_el[0])
            dy = (p_er[1] - p_el[1])
            angle = math.degrees(math.atan2(dy, dx))  # ~0 si horizontal
            orientation_tete = float(angle)

    # épaules (Pose)
    if res_pose and res_pose.pose_landmarks:
        pl = res_pose.pose_landmarks.landmark
        # indices épaules droite/gauche mediapipe pose: 11 (gauche), 12 (droite)
        p_ls = _safe_get(pl, 11)
        p_rs = _safe_get(pl, 12)
        if p_ls and p_rs:
            p_ls = _norm_xy(p_ls, w, h)
            p_rs = _norm_xy(p_rs, w, h)
            # normaliser par la largeur de l'image
            ouverture_epaules = _dist(p_ls, p_rs) / max(1.0, float(w))

    # libération
    fm.close()
    pose.close()
    hands.close()

    return {"bouche_ouverture": bouche_ouverture,
            "orientation_tete": orientation_tete,
            "ouverture_epaules": ouverture_epaules,
            "nb_mains": int(nb_mains)}


# =========================
# API demandée par main.py
# =========================

def calculer_attitudes_depuis_images(
    images_store,
    activer_openface: bool = True,
    activer_mediapipe: bool = True,
    activer_openpose: bool = False,
    chemin_openface: str = "FeatureExtraction",
    binaire_openpose: str = "openpose"
):
    """
    calculer les attitudes à partir des images déjà chargées dans st.session_state["images_store"]
    images_store: liste de dicts {"name": str, "bytes": bytes}
    renvoie (df_nv_images, df_nv_agrege)

    Remarque :
    - OpenFace et OpenPose ne sont pas exécutés ici (binaire externe requis).
      On expose malgré tout des colonnes 'au_*' et 'openpose_ok' à NaN/False pour compatibilité.
    - Si MediaPipe n'est pas disponible ou désactivé, des NaN seront renvoyés et un avertissement affiché.
    - Les timestamps sont récupérés depuis st.session_state["df_images"] (colonne t_image).
    """

    if images_store is None or len(images_store) == 0:
        return pd.DataFrame(columns=[
            "fichier_image", "t_image",
            "bouche_ouverture", "orientation_tete", "ouverture_epaules", "nb_mains",
            "au_01", "au_02", "au_04", "au_06", "au_12", "openpose_ok"
        ]), pd.DataFrame()

    df_images_meta = st.session_state.get("df_images")
    if df_images_meta is None or df_images_meta.empty:
        # on crée une meta minimale sans timestamp
        df_images_meta = pd.DataFrame([{"fichier_image": it["name"], "t_image": np.nan} for it in images_store])

    # index rapide: nom -> t_image
    t_by_name = {row["fichier_image"]: row["t_image"] for _, row in df_images_meta.iterrows()}

    lignes = []
    for it in images_store:
        name = it["name"]
        b = it["bytes"]

        img = _pil_from_bytes(b)
        if img is None:
            # image illisible
            lignes.append({
                "fichier_image": name,
                "t_image": float(t_by_name.get(name, np.nan)),
                "bouche_ouverture": np.nan,
                "orientation_tete": np.nan,
                "ouverture_epaules": np.nan,
                "nb_mains": 0,
                # placeholders OpenFace/OpenPose
                "au_01": np.nan, "au_02": np.nan, "au_04": np.nan, "au_06": np.nan, "au_12": np.nan,
                "openpose_ok": False
            })
            continue

        arr = _to_np(img)

        # MediaPipe si demandé et dispo
        if activer_mediapipe and _mp_ok:
            feats = _analyser_image_mediapipe(arr)
        else:
            feats = {"bouche_ouverture": np.nan, "orientation_tete": np.nan,
                     "ouverture_epaules": np.nan, "nb_mains": 0}

        # OpenFace / OpenPose non exécutés ici (binaire externe requis)
        au_01 = np.nan
        au_02 = np.nan
        au_04 = np.nan
        au_06 = np.nan
        au_12 = np.nan
        openpose_ok = False

        lignes.append({
            "fichier_image": name,
            "t_image": float(t_by_name.get(name, np.nan)),
            "bouche_ouverture": feats["bouche_ouverture"],
            "orientation_tete": feats["orientation_tete"],
            "ouverture_epaules": feats["ouverture_epaules"],
            "nb_mains": feats["nb_mains"],
            "au_01": au_01, "au_02": au_02, "au_04": au_04, "au_06": au_06, "au_12": au_12,
            "openpose_ok": openpose_ok
        })

    df_nv_images = pd.DataFrame(lignes)

    # agrégation simple par seconde (floor)
    df_agg = df_nv_images.copy()
    if "t_image" in df_agg.columns:
        df_agg["t_sec"] = np.floor(df_agg["t_image"].astype(float)).astype("Int64")
    else:
        df_agg["t_sec"] = pd.NA

    agg_cols_mean = ["bouche_ouverture", "orientation_tete", "ouverture_epaules"]
    agg_cols_sum = ["nb_mains"]

    df_mean = df_agg.groupby("t_sec", dropna=True)[agg_cols_mean].mean().reset_index()
    df_sum  = df_agg.groupby("t_sec", dropna=True)[agg_cols_sum].sum().reset_index()
    df_nv_agrege = pd.merge(df_mean, df_sum, on="t_sec", how="outer").sort_values("t_sec")

    if not _mp_ok and activer_mediapipe:
        st.warning("MediaPipe n’est pas installé. Les indicateurs non verbaux issus de MediaPipe sont renvoyés en NaN.")

    return df_nv_images, df_nv_agrege


def ui_attitudes_images(df_nv_images: pd.DataFrame, df_nv_agrege: pd.DataFrame):
    """
    interface Streamlit pour visualiser les attitudes calculées à partir des images.
    Affiche tableaux + graphiques Altair temporels (par seconde).
    """
    if df_nv_images is None or df_nv_images.empty:
        st.info("Aucun résultat d’attitudes à afficher pour les images.")
        return

    st.markdown("**Résultats par image**")
    st.dataframe(df_nv_images)

    if df_nv_agrege is not None and not df_nv_agrege.empty:
        st.markdown("**Agrégation temporelle (par seconde)**")
        st.dataframe(df_nv_agrege)

        # courbe bouche_ouverture
        if "bouche_ouverture" in df_nv_agrege.columns:
            c1 = alt.Chart(df_nv_agrege).mark_line(point=True).encode(
                x=alt.X("t_sec:Q", title="Temps (s)"),
                y=alt.Y("bouche_ouverture:Q", title="Ouverture de la bouche (rat.)")
            ).properties(height=220, title="Bouche – ouverture")
            st.altair_chart(c1, use_container_width=True)

        # courbe orientation_tete
        if "orientation_tete" in df_nv_agrege.columns:
            c2 = alt.Chart(df_nv_agrege).mark_line(point=True).encode(
                x=alt.X("t_sec:Q", title="Temps (s)"),
                y=alt.Y("orientation_tete:Q", title="Orientation tête (°)")
            ).properties(height=220, title="Tête – orientation (approx.)")
            st.altair_chart(c2, use_container_width=True)

        # courbe ouverture_epaules
        if "ouverture_epaules" in df_nv_agrege.columns:
            c3 = alt.Chart(df_nv_agrege).mark_line(point=True).encode(
                x=alt.X("t_sec:Q", title="Temps (s)"),
                y=alt.Y("ouverture_epaules:Q", title="Ouverture épaules (norm.)")
            ).properties(height=220, title="Posture – ouverture des épaules")
            st.altair_chart(c3, use_container_width=True)

        # barres sur nb_mains
        if "nb_mains" in df_nv_agrege.columns:
            c4 = alt.Chart(df_nv_agrege).mark_bar().encode(
                x=alt.X("t_sec:Q", title="Temps (s)"),
                y=alt.Y("nb_mains:Q", title="Mains visibles (somme)")
            ).properties(height=220, title="Mains visibles")
            st.altair_chart(c4, use_container_width=True)

    else:
        st.caption("Aucune agrégation disponible.")
