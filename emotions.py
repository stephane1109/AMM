# emotions.py
# Détection d'émotions sur images fixes synchronisées – sortie 6 labels FER (sans "neutral")

from __future__ import annotations
import io, os, json
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from PIL import Image, ImageOps, ImageDraw, ImageFont

# ===== Dépendances optionnelles
_FER_ERR = ""
_CV2_ERR = ""
_XLSX_OK = False
try:
    import xlsxwriter  # type: ignore
    _XLSX_OK = True
except Exception:
    pass

try:
    import cv2  # type: ignore
    _CV2_OK = True
except Exception as exc:
    _CV2_OK = False
    _CV2_ERR = str(exc)

try:
    from fer import FER  # type: ignore
    _FER_OK = True
except Exception as exc:
    _FER_OK = False
    _FER_ERR = str(exc)

# ===== Constantes
EMO6 = ["angry", "disgust", "fear", "happy", "sad", "surprise"]  # 6 labels demandés

# ===== Outils

def _img_from_bytes(b: bytes) -> Image.Image | None:
    try:
        im = Image.open(io.BytesIO(b))
        return ImageOps.exif_transpose(im).convert("RGB")
    except Exception:
        return None

def _to_numpy(im: Image.Image) -> np.ndarray:
    return np.array(im)

def _ensure_6(emotions: Dict[str, float]) -> Dict[str, float]:
    """
    Convertit un dictionnaire de scores FER en 6 labels (sans 'neutral').
    - Supprime neutral s'il existe et renormalise
    - Ajoute les labels manquants à 0.0
    """
    if not isinstance(emotions, dict) or not emotions:
        return {k: 0.0 for k in EMO6}
    # supprime neutral
    emo = {k: float(v) for k, v in emotions.items() if k in EMO6}
    s = sum(emo.values())
    if s > 0:
        emo = {k: v / s for k, v in emo.items()}
    else:
        emo = {k: 0.0 for k in EMO6}
    # complète clés manquantes
    for k in EMO6:
        emo.setdefault(k, 0.0)
    return emo

def _top_emo6(emotions_6: Dict[str, float]) -> Tuple[str, float]:
    if not emotions_6:
        return "none", 0.0
    k = max(emotions_6.items(), key=lambda x: x[1])
    return k[0], float(k[1])

# ===== Modèles

@st.cache_resource(show_spinner=False)
def _load_fer() -> Tuple[Any | None, str]:
    if not _FER_OK:
        msg = "Le paquet `fer` n’est pas disponible. Installez-le avec `pip install fer`."
        if _FER_ERR:
            msg += f" (détail: {_FER_ERR})"
        return None, msg
    try:
        # MTCNN améliore la localisation des visages
        det = FER(mtcnn=True)
        return det, "FER initialisé (mtcnn=True)."
    except Exception as e:
        return None, f"Échec init FER: {e}"

@st.cache_resource(show_spinner=False)
def _load_cv2_face() -> Tuple[Any | None, str]:
    if not _CV2_OK:
        return None, "OpenCV indisponible."
    try:
        cascade = cv2.CascadeClassifier(
            os.path.join(getattr(cv2.data, "haarcascades", ""), "haarcascade_frontalface_default.xml")
        )
        if cascade.empty():
            return None, "Cascade de visage OpenCV introuvable."
        return cascade, "Cascade OpenCV chargée."
    except Exception as e:
        return None, f"Échec cascade OpenCV: {e}"

# ===== Pipelines

def _fer_detect_full(fer_det: Any, arr: np.ndarray) -> List[Dict[str, Any]]:
    """
    Utilise FER.detect_emotions directement sur l'image complète.
    Retour: liste de dicts avec 'box' ([x,y,w,h]) et 'emotions' (scores 7 ou 8 labels).
    """
    try:
        preds = fer_det.detect_emotions(arr)  # type: ignore[attr-defined]
        return preds or []
    except Exception:
        return []

def _cv2_faces(cascade: Any, arr: np.ndarray) -> List[Tuple[int,int,int,int]]:
    try:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(36, 36))
        out = []
        for (x, y, w, h) in faces:
            if w > 0 and h > 0:
                out.append((int(x), int(y), int(x+w), int(y+h)))
        return out
    except Exception:
        return []

def _crop(arr: np.ndarray, x1:int, y1:int, x2:int, y2:int) -> np.ndarray:
    x1 = max(0, min(arr.shape[1]-1, x1))
    y1 = max(0, min(arr.shape[0]-1, y1))
    x2 = max(x1+1, min(arr.shape[1], x2))
    y2 = max(y1+1, min(arr.shape[0], y2))
    return arr[y1:y2, x1:x2]

def _bbox_from_box(box: Any, w: int, h: int) -> Tuple[int,int,int,int]:
    """
    FER renvoie box=[x,y,w,h] ; on convertit en (x1,y1,x2,y2) borné à l’image.
    """
    try:
        x, y, bw, bh = [int(round(float(v))) for v in box[:4]]
    except Exception:
        return 0,0,0,0
    x1, y1 = max(0,x), max(0,y)
    x2, y2 = min(w, x+bw), min(h, y+bh)
    if x2 <= x1 or y2 <= y1:
        return 0,0,0,0
    return x1,y1,x2,y2

def _analyze_one_image(fer_det: Any | None, cv2_cascade: Any | None, b: bytes) -> List[Dict[str, Any]]:
    """
    Sortie: liste de {bbox:(x1,y1,x2,y2), emotions_6:dict, label:str, score:float}
    - 1) On tente FER plein cadre
    - 2) Si zéro résultat mais cascade dispo => on croppe chaque visage et on reclasse avec FER
    """
    im = _img_from_bytes(b)
    if im is None:
        return []
    arr = _to_numpy(im)
    H, W = arr.shape[:2]
    results: List[Dict[str, Any]] = []

    if fer_det is not None:
        # 1) FER direct
        preds = _fer_detect_full(fer_det, arr)
        for p in preds:
            box = p.get("box", [])
            emos = p.get("emotions", {}) or {}
            x1,y1,x2,y2 = _bbox_from_box(box, W, H)
            if x2<=x1 or y2<=y1:
                continue
            emo6 = _ensure_6(emos)
            lab, sc = _top_emo6(emo6)
            results.append({"bbox":(x1,y1,x2,y2), "emotions_6":emo6, "label":lab, "score":sc})

    # 2) Fallback: visages OpenCV + reclassement par FER (si dispo)
    if (not results) and (cv2_cascade is not None) and (fer_det is not None):
        faces = _cv2_faces(cv2_cascade, arr)
        for (x1,y1,x2,y2) in faces:
            face = _crop(arr, x1,y1,x2,y2)
            try:
                sub = fer_det.detect_emotions(face) or []  # type: ignore[attr-defined]
            except Exception:
                sub = []
            if sub:
                emos = sub[0].get("emotions", {}) or {}
                emo6 = _ensure_6(emos)
                lab, sc = _top_emo6(emo6)
            else:
                emo6 = {k:0.0 for k in EMO6}
                lab, sc = "none", 0.0
            results.append({"bbox":(x1,y1,x2,y2), "emotions_6":emo6, "label":lab, "score":sc})

    # Classement: visage le plus crédible en premier (aire * score)
    if results:
        def _rank(r):
            x1,y1,x2,y2 = r["bbox"]
            return (x2-x1)*(y2-y1) * (0.25 + float(r.get("score",0.0)))
        results.sort(key=_rank, reverse=True)
        for i, r in enumerate(results):
            r["face_id"] = i
            r["is_primary_face"] = (i==0)

    return results

# ===== Rendu

def _draw_boxes(im: Image.Image, detections: List[Dict[str, Any]]) -> Image.Image:
    if not detections:
        return im
    im = im.copy()
    drw = ImageDraw.Draw(im)
    font = ImageFont.load_default()
    for d in detections:
        x1,y1,x2,y2 = d["bbox"]
        drw.rectangle([(x1,y1),(x2,y2)], outline="green", width=3)
        lab = d["label"]; sc = float(d.get("score",0.0))
        tag = f"{'Visage principal' if d.get('is_primary_face') else 'Visage'}: {lab} ({sc:.2f})"
        tw, th = drw.textlength(tag, font=font), 11
        drw.rectangle([(x1, max(0,y1-th-6)), (x1+int(tw)+6, max(0,y1-th-6)+th+4)], fill=(0,128,0))
        drw.text((x1+3, max(0,y1-th-6)+2), tag, fill="white", font=font)
    return im

def _timestamp_from_df(df_images: pd.DataFrame | None, name: str) -> float | np.nan:
    if df_images is None or df_images.empty:
        return np.nan
    row = df_images[df_images["fichier_image"] == name]
    if row.empty:
        return np.nan
    v = row.iloc[0].get("t_image", np.nan)
    try:
        return float(v) if pd.notna(v) else np.nan
    except Exception:
        return np.nan

def _export_excel(df: pd.DataFrame, names: List[str]) -> Tuple[str | None, Dict[str, List[float]] | None]:
    if not _XLSX_OK:
        return None, None
    try:
        outdir = os.path.join(os.getcwd(), "emotion")
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, "emotions_scores.xlsx")
        with xlsxwriter.Workbook(path) as wb:  # type: ignore
            ws = wb.add_worksheet()
            ws.write(0,0,"Image")
            for j, emo in enumerate(EMO6, start=1):
                ws.write(0,j,emo)
            emo_data = {k:[] for k in EMO6}
            r = 1
            for name in names:
                sdf = df[df["fichier_image"] == name]
                mean = {k:0.0 for k in EMO6}
                n=0
                for _, row in sdf.iterrows():
                    scs = row.get("emotions_6", {})
                    if isinstance(scs, dict) and scs:
                        n += 1
                        for k in EMO6:
                            mean[k] += float(scs.get(k,0.0))
                if n>0:
                    for k in EMO6:
                        mean[k] /= n
                ws.write(r,0,name)
                for j, emo in enumerate(EMO6, start=1):
                    ws.write(r,j,float(mean.get(emo,0.0)))
                    emo_data[emo].append(float(mean.get(emo,0.0)))
                r += 1
        return path, emo_data
    except Exception:
        return None, None

# ===== UI principale

def ui_emotions_images(df_images: pd.DataFrame | None, orientation_images: str | None = None) -> None:
    st.subheader("Analyse des émotions (images, 6 labels FER)")
    # Charger modèles
    fer_det, msg_fer = _load_fer()
    st.caption(msg_fer)
    cv_cascade, msg_cv = _load_cv2_face()
    if cv_cascade is None:
        st.caption(msg_cv)

    images_store = st.session_state.get("images_store", []) or []
    names = [it["name"] for it in images_store if isinstance(it, dict) and "name" in it]
    if not names:
        st.info("Importe d’abord des images dans l’onglet « 1. Données ».")
        return

    results: List[Dict[str, Any]] = []
    annotated: List[Dict[str, Any]] = []

    with st.spinner("Détection en cours…"):
        for nm in names:
            b = None
            for it in images_store:
                if isinstance(it, dict) and it.get("name") == nm:
                    b = it.get("bytes")
                    break
            if b is None:
                continue

            dets = _analyze_one_image(fer_det, cv_cascade, b)

            # Annoté
            im = _img_from_bytes(b)
            if im is None:
                continue
            annotated.append({"fichier_image": nm, "image": _draw_boxes(im, dets), "dets": dets})

            # Lignes de sortie
            timg = _timestamp_from_df(df_images, nm)
            for d in dets if dets else [{"bbox":(), "emotions_6":{k:0.0 for k in EMO6}, "label":"none", "score":0.0, "face_id":np.nan, "is_primary_face":False}]:
                row = {
                    "fichier_image": nm,
                    "t_image": timg,
                    "face_id": d.get("face_id", np.nan),
                    "is_primary_face": bool(d.get("is_primary_face", False)),
                    "predicted_emotion": str(d.get("label","none")),
                    "score": float(d.get("score",0.0)),
                    "bbox": list(d.get("bbox", ())),
                    "emotions_6": d.get("emotions_6", {k:0.0 for k in EMO6}),
                }
                results.append(row)

    if not results:
        st.warning("Aucun visage/émotion détecté.")
        return

    df = pd.DataFrame(results)
    # Colonnes 6 labels à plat (pour stats)
    for k in EMO6:
        df[f"emo_{k}"] = df["emotions_6"].apply(lambda d: float(d.get(k,0.0)) if isinstance(d, dict) else 0.0)

    st.session_state["df_emotions"] = df.copy()

    st.markdown("#### Résultats (par visage)")
    show_cols = ["fichier_image","t_image","face_id","is_primary_face","predicted_emotion","score","bbox"] + [f"emo_{k}" for k in EMO6]
    st.dataframe(df[show_cols], use_container_width=True)

    # Distribution
    st.markdown("#### Répartition des émotions (6 labels)")
    distrib = df[df["predicted_emotion"]!="none"]["predicted_emotion"].value_counts().reindex(EMO6, fill_value=0)
    st.bar_chart(distrib)

    # Stream temporel (si timestamps)
    st.markdown("#### Évolution temporelle")
    dft = df[df["predicted_emotion"]!="none"].copy()
    if not dft.empty:
        if dft["t_image"].notna().any():
            dft["axe"] = dft["t_image"]
            xlabel = "Temps (s)"
        else:
            order = {n:i for i,n in enumerate(names)}
            dft["axe"] = dft["fichier_image"].map(order).astype(float)
            xlabel = "Ordre d’image"
        stream = (
            alt.Chart(dft)
            .mark_area()
            .encode(
                x=alt.X("axe:Q", title=xlabel),
                y=alt.Y("score:Q", stack="center", title="Confiance"),
                color=alt.Color("predicted_emotion:N", title="Émotion"),
                tooltip=["fichier_image","predicted_emotion","score"]
            ).properties(height=280)
        )
        st.altair_chart(stream, use_container_width=True)
    else:
        st.caption("Aucun point temporel exploitable.")

    # Vignettes annotées
    st.markdown("#### Vignettes annotées")
    size = st.slider("Largeur des vignettes (px)", 200, 800, 320, 20)
    ncols = 3
    rows = [annotated[i:i+ncols] for i in range(0, len(annotated), ncols)]
    for r in rows:
        cols = st.columns(len(r))
        for c, item in zip(cols, r):
            c.image(item["image"], caption=item["fichier_image"], width=size)
            dets = item["dets"]
            if dets:
                c.markdown(
                    "\n".join(
                        f"- {'Visage principal' if d.get('is_primary_face') else f\"Visage {d.get('face_id')}\"} : **{d['label']}** ({d['score']:.2f})"
                        for d in dets
                    )
                )
            else:
                c.markdown("- Aucun visage détecté.")

    # Export
    st.markdown("#### Export")
    csv = df.drop(columns=["emotions_6"], errors="ignore").to_csv(index=False).encode("utf-8")
    st.download_button("Télécharger (CSV)", data=csv, file_name="emotions_images.csv", mime="text/csv")

    if _XLSX_OK:
        path, emo_data = _export_excel(df, names)
        if path and os.path.exists(path):
            with open(path, "rb") as f:
                st.download_button(
                    "Télécharger scores (Excel)",
                    data=f.read(),
                    file_name="emotions_scores.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
        else:
            st.caption("xlsxwriter indisponible ou export impossible.")
