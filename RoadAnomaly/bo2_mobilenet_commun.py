"""
Code commun de ma partie du BO2 (classification des régions par MobileNetV2).

Utilisé par BO2_mobilenetv2_pretrained, BO2_mobilenetv2_scratch, BO2_mobilenetv2_comparaison et
BO2_mobilenetv2_xai, pour que les deux modèles soient découpés, entraînés et évalués avec le même code.

Classes : MÊME mapping que bo2_classification.py (fonction category_from_name) :
    animals* -> animal, obstacles* et lost_tires* -> obstacle, cones* -> cone, vehicle* -> vehicle
Classes de pixels : 0 = fond, 1 = animal, 2 = obstacle, 3 = cone, 4 = vehicle.
Le nom de fichier sert UNIQUEMENT à construire la cible ; il n'est jamais une entrée du modèle.
Seule exception à « une classe par image » : les trois cônes annotés de vehicle02 (classe cone).
"""
import os, re, json, random, time
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import torchvision
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix

cv2.setNumThreads(0)

SEED = 42
IMG_W, IMG_H = 512, 288                       # même résolution que le BO1
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CLASSES = ["fond", "animal", "obstacle", "cone", "vehicle"]
OBJ_CLASSES = CLASSES[1:]                     # les 4 classes d'objets
N_CLS = len(CLASSES)


def category_from_name(name):
    """Même règle que bo2_classification.py : préfixe complet du nom de fichier -> classe BO2."""
    n = os.path.splitext(os.path.basename(str(name)))[0].lower()
    if n.startswith("animals"):
        return "animal"
    if n.startswith("cones"):
        return "cone"
    if n.startswith("obstacles") or n.startswith("lost_tires"):
        return "obstacle"
    if n.startswith("vehicle"):
        return "vehicle"
    raise ValueError(f"Catégorie inconnue dans le nom : {name}")
CLASS_COLORS = np.array([[0, 0, 0], [230, 60, 60], [60, 120, 230], [255, 150, 0], [170, 60, 200]], dtype=np.uint8)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------
# Données
# ------------------------------------------------------------------
def find_data_root():
    """Remonte l'arborescence jusqu'au dossier du dataset (frames/ + frame_list.json)."""
    here = os.path.abspath(os.getcwd())
    for _ in range(6):
        for cand in (here, os.path.join(here, "RoadAnomaly")):
            if os.path.exists(os.path.join(cand, "frames")) and os.path.exists(os.path.join(cand, "frame_list.json")):
                return cand
        here = os.path.dirname(here)
    raise FileNotFoundError(f"Dataset introuvable depuis {os.getcwd()} : ouvrir le notebook depuis le dépôt du projet.")


ROOT = find_data_root()
FRAMES = os.path.join(ROOT, "frames")


def get_paths(name):
    base = os.path.splitext(name)[0]
    return base, os.path.join(FRAMES, base + ".webp"), os.path.join(FRAMES, base + ".labels", "labels_semantic.png")


def get_category(name):
    m = re.match(r"^([a-zA-Z_]+?)\d", name)
    return m.group(1) if m else "autre"


def build_split_df():
    """Recalcule le découpage exactement comme dans le notebook d'exploration (contrôle de split.csv)."""
    with open(os.path.join(ROOT, "frame_list.json")) as f:
        frame_list = json.load(f)
    rows = []
    for name in frame_list:
        base, _, mask_p = get_paths(name)
        mask = np.array(Image.open(mask_p))
        rows.append({"name": base, "prop_anomalie": float((mask == 2).mean())})
    d = pd.DataFrame(rows)
    d["categorie"] = d["name"].apply(get_category)
    d["groupe"] = d["categorie"].apply(lambda c: c if c in ["animals", "obstacles", "cones"] else "autres")
    seuil = d["prop_anomalie"].median()
    d["taille"] = np.where(d["prop_anomalie"] < seuil, "petite", "grande")
    d["strat"] = d["groupe"] + "_" + d["taille"]
    trainval, test = train_test_split(d, test_size=0.20, stratify=d["strat"], random_state=SEED)
    train, val = train_test_split(trainval, test_size=0.25, stratify=trainval["strat"], random_state=SEED)
    d["split"] = "train"
    d.loc[val.index, "split"] = "val"
    d.loc[test.index, "split"] = "test"
    return d


def load_split():
    """Lit split.csv (commun à toute l'équipe), vérifie qu'il est identique au découpage recalculé, ajoute la classe."""
    df = pd.read_csv(os.path.join(ROOT, "split.csv"))
    df_new = build_split_df()
    same = dict(zip(df["name"], df["split"])) == dict(zip(df_new["name"], df_new["split"]))
    assert same, "split.csv ne correspond pas au découpage commun : ne pas continuer."
    df["classe"] = df["name"].map(lambda n: CLASSES.index(category_from_name(n))).astype(int)
    df["classe_nom"] = df["classe"].map(lambda k: CLASSES[k])
    tr = df[df.split == "train"]["name"].tolist()
    va = df[df.split == "val"]["name"].tolist()
    te = df[df.split == "test"]["name"].tolist()
    assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))
    assert len(tr) + len(va) + len(te) == len(df) == 60
    return df, tr, va, te


# Seule image qui contient deux types d'objets annotés : dans vehicle02, les instances 1 à 3 sont
# les trois cônes (environ 7 000 px), l'instance 4 l'engin de chantier (environ 85 000 px).
# On donne aux cônes la classe « cone » (labels_instance.png). La classe de l'image reste « vehicle » (objet majoritaire).
INSTANCE_CORRECTIONS = {"vehicle02_Broadmoor_Cones_Skidloader": {1: 3, 2: 3, 3: 3}}


def load_cache(df):
    """{nom: (image RGB uint8 (720,1280,3), masque de classes uint8 (720,1280) avec 0..4)}"""
    cache = {}
    cls_of = dict(zip(df["name"], df["classe"]))
    for n in df["name"]:
        _, img_p, mask_p = get_paths(n)
        img = np.array(Image.open(img_p).convert("RGB"))
        anom = np.array(Image.open(mask_p)) == 2
        m = anom.astype(np.uint8) * cls_of[n]
        if n in INSTANCE_CORRECTIONS:
            inst = np.array(Image.open(os.path.join(FRAMES, n + ".labels", "labels_instance.png")))
            for i, k in INSTANCE_CORRECTIONS[n].items():
                m[(inst == i) & anom] = k
        assert img.shape[:2] == m.shape
        cache[n] = (img, m)
    return cache


def to_tensor(img_uint8):
    x = (img_uint8.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


def denorm(x):
    img = x.numpy().transpose(1, 2, 0) * STD + MEAN
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def colorize(cls_map):
    return CLASS_COLORS[cls_map]


def overlay(img, cls_map, alpha=0.55):
    out = img.astype(np.float32).copy()
    m = cls_map > 0
    out[m] = (1 - alpha) * out[m] + alpha * CLASS_COLORS[cls_map[m]].astype(np.float32)
    return out.astype(np.uint8)


# ------------------------------------------------------------------
# Métriques BO2
# ------------------------------------------------------------------
def pixel_confusion(names, cache, pred_maps):
    """Matrice de confusion 5x5 sur tous les pixels (lignes = vérité, colonnes = prédiction), résolution d'origine."""
    C = np.zeros((N_CLS, N_CLS), np.int64)
    for n in names:
        gt = cache[n][1].ravel().astype(np.int64)
        pr = pred_maps[n].ravel().astype(np.int64)
        C += np.bincount(gt * N_CLS + pr, minlength=N_CLS * N_CLS).reshape(N_CLS, N_CLS)
    return C


def scores_from_confusion(C, eps=1e-9):
    """
    F1 pixel par classe d'objet (localisation ET classe justes), macro-F1 sur les classes présentes,
    métriques de détection (anomalie oui/non, comme le BO1) et part des pixels détectés qui ont la bonne classe.
    """
    out, f1s = {}, []
    for k in range(1, N_CLS):
        tp = C[k, k]; fp = C[:, k].sum() - tp; fn = C[k, :].sum() - tp
        p = tp / (tp + fp + eps); r = tp / (tp + fn + eps); f1 = 2 * p * r / (p + r + eps)
        out[f"precision_{CLASSES[k]}"], out[f"rappel_{CLASSES[k]}"], out[f"f1_{CLASSES[k]}"] = float(p), float(r), float(f1)
        if C[k, :].sum() > 0:
            f1s.append(f1)
    out["macro_f1"] = float(np.mean(f1s))
    tp = C[1:, 1:].sum(); fp = C[0, 1:].sum(); fn = C[1:, 0].sum()
    out["det_precision"] = float(tp / (tp + fp + eps)); out["det_rappel"] = float(tp / (tp + fn + eps))
    out["det_dice"] = float(2 * tp / (2 * tp + fp + fn + eps))
    out["bonne_classe_parmi_detectes"] = float(np.trace(C[1:, 1:]) / (C[1:, 1:].sum() + eps))
    # même mesure, mais moyenne des classes (sinon les animaux, majoritaires en pixels, dominent)
    acc_k = [C[k, k] / C[k, 1:].sum() for k in range(1, N_CLS) if C[k, 1:].sum() > 0]
    out["bonne_classe_equilibree"] = float(np.mean(acc_k)) if acc_k else float("nan")
    return out


def image_decision(pred_map):
    """Classe annoncée pour l'image : classe d'objet majoritaire parmi les pixels signalés (0 = rien signalé)."""
    counts = np.bincount(pred_map.ravel(), minlength=N_CLS)[1:]
    return int(np.argmax(counts) + 1) if counts.sum() > 0 else 0


def image_level(names, cache, pred_maps, cls_of):
    rows = []
    for n in names:
        gt_map, pr = cache[n][1], pred_maps[n]
        gt = gt_map > 0
        det = pr > 0
        rows.append({"name": n, "vraie": int(cls_of[n]), "predite": image_decision(pr),
                     "rappel_detection": float((det & gt).sum() / max(gt.sum(), 1)),
                     "pixels_bonne_classe": float((pr[gt] == gt_map[gt]).sum() / max(gt.sum(), 1)),
                     "fausses_alertes_%": float(100 * (det & ~gt).sum() / gt.size)})
    t = pd.DataFrame(rows)
    t["correct"] = t["vraie"] == t["predite"]
    labels = [1, 2, 3, 4]
    present = [k for k in labels if (t["vraie"] == k).any()]
    res = {"accuracy_image": float(t["correct"].mean()),
           "macro_f1_image": float(f1_score(t["vraie"], t["predite"], labels=present, average="macro", zero_division=0))}
    return res, t


def image_confusion(t):
    """Matrice 4 x 5 : lignes = vraie classe, colonnes = classe annoncée (dont « rien »)."""
    return confusion_matrix(t["vraie"], t["predite"], labels=[1, 2, 3, 4, 0])[:4]


# ------------------------------------------------------------------
# Régions (pour la chaîne « localiser puis classer »)
# ------------------------------------------------------------------
MERGE_PX = 31       # dilatation (px, résolution d'origine) qui regroupe les fragments proches (troupeau, éboulis)
MIN_PX = 150        # régions plus petites ignorées par le classifieur (bruit)
CROP = 224          # taille du recadrage donné au classifieur (entrée standard de MobileNetV2)
MARGIN = 0.25       # contexte ajouté autour de la boîte (25 % de chaque côté)


def extract_regions(binary, merge_px=MERGE_PX, min_px=MIN_PX):
    """binary (H,W) bool -> liste de dict {mask (H,W) bool, bbox (x0,y0,x1,y1), area}. Fragments proches regroupés."""
    b = binary.astype(np.uint8)
    if b.sum() == 0:
        return []
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (merge_px, merge_px))
    n, lab = cv2.connectedComponents(cv2.dilate(b, k), connectivity=8)
    regs = []
    for i in range(1, n):
        m = (lab == i) & binary
        a = int(m.sum())
        if a < min_px:
            continue
        ys, xs = np.where(m)
        regs.append({"mask": m, "bbox": (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1), "area": a})
    return regs


def square_box(bbox, margin=MARGIN, jitter=None, min_side=48):
    """Boîte carrée centrée sur la région, avec du contexte. jitter = (dx, dy, échelle) pour l'augmentation."""
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(x1 - x0, y1 - y0) * (1 + 2 * margin)
    if jitter is not None:
        dx, dy, s = jitter
        side *= s; cx += dx * side; cy += dy * side
    side = max(side, min_side)
    return int(round(cx - side / 2)), int(round(cy - side / 2)), int(round(side))


def crop_square(img, box, out=CROP):
    """Recadrage carré (complété par du gris moyen hors image) redimensionné à out x out."""
    x, y, s = box
    H, W = img.shape[:2]
    pad = int(max(0, -x, -y, x + s - W, y + s - H))
    if pad > 0:
        fill = tuple(int(v) for v in (MEAN * 255))
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=fill)
        x, y = x + pad, y + pad
    c = img[y:y + s, x:x + s]
    return cv2.resize(c, (out, out), interpolation=cv2.INTER_AREA if s > out else cv2.INTER_LINEAR)


def gt_regions(names, cache):
    """Régions de vérité terrain : liste de (nom, bbox, classe 0..3, aire, part de l'aire de l'image-objet)."""
    rows = []
    for n in names:
        m = cache[n][1]
        regs = extract_regions(m > 0)
        tot = sum(r["area"] for r in regs)
        for r in regs:
            cls = int(np.bincount(m[r["mask"]], minlength=N_CLS)[1:].argmax()) + 1     # classe majoritaire de la région
            rows.append({"name": n, "bbox": r["bbox"], "y": cls - 1, "area": r["area"], "poids": r["area"] / tot})
    return rows


# ------------------------------------------------------------------
# Modèles
# ------------------------------------------------------------------
def build_mobilenet(pretrained=True, n_out=4, dropout=0.2):
    """MobileNetV2 de torchvision. pretrained=True : poids ImageNet ; False : initialisation aléatoire (from scratch).
    La dernière couche (1000 classes ImageNet) est remplacée par Dropout + Linear(1280, 4)."""
    weights = torchvision.models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    try:
        m = torchvision.models.mobilenet_v2(weights=weights)
    except Exception as e:
        raise RuntimeError("Poids ImageNet de MobileNetV2 impossibles à télécharger : vérifier la connexion internet.") from e
    m.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.last_channel, n_out))
    return m


def load_loc_unet(path):
    """U-Net binaire du BO1 (01_unet_pretrained.ipynb), réutilisé tel quel pour localiser les anomalies."""
    if not os.path.exists(path):
        raise FileNotFoundError("Poids du U-Net du BO1 introuvables : " + path +
                                "\nCe fichier n'est pas versionné : le récupérer ou relancer 01_unet_pretrained.ipynb.")
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    m = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1)
    m.load_state_dict(ck["state_dict"])
    return m.to(DEVICE).eval(), ck


def n_params(model):
    return sum(p.numel() for p in model.parameters())


# ------------------------------------------------------------------
# Inférence
# ------------------------------------------------------------------
@torch.no_grad()
def unet_probs(model, names, cache, bs=4):
    """Sorties du U-Net à la résolution d'origine.
    Binaire (1 canal)  -> {nom: p_anomalie (H,W) float32}
    Multi (5 canaux)   -> {nom: (p_anomalie = 1 - p_fond, classe d'objet la plus probable 1..4)}"""
    model.eval(); out = {}
    H, W = next(iter(cache.values()))[1].shape
    for i in range(0, len(names), bs):
        batch = names[i:i + bs]
        x = torch.stack([to_tensor(cv2.resize(cache[n][0], (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)) for n in batch]).to(DEVICE)
        logits = F.interpolate(model(x), size=(H, W), mode="bilinear", align_corners=False)
        if logits.shape[1] == 1:
            p = torch.sigmoid(logits)[:, 0].cpu().numpy()
            for n, pp in zip(batch, p):
                out[n] = pp.astype(np.float32)
        else:
            pr = torch.softmax(logits, 1)
            p_anom = (1 - pr[:, 0]).cpu().numpy()
            cls = (pr[:, 1:].argmax(1) + 1).cpu().numpy().astype(np.uint8)
            for n, pa, c in zip(batch, p_anom, cls):
                out[n] = (pa.astype(np.float32), c)
    return out


def multiclass_maps(probs, thr):
    """U-Net multi-classe : pixel signalé si p_anomalie >= seuil, avec la classe d'objet la plus probable."""
    return {n: np.where(pa >= thr, c, 0).astype(np.uint8) for n, (pa, c) in probs.items()}


@torch.no_grad()
def classify_crops(clf, crops, bs=64):
    """Liste d'images uint8 CROPxCROP -> probabilités (N, 4)."""
    clf.eval()
    if len(crops) == 0:
        return np.zeros((0, 4), np.float32)
    outs = []
    for i in range(0, len(crops), bs):
        x = torch.stack([to_tensor(c) for c in crops[i:i + bs]]).to(DEVICE)
        outs.append(torch.softmax(clf(x), 1).cpu().numpy())
    return np.concatenate(outs)


def pipeline_maps(loc_probs, clf, names, cache, thr, return_regions=False):
    """Chaîne en deux étapes : régions signalées par le modèle de localisation, puis une classe par région.
    Les rares pixels signalés hors des régions retenues (fragments < MIN_PX) prennent la classe majoritaire de l'image."""
    maps, info = {}, {}
    for n in names:
        img = cache[n][0]
        binary = loc_probs[n] >= thr
        regs = extract_regions(binary)
        crops = [crop_square(img, square_box(r["bbox"])) for r in regs]
        P = classify_crops(clf, crops)
        pm = np.zeros(binary.shape, np.uint8)
        for r, p in zip(regs, P):
            pm[r["mask"]] = int(p.argmax()) + 1
        rest = binary & (pm == 0)
        if rest.any():
            if len(regs):
                votes = np.zeros(4)
                for r, p in zip(regs, P):
                    votes += r["area"] * p
                pm[rest] = int(votes.argmax()) + 1
            else:
                crop = crop_square(img, square_box((int(np.where(rest)[1].min()), int(np.where(rest)[0].min()),
                                                    int(np.where(rest)[1].max()) + 1, int(np.where(rest)[0].max()) + 1)))
                pm[rest] = int(classify_crops(clf, [crop])[0].argmax()) + 1
        maps[n] = pm
        info[n] = [{"bbox": r["bbox"], "area": r["area"], "probs": p} for r, p in zip(regs, P)]
    return (maps, info) if return_regions else maps


def sweep(names, cache, make_maps, thresholds):
    """Balaye le seuil de localisation, renvoie le tableau et le seuil au meilleur macro-F1 pixel."""
    rows = []
    for t in thresholds:
        s = scores_from_confusion(pixel_confusion(names, cache, make_maps(t)))
        rows.append({"thr": float(t), **s})
    r = pd.DataFrame(rows)
    return r, float(r.loc[r["macro_f1"].idxmax(), "thr"])


def time_inference(fn, n_warm=5, n_rep=30):
    for _ in range(n_warm):
        fn()
    ts = []
    for _ in range(n_rep):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return 1000 * float(np.median(ts))
