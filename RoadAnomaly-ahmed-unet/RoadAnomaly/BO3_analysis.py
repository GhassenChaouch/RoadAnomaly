"""BO3 - Agrege les 12 entrainements, compare les modeles et produit tableaux et figures.

A lancer apres BO3_train_eval.py : python BO3_analysis.py
"""
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from BO3_train_eval import BLUR_SIGMA, H, W, augment, best_f1_threshold, fpr_at_95_tpr, load_image, load_mask, pixel_metrics, thresholded_metrics

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs_bo3"
RUNS = OUT / "runs"
SEEDS = [42, 43, 44]
MODELS = {"dcae": "M1 Autoencodeur convolutif debruitant", "feat": "M2 Autoencodeur de features ResNet18"}
SHORT = {"dcae": "M1 DCAE", "feat": "M2 Feature AE"}
COLORS = {"dcae": "#2a78d6", "feat": "#eb6834", "prior": "#52514e"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
N_BOOT, N_BINS = 2000, 4096

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
})

split = pd.read_csv(ROOT / "split.csv")
names = {s: split.loc[split["split"] == s, "name"].tolist() for s in ("train", "val", "test")}
masks = {s: [load_mask(n) for n in names[s]] for s in names}


def run_dir(model, aug, seed):
    return RUNS / f"{model}_aug{aug}_seed{seed}"


def load_results(model, aug, seed):
    return json.loads((run_dir(model, aug, seed) / "results.json").read_text())


def load_maps(model, aug, seed, subset="test"):
    return np.load(run_dir(model, aug, seed) / f"maps_{subset}.npy")


# ---------------------------------------------------------------- tableau principal
rows = []
for model in MODELS:
    for aug in (1, 0):
        results = [load_results(model, aug, s) for s in SEEDS]
        row = {"modele": SHORT[model], "augmentation": "oui" if aug else "non"}
        for key in ("auroc_pixel", "ap_pixel", "fpr95", "auroc_image_mean", "ap_image_mean"):
            values = [r["test"][key] for r in results]
            row[f"{key}_mean"], row[f"{key}_std"] = np.mean(values), np.std(values, ddof=1)
        for key in ("precision", "recall", "dice", "iou", "false_alert_surface_pct", "obstacles_detected"):
            values = [r["test_thresholded"][key] for r in results]
            row[f"{key}_mean"], row[f"{key}_std"] = np.mean(values), np.std(values, ddof=1)
        row["val_auroc_mean"] = np.mean([r["val"]["auroc_pixel"] for r in results])
        row["val_ap_mean"] = np.mean([r["val"]["ap_pixel"] for r in results])
        row["epochs_run_mean"] = np.mean([r["epochs_run"] for r in results])
        row["best_epoch_mean"] = np.mean([r["best_epoch"] for r in results])
        row["train_min_mean"] = np.mean([r["train_seconds"] for r in results]) / 60
        row["params_M"] = results[0]["trainable_params"] / 1e6
        row["weights_mb"] = results[0]["weights_mb"]
        row["inference_ms_median"] = float(np.median([r["inference_ms_median"] for r in results]))
        row["inference_ms_gpu_median"] = float(np.median([r["inference_ms_gpu_median"] for r in results]))
        row["device"] = results[0]["device"]
        rows.append(row)
table = pd.DataFrame(rows)
table.to_csv(OUT / "comparaison_bo3.csv", index=False)

# ---------------------------------------------------------------- reference : a priori de position
# Utilise les masques du train : c'est une reference, pas un modele non supervise.
prior = gaussian_filter(np.mean(masks["train"], axis=0).astype(np.float32), BLUR_SIGMA)
prior_maps = {s: [prior for _ in names[s]] for s in ("val", "test")}
prior_test = pixel_metrics(masks["test"], prior_maps["test"])
prior_thr = thresholded_metrics(masks["test"], prior_maps["test"], best_f1_threshold(masks["val"], prior_maps["val"]))
prevalence = float(np.concatenate([m.ravel() for m in masks["test"]]).mean())


# ---------------------------------------------------------------- bootstrap sur les images de test
def per_image_hist(maps):
    """Histogrammes par image (positifs, negatifs) : permet un bootstrap rapide."""
    scores = maps.ravel()
    edges = np.unique(np.quantile(scores, np.linspace(0, 1, N_BINS + 1)))
    pos, neg = [], []
    for mask, score in zip(masks["test"], maps):
        idx = np.clip(np.searchsorted(edges, score.ravel(), side="right") - 1, 0, len(edges) - 2)
        pos.append(np.bincount(idx[mask.ravel()], minlength=len(edges) - 1))
        neg.append(np.bincount(idx[~mask.ravel()], minlength=len(edges) - 1))
    return np.array(pos, dtype=np.float64), np.array(neg, dtype=np.float64)


def hist_metrics(pos, neg):
    below = np.concatenate([[0.0], np.cumsum(neg)[:-1]])
    auroc = float((pos * (below + 0.5 * neg)).sum() / (pos.sum() * neg.sum()))
    tp, fp = np.cumsum(pos[::-1]), np.cumsum(neg[::-1])
    precision = tp / np.maximum(tp + fp, 1e-12)
    ap = float((precision * pos[::-1]).sum() / pos.sum())
    return auroc, ap


rng = np.random.default_rng(42)
samples = rng.integers(0, 12, size=(N_BOOT, 12))
histograms = {(m, a, s): per_image_hist(load_maps(m, a, s)) for m in MODELS for a in (0, 1) for s in SEEDS}
boot = {}
for model in MODELS:
    for aug in (0, 1):
        values = np.zeros((N_BOOT, len(SEEDS), 2))
        for j, seed in enumerate(SEEDS):
            pos, neg = histograms[(model, aug, seed)]
            for b, sample in enumerate(samples):
                values[b, j] = hist_metrics(pos[sample].sum(0), neg[sample].sum(0))
        boot[(model, aug)] = values.mean(1)  # moyenne sur les seeds, un tirage d'images commun


def interval(values):
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


inference = {
    "n_boot": N_BOOT,
    "prevalence_test": prevalence,
    "reference_position": {"test": prior_test, "test_thresholded": prior_thr},
}
for model in MODELS:
    for aug in (0, 1):
        inference[f"{model}_aug{aug}"] = {"auroc_ic95": interval(boot[(model, aug)][:, 0]), "ap_ic95": interval(boot[(model, aug)][:, 1])}
    diff = boot[(model, 1)] - boot[(model, 0)]
    inference[f"{model}_gain_augmentation"] = {"auroc_ic95": interval(diff[:, 0]), "ap_ic95": interval(diff[:, 1])}
diff = boot[("feat", 1)] - boot[("dcae", 1)]
inference["feat_moins_dcae_aug1"] = {"auroc_ic95": interval(diff[:, 0]), "ap_ic95": interval(diff[:, 1])}
(OUT / "resume_statistique.json").write_text(json.dumps(inference, indent=2))

# ---------------------------------------------------------------- resultats par image
per_image = split.loc[split["split"] == "test", ["name", "categorie", "groupe", "taille", "prop_anomalie"]].copy().reset_index(drop=True)
for model in MODELS:
    auroc, ap = np.zeros((len(SEEDS), 12)), np.zeros((len(SEEDS), 12))
    for j, seed in enumerate(SEEDS):
        for i, (mask, score) in enumerate(zip(masks["test"], load_maps(model, 1, seed))):
            auroc[j, i] = roc_auc_score(mask.ravel(), score.ravel())
            ap[j, i] = average_precision_score(mask.ravel(), score.ravel())
    per_image[f"auroc_{model}"], per_image[f"ap_{model}"] = auroc.mean(0), ap.mean(0)
per_image.to_csv(OUT / "resultats_par_image_test.csv", index=False)
per_image.groupby("groupe")[["auroc_dcae", "auroc_feat", "ap_dcae", "ap_feat"]].mean().assign(n=per_image.groupby("groupe").size()).to_csv(OUT / "resultats_par_groupe.csv")
per_image.groupby("taille")[["auroc_dcae", "auroc_feat", "ap_dcae", "ap_feat"]].mean().assign(n=per_image.groupby("taille").size()).to_csv(OUT / "resultats_par_taille.csv")

# ---------------------------------------------------------------- figure ROC et PR (seed 42, augmentation)
y = np.concatenate([m.ravel() for m in masks["test"]])
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
for key, label, scores in [
    ("dcae", SHORT["dcae"], load_maps("dcae", 1, 42).ravel()),
    ("feat", SHORT["feat"], load_maps("feat", 1, 42).ravel()),
    ("prior", "Reference position", np.concatenate([p.ravel() for p in prior_maps["test"]])),
]:
    fpr, tpr, _ = roc_curve(y, scores)
    precision, recall, _ = precision_recall_curve(y, scores)
    step = max(len(fpr) // 3000, 1)
    style = {"color": COLORS[key], "linewidth": 2, "linestyle": "--" if key == "prior" else "-"}
    axes[0].plot(fpr[::step], tpr[::step], label=f"{label} (AUROC {roc_auc_score(y, scores):.3f})", **style)
    step = max(len(recall) // 3000, 1)
    axes[1].plot(recall[::step], precision[::step], label=f"{label} (AP {average_precision_score(y, scores):.3f})", **style)
axes[0].set(title="ROC pixel, test (seed 42)", xlabel="Taux de faux positifs", ylabel="Taux de vrais positifs", xlim=(0, 1), ylim=(0, 1))
axes[1].axhline(prevalence, color=MUTED, linewidth=1, linestyle=":", label=f"Hasard (AP {prevalence:.3f})")
axes[1].set(title="Precision-rappel pixel, test (seed 42)", xlabel="Rappel", ylabel="Precision", xlim=(0, 1), ylim=(0, 1))
for ax in axes:
    ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "courbes_roc_pr.png", dpi=130)
plt.close(fig)

# ---------------------------------------------------------------- figure ablation augmentation
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
for ax, key, title in zip(axes, ("auroc_pixel", "ap_pixel"), ("AUROC pixel (test)", "AP pixel (test)")):
    for k, model in enumerate(MODELS):
        for pos, aug, marker in [(k - 0.12, 0, "o"), (k + 0.12, 1, "s")]:
            r = table[(table.modele == SHORT[model]) & (table.augmentation == ("oui" if aug else "non"))].iloc[0]
            ax.errorbar(pos, r[f"{key}_mean"], yerr=r[f"{key}_std"], fmt=marker, color=COLORS[model], markersize=9, capsize=4, markeredgecolor="#fcfcfb")
            ax.annotate(f"{r[f'{key}_mean']:.3f}", (pos, r[f"{key}_mean"]), textcoords="offset points", xytext=(10 if aug else -10, 2), ha="left" if aug else "right", fontsize=9, color=INK)
    ax.set_xticks([0, 1], [SHORT[m] for m in MODELS])
    ax.set_xlim(-0.5, 1.6)
    ax.set_title(title)
    ax.grid(axis="x", visible=False)
axes[0].plot([], [], "o", color=MUTED, label="Sans augmentation")
axes[0].plot([], [], "s", color=MUTED, label="Avec augmentation")
axes[0].legend(frameon=False, fontsize=9, loc="best")
fig.tight_layout()
fig.savefig(OUT / "ablation_augmentation.png", dpi=130)
plt.close(fig)

# ---------------------------------------------------------------- figure entrainement
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
for ax, model in zip(axes, MODELS):
    history = pd.read_csv(run_dir(model, 1, 42) / "history.csv")
    ax.plot(history.epoch, history.train_loss, color=COLORS[model], linewidth=2, label="Train")
    ax.plot(history.epoch, history.val_loss, color=MUTED, linewidth=2, linestyle="--", label="Validation")
    ax.set(title=f"{SHORT[model]} (seed 42)", xlabel="Epoch", ylabel="Erreur de reconstruction (MSE)")
    ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "courbes_apprentissage.png", dpi=130)
plt.close(fig)

# ---------------------------------------------------------------- figure qualitative
maps = {m: load_maps(m, 1, 42) for m in MODELS}
limits = {m: np.percentile(maps[m], 99.5) for m in MODELS}
order = per_image.assign(score=per_image.auroc_feat).sort_values("score", ascending=False).index.tolist()
selection = order[:3] + order[-3:]
fig, axes = plt.subplots(len(selection), 4, figsize=(13, 2.15 * len(selection)))
for row, idx in enumerate(selection):
    image = load_image(names["test"][idx]).permute(1, 2, 0).numpy()
    overlay = image.copy()
    overlay[masks["test"][idx]] = 0.45 * overlay[masks["test"][idx]] + 0.55 * np.array([0.92, 0.41, 0.2])
    axes[row, 0].imshow(image)
    axes[row, 1].imshow(overlay)
    for col, model in enumerate(MODELS, start=2):
        axes[row, col].imshow(maps[model][idx], cmap="Blues", vmin=0, vmax=limits[model])
    auroc = {m: roc_auc_score(masks["test"][idx].ravel(), maps[m][idx].ravel()) for m in MODELS}
    axes[row, 2].set_xlabel(f"AUROC {auroc['dcae']:.2f}", fontsize=9, color=INK)
    axes[row, 3].set_xlabel(f"AUROC {auroc['feat']:.2f}", fontsize=9, color=INK)
    axes[row, 0].set_ylabel(names["test"][idx][:16], fontsize=8, color=INK)
    for ax in axes[row]:
        ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
for col, title in enumerate(["Image", "Verite terrain", SHORT["dcae"] + " : erreur", SHORT["feat"] + " : erreur"]):
    axes[0, col].set_title(title)
fig.tight_layout()
fig.savefig(OUT / "exemples_test.png", dpi=110)
plt.close(fig)

# ---------------------------------------------------------------- figure augmentation
random.seed(42)
torch.manual_seed(42)
example = load_image(names["train"][0])
fig, axes = plt.subplots(2, 3, figsize=(12, 4.7))
for k, ax in enumerate(axes.ravel()):
    ax.imshow((example if k == 0 else augment(example)).permute(1, 2, 0).numpy())
    ax.set_title("Image d'origine" if k == 0 else f"Version augmentee {k}", fontsize=10)
    ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
fig.tight_layout()
fig.savefig(OUT / "exemples_augmentation.png", dpi=100)
plt.close(fig)

pd.set_option("display.width", 250, "display.max_columns", 50)
print(table.round(3).T)
print(json.dumps(inference, indent=2))
print(per_image.round(3).to_string())
