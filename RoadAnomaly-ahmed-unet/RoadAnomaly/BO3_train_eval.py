"""BO3 - Detection d'objets jamais vus, sans etiquettes d'anomalies a l'entrainement.

Deux autoencodeurs, entraines sur les images de `split.csv` (train) SANS lire les masques :
  dcae : autoencodeur convolutif debruitant sur les pixels, appris depuis zero ;
  feat : autoencodeur sur les caracteristiques d'un ResNet18 ImageNet gele.
Le score d'anomalie d'un pixel est son erreur de reconstruction.

Les masques ne servent qu'apres l'entrainement, sur val (choix du seuil) et test (metriques).

Exemple :
  python BO3_train_eval.py --model feat --aug 1 --seed 42
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from torchvision.models import ResNet18_Weights, resnet18

ROOT = Path(__file__).resolve().parent
W, H = 512, 288  # resolution commune de l'equipe
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
BLUR_SIGMA = 4.0


# ---------------------------------------------------------------- donnees
def load_image(name):
    img = Image.open(ROOT / "frames" / f"{name}.webp").convert("RGB").resize((W, H), Image.BILINEAR)
    return torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0


def load_mask(name):
    mask = Image.open(ROOT / "frames" / f"{name}.labels" / "labels_semantic.png").resize((W, H), Image.NEAREST)
    return np.array(mask) == 2


def augment(img):
    """Une image [0,1] CxHxW -> une version augmentee (flip, rotation, zoom, couleur)."""
    if random.random() < 0.5:
        img = TF.hflip(img)
    # Rotation avec bords en miroir : evite d'apprendre des coins noirs.
    pad = 64
    img = F.pad(img.unsqueeze(0), (pad, pad, pad, pad), mode="reflect").squeeze(0)
    img = TF.rotate(img, random.uniform(-10, 10), interpolation=TF.InterpolationMode.BILINEAR)
    img = img[:, pad:pad + H, pad:pad + W]
    # Zoom aleatoire au ratio 16:9.
    scale = random.uniform(0.7, 1.0)
    h, w = int(H * scale), int(W * scale)
    top, left = random.randint(0, H - h), random.randint(0, W - w)
    img = TF.resized_crop(img, top, left, h, w, [H, W], interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
    img = TF.adjust_brightness(img, random.uniform(0.8, 1.2))
    img = TF.adjust_contrast(img, random.uniform(0.8, 1.2))
    img = TF.adjust_saturation(img, random.uniform(0.8, 1.2))
    return img.clamp(0, 1)


# ---------------------------------------------------------------- modeles
def conv(c_in, c_out, stride=2):
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))


def deconv(c_in, c_out):
    return nn.Sequential(
        nn.ConvTranspose2d(c_in, c_out, 4, 2, 1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)
    )


class DenoisingCAE(nn.Module):
    """Modele 1 : autoencodeur convolutif debruitant, goulot 64 x 9 x 16."""

    def __init__(self, noise_std=0.1):
        super().__init__()
        self.noise_std = noise_std
        self.encoder = nn.Sequential(conv(3, 32), conv(32, 64), conv(64, 128), conv(128, 192), conv(192, 64))
        self.decoder = nn.Sequential(
            deconv(64, 192), deconv(192, 128), deconv(128, 64), deconv(64, 32),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def training_loss(self, x):
        noisy = (x + self.noise_std * torch.randn_like(x)).clamp(0, 1)
        return F.mse_loss(self(noisy), x)

    def reconstruction_error(self, x):
        return (self(x) - x).pow(2).mean(1, keepdim=True)


class ResNetFeatures(nn.Module):
    """ResNet18 ImageNet gele : couches 1 a 3 concatenees a la resolution 1/8."""

    def __init__(self):
        super().__init__()
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1, self.layer2, self.layer3 = net.layer1, net.layer2, net.layer3
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    @torch.no_grad()
    def forward(self, x):
        x = (x - IMAGENET_MEAN.to(x)) / IMAGENET_STD.to(x)
        c1 = self.layer1(self.stem(x))
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        size = c2.shape[-2:]
        return torch.cat(
            [F.adaptive_avg_pool2d(c1, size), c2, F.interpolate(c3, size=size, mode="bilinear", align_corners=False)], 1
        )


class FeatureAE(nn.Module):
    """Modele 2 : autoencodeur (convolutions 1x1) sur les caracteristiques ResNet18 standardisees."""

    def __init__(self, in_channels=448, latent=64):
        super().__init__()
        self.extractor = ResNetFeatures()
        self.register_buffer("mean", torch.zeros(1, in_channels, 1, 1))
        self.register_buffer("std", torch.ones(1, in_channels, 1, 1))
        self.autoencoder = nn.Sequential(
            conv(in_channels, 224, 1), conv(224, 128, 1), nn.Conv2d(128, latent, 1),
            nn.ReLU(inplace=True),
            conv(latent, 128, 1), conv(128, 224, 1), nn.Conv2d(224, in_channels, 1),
        )

    def train(self, mode=True):
        super().train(mode)
        self.extractor.eval()  # BatchNorm du ResNet toujours fige
        return self

    @torch.no_grad()
    def fit_standardization(self, images):
        device = self.mean.device
        features = torch.cat([self.extractor(x.unsqueeze(0).to(device)) for x in images], 0)
        self.mean.copy_(features.mean(dim=(0, 2, 3), keepdim=True))
        self.std.copy_(features.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-6))

    def target(self, x):
        return (self.extractor(x) - self.mean) / self.std

    def training_loss(self, x):
        target = self.target(x)
        return F.mse_loss(self.autoencoder(target), target)

    def reconstruction_error(self, x):
        target = self.target(x)
        return (self.autoencoder(target) - target).pow(2).mean(1, keepdim=True)


def gaussian_blur(x, sigma=BLUR_SIGMA):
    radius = int(3 * sigma)
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-coords.pow(2) / (2 * sigma**2))
    kernel = (kernel / kernel.sum()).to(x)
    x = F.pad(x, (radius, radius, radius, radius), mode="reflect")
    x = F.conv2d(x, kernel.view(1, 1, 1, -1))
    return F.conv2d(x, kernel.view(1, 1, -1, 1))


@torch.no_grad()
def anomaly_map(model, image):
    model.eval()
    device = next(model.parameters()).device
    err = model.reconstruction_error(image.unsqueeze(0).to(device))
    err = F.interpolate(err, size=(H, W), mode="bilinear", align_corners=False)
    return gaussian_blur(err).squeeze().cpu().numpy()


# ---------------------------------------------------------------- metriques
def fpr_at_95_tpr(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    return float(fpr[np.searchsorted(tpr, 0.95)])


def pixel_metrics(masks, maps):
    y, s = np.concatenate([m.ravel() for m in masks]), np.concatenate([m.ravel() for m in maps])
    return {
        "auroc_pixel": float(roc_auc_score(y, s)),
        "ap_pixel": float(average_precision_score(y, s)),
        "fpr95": fpr_at_95_tpr(y, s),
        "auroc_image_mean": float(np.mean([roc_auc_score(m.ravel(), a.ravel()) for m, a in zip(masks, maps)])),
        "ap_image_mean": float(np.mean([average_precision_score(m.ravel(), a.ravel()) for m, a in zip(masks, maps)])),
    }


def best_f1_threshold(masks, maps):
    y, s = np.concatenate([m.ravel() for m in masks]), np.concatenate([m.ravel() for m in maps])
    precision, recall, thresholds = precision_recall_curve(y, s)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[np.argmax(f1[:-1])])


def thresholded_metrics(masks, maps, threshold):
    tp = fp = fn = 0
    detected, false_surface = 0, []
    for mask, score in zip(masks, maps):
        pred = score >= threshold
        tp_i = int((pred & mask).sum())
        tp, fp, fn = tp + tp_i, fp + int((pred & ~mask).sum()), fn + int((~pred & mask).sum())
        detected += tp_i / mask.sum() >= 0.5
        false_surface.append((pred & ~mask).mean())
    return {
        "threshold": threshold,
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "dice": 2 * tp / max(2 * tp + fp + fn, 1),
        "iou": tp / max(tp + fp + fn, 1),
        "obstacles_detected": int(detected),
        "n_images": len(masks),
        "false_alert_surface_pct": 100 * float(np.mean(false_surface)),
    }


# ---------------------------------------------------------------- entrainement
def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    split = pd.read_csv(ROOT / "split.csv")
    names = {s: split.loc[split["split"] == s, "name"].tolist() for s in ("train", "val", "test")}
    train_images = [load_image(n) for n in names["train"]]  # aucune lecture de masque ici
    val_images = [load_image(n) for n in names["val"]]

    model = (DenoisingCAE() if args.model == "dcae" else FeatureAE()).to(device)
    if args.model == "feat":
        model.fit_standardization(train_images)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out = ROOT / args.out / f"{args.model}_aug{args.aug}_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    weights = out / "best.pt"
    history, best_val, stale = [], float("inf"), 0
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = [i for _ in range(args.repeat) for i in random.sample(range(len(train_images)), len(train_images))]
        train_loss = 0.0
        for start in range(0, len(order), args.batch):
            batch = [train_images[i] for i in order[start : start + args.batch]]
            x = torch.stack([augment(b) if args.aug else b for b in batch]).to(device)
            optimizer.zero_grad()
            loss = model.training_loss(x)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch)
        scheduler.step()
        train_loss /= len(order)

        # Early stopping sur l'erreur de reconstruction de validation : aucun masque utilise.
        model.eval()
        with torch.no_grad():
            val_loss = float(np.mean([model.reconstruction_error(x.unsqueeze(0).to(device)).mean().item() for x in val_images]))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "seconds": time.time() - started})
        print(f"[{out.name}] {epoch}/{args.epochs} train={train_loss:.5f} val={val_loss:.5f} t={time.time()-started:.0f}s", flush=True)
        if val_loss < best_val:
            best_val, stale = val_loss, 0
            torch.save(model.state_dict(), weights)
        else:
            stale += 1
            if stale >= args.patience:
                break

    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()
    train_seconds = time.time() - started

    # Les masques n'entrent en jeu qu'a partir d'ici.
    test_images = [load_image(n) for n in names["test"]]
    masks = {s: [load_mask(n) for n in names[s]] for s in ("val", "test")}
    maps = {
        "val": [anomaly_map(model, x) for x in val_images],
        "test": [anomaly_map(model, x) for x in test_images],
    }
    threshold = best_f1_threshold(masks["val"], maps["val"])

    def median_inference_ms(x):
        timings = []
        for i in range(35):
            t0 = time.perf_counter()
            anomaly_map(model, x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            if i >= 5:
                timings.append(1000 * (time.perf_counter() - t0))
        return float(np.median(timings))

    inference_gpu = median_inference_ms(test_images[0]) if device.type == "cuda" else None
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    # Temps CPU toujours mesure : c'est l'unite utilisee par le reste de l'equipe.
    model.cpu()
    device = torch.device("cpu")
    inference_cpu = median_inference_ms(test_images[0])

    n_params = sum(p.numel() for p in params)
    results = {
        "model": args.model, "aug": args.aug, "seed": args.seed, "lr": args.lr, "epochs_run": len(history),
        "best_epoch": min(history, key=lambda h: h["val_loss"])["epoch"], "repeat": args.repeat,
        "batch": args.batch, "resolution": [W, H], "blur_sigma": BLUR_SIGMA,
        "trainable_params": n_params, "weights_mb": weights.stat().st_size / 1e6,
        "device": device_name, "inference_ms_median": inference_cpu, "inference_ms_gpu_median": inference_gpu,
        "train_seconds": train_seconds,
        "val": pixel_metrics(masks["val"], maps["val"]),
        "test": pixel_metrics(masks["test"], maps["test"]),
        "test_thresholded": thresholded_metrics(masks["test"], maps["test"], threshold),
        "test_names": names["test"],
    }
    (out / "results.json").write_text(json.dumps(results, indent=2))
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    for s in ("val", "test"):
        np.save(out / f"maps_{s}.npy", np.stack(maps[s]).astype(np.float32))
    print(f"[{out.name}] termine en {time.time()-started:.0f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["dcae", "feat"], required=True)
    parser.add_argument("--aug", type=int, choices=[0, 1], default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--repeat", type=int, default=4, help="passages sur le train par epoch")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", default="outputs_bo3/runs")
    parser.add_argument("--cpu", action="store_true", help="forcer le CPU meme si un GPU est disponible")
    run(parser.parse_args())
