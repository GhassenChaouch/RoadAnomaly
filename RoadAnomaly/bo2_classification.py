"""BO2: classify a localized road anomaly by object type.

The classifier is trained on anomaly crops made from the training masks, while
labels are derived from filename categories. At inference time, an image and a
trained U-Net are used to predict the anomaly crop; filenames and annotation
masks are not required.

Examples:
    python bo2_classification.py --model scratch --epochs 20
    python bo2_classification.py --model resnet18 --epochs 20

The default paths are relative to this file. The current dataset is small,
so the reported vehicle F1 is expected to be unstable until more vehicle
images are added.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import random
import time
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
import segmentation_models_pytorch as smp


CLASSES: Final = ("animal", "obstacle", "cone", "vehicle")
CLASS_TO_INDEX: Final = {name: index for index, name in enumerate(CLASSES)}
IMAGE_SIZE: Final = 224
IMAGE_MEAN: Final = (0.485, 0.456, 0.406)
IMAGE_STD: Final = (0.229, 0.224, 0.225)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def category_from_name(name: str) -> str:
    """Map the dataset's filename category to one BO2 class."""
    normalized = Path(name).stem.lower()
    if normalized.startswith("animals"):
        return "animal"
    if normalized.startswith("cones"):
        return "cone"
    if normalized.startswith("obstacles") or normalized.startswith("lost_tires"):
        return "obstacle"
    if normalized.startswith("vehicle"):
        return "vehicle"
    raise ValueError(f"Unsupported anomaly category in filename: {name}")


def anomaly_crop(image_path: Path, mask_path: Path, padding: float = 0.15) -> Image.Image:
    """Return a square-ish masked crop while retaining visual context."""
    image = Image.open(image_path).convert("RGB")
    mask = np.asarray(Image.open(mask_path).convert("L"))
    anomaly = mask > 0
    ys, xs = np.where(anomaly)
    if len(xs) == 0:
        raise ValueError(f"Mask contains no anomaly pixels: {mask_path}")

    left, right = int(xs.min()), int(xs.max()) + 1
    top, bottom = int(ys.min()), int(ys.max()) + 1
    width, height = right - left, bottom - top
    left = max(0, int(left - width * padding))
    right = min(image.width, int(right + width * padding))
    top = max(0, int(top - height * padding))
    bottom = min(image.height, int(bottom + height * padding))
    return image.crop((left, top, right, bottom))


def crop_from_mask(image: Image.Image, mask: np.ndarray, padding: float = 0.15) -> Image.Image:
    """Crop an in-memory image from a predicted binary anomaly mask."""
    anomaly = mask > 0
    ys, xs = np.where(anomaly)
    if len(xs) == 0:
        raise ValueError("The predicted anomaly mask contains no positive pixels.")
    left, right = int(xs.min()), int(xs.max()) + 1
    top, bottom = int(ys.min()), int(ys.max()) + 1
    width, height = right - left, bottom - top
    left = max(0, int(left - width * padding))
    right = min(image.width, int(right + width * padding))
    top = max(0, int(top - height * padding))
    bottom = min(image.height, int(bottom + height * padding))
    return image.crop((left, top, right, bottom))


def load_unet(checkpoint_path: Path, device: torch.device) -> nn.Module:
    """Load the project's pretrained U-Net for anomaly-mask prediction."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"U-Net checkpoint not found: {checkpoint_path}. "
            "Run 01_unet_pretrained.ipynb first."
        )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1)
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()


def predict_anomaly_crop(
    image_path: Path,
    unet: nn.Module,
    device: torch.device,
    threshold: float = 0.27,
) -> Image.Image:
    """Predict an anomaly mask with U-Net and return its image crop."""
    image = Image.open(image_path).convert("RGB")
    resized = image.resize((512, 288), Image.Resampling.BILINEAR)
    tensor = transforms.ToTensor()(resized)
    tensor = transforms.Normalize(IMAGE_MEAN, IMAGE_STD)(tensor).unsqueeze(0).to(device)
    with torch.no_grad():
        probability = torch.sigmoid(unet(tensor))[0, 0].cpu().numpy()
    mask = probability >= threshold
    mask_image = Image.fromarray((mask * 255).astype(np.uint8)).resize(
        image.size, Image.Resampling.NEAREST
    )
    return crop_from_mask(image, np.asarray(mask_image), padding=0.15)


def classify_image(
    image_path: Path,
    classifier: nn.Module,
    unet: nn.Module,
    device: torch.device,
    threshold: float = 0.27,
) -> tuple[str, float]:
    """Classify an uploaded image using only image pixels and model weights."""
    crop = predict_anomaly_crop(image_path, unet, device, threshold)
    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
        ]
    )
    with torch.no_grad():
        probabilities = torch.softmax(
            classifier(transform(crop).unsqueeze(0).to(device)), dim=1
        )[0]
    index = int(probabilities.argmax().item())
    return CLASSES[index], float(probabilities[index].item())


class AnomalyDataset(Dataset[tuple[Tensor, int]]):
    def __init__(
        self,
        frame_dir: Path,
        split_csv: Path,
        split: str,
        training: bool,
    ) -> None:
        self.frame_dir = frame_dir
        rows = pd.read_csv(split_csv)
        rows["class_name"] = rows["name"].map(category_from_name)
        self.rows = rows[rows["split"] == split].reset_index(drop=True)
        self.transform = transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.RandomHorizontalFlip() if training else transforms.Lambda(lambda image: image),
                transforms.ColorJitter(brightness=0.15, contrast=0.15) if training else transforms.Lambda(lambda image: image),
                transforms.ToTensor(),
                transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        row = self.rows.iloc[index]
        image_path = self.frame_dir / f"{row['name']}.webp"
        mask_path = self.frame_dir / f"{row['name']}.labels" / "labels_semantic.png"
        crop = anomaly_crop(image_path, mask_path)
        return self.transform(crop), CLASS_TO_INDEX[row["class_name"]]

    @property
    def labels(self) -> list[int]:
        return [CLASS_TO_INDEX[name] for name in self.rows["class_name"]]


class ScratchCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.classifier(self.features(inputs))


def build_model(name: str, pretrained: bool) -> nn.Module:
    if name == "scratch":
        return ScratchCNN(len(CLASSES))
    if name != "resnet18":
        raise ValueError(f"Unsupported model: {name}")
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    return model


def make_loader(dataset: AnomalyDataset, batch_size: int, training: bool) -> DataLoader[tuple[Tensor, int]]:
    if not training:
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    counts = np.bincount(dataset.labels, minlength=len(CLASSES))
    sample_weights = [1.0 / counts[label] for label in dataset.labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0)


def run_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, int]],
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, list[int], list[int]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    targets: list[int] = []
    predictions: list[int] = []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        if training:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * len(labels)
        targets.extend(labels.cpu().tolist())
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
    return total_loss / len(loader.dataset), targets, predictions


def print_metrics(title: str, targets: list[int], predictions: list[int]) -> None:
    labels = list(range(len(CLASSES)))
    macro = f1_score(targets, predictions, labels=labels, average="macro", zero_division=0)
    print(f"\n{title} | macro-F1: {macro:.3f}")
    print(
        classification_report(
            targets,
            predictions,
            labels=labels,
            target_names=CLASSES,
            zero_division=0,
            digits=3,
        )
    )
    print("Confusion matrix (rows=true, columns=predicted):")
    print(pd.DataFrame(confusion_matrix(targets, predictions, labels=labels), index=CLASSES, columns=CLASSES))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("resnet18", "scratch"), default="resnet18")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("outputs_bo2"))
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--inference-image", type=Path, default=None)
    parser.add_argument("--unet-checkpoint", type=Path, default=None)
    parser.add_argument("--unet-threshold", type=float, default=0.27)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def configure_logging(log_file: Path | None) -> logging.Logger:
    logger = logging.getLogger("bo2")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def main() -> None:
    args = parse_args()
    logger = configure_logging(args.log_file)
    set_seed(args.seed)
    root = Path(__file__).resolve().parent
    frame_dir = root / "frames"
    split_csv = root / "split.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started_total = time.perf_counter()

    train_set = AnomalyDataset(frame_dir, split_csv, "train", training=True)
    val_set = AnomalyDataset(frame_dir, split_csv, "val", training=False)
    test_set = AnomalyDataset(frame_dir, split_csv, "test", training=False)
    logger.info("Configuration: %s", json.dumps({
        "model": args.model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "device": str(device),
        "dataset": str(root),
        "classes": CLASSES,
        "unet_threshold": args.unet_threshold,
    }))
    logger.info("Class counts: %s", pd.Series(train_set.labels).map(dict(enumerate(CLASSES))).value_counts().reindex(CLASSES, fill_value=0).to_dict())
    logger.info("Dataset sizes: train=%d val=%d test=%d", len(train_set), len(val_set), len(test_set))

    train_loader = make_loader(train_set, args.batch_size, training=True)
    val_loader = make_loader(val_set, args.batch_size, training=False)
    test_loader = make_loader(test_set, args.batch_size, training=False)
    model = build_model(args.model, pretrained=args.model == "resnet18" and not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    best_state: dict[str, Tensor] | None = None
    best_macro_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        train_loss, _, _ = run_epoch(model, train_loader, criterion, device, optimizer)
        with torch.no_grad():
            val_loss, val_targets, val_predictions = run_epoch(model, val_loader, criterion, device)
        val_f1 = f1_score(val_targets, val_predictions, labels=list(range(len(CLASSES))), average="macro", zero_division=0)
        if val_f1 > best_macro_f1:
            best_macro_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
        logger.info(
            "Epoch %02d/%d | train_loss=%.4f | val_loss=%.4f | val_macro_f1=%.3f | %.1fs",
            epoch, args.epochs, train_loss, val_loss, val_f1,
            time.perf_counter() - started,
        )

    if best_state is None:
        raise RuntimeError("Training did not produce a best model state.")
    model.load_state_dict(best_state)
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / f"bo2_{args.model}_best.pt"
    torch.save({"state_dict": model.state_dict(), "classes": CLASSES, "seed": args.seed}, checkpoint)
    with torch.no_grad():
        _, test_targets, test_predictions = run_epoch(model, test_loader, criterion, device)
    test_report = classification_report(
        test_targets,
        test_predictions,
        labels=list(range(len(CLASSES))),
        target_names=CLASSES,
        zero_division=0,
        output_dict=True,
    )
    test_macro_f1 = f1_score(
        test_targets,
        test_predictions,
        labels=list(range(len(CLASSES))),
        average="macro",
        zero_division=0,
    )
    logger.info("Test macro-F1=%.3f | accuracy=%.3f", test_macro_f1, test_report["accuracy"])
    logger.info("Test classification report: %s", json.dumps(test_report))
    logger.info("Test confusion matrix: %s", confusion_matrix(test_targets, test_predictions, labels=list(range(len(CLASSES)))).tolist())
    logger.info("Saved checkpoint: %s", checkpoint)

    if args.inference_image is not None:
        unet_checkpoint = args.unet_checkpoint or (
            root / "outputs_unet_resnet34_pretrained" / "unet_resnet34_pretrained_best.pt"
        )
        inference_started = time.perf_counter()
        unet = load_unet(unet_checkpoint, device)
        classifier = build_model(args.model, pretrained=False).to(device)
        classifier.load_state_dict(torch.load(checkpoint, map_location=device)["state_dict"])
        classifier.eval()
        predicted_class, confidence = classify_image(
            args.inference_image,
            classifier,
            unet,
            device,
            threshold=args.unet_threshold,
        )
        logger.info(
            "Inference image=%s | predicted_class=%s | confidence=%.4f | elapsed=%.3fs",
            args.inference_image,
            predicted_class,
            confidence,
            time.perf_counter() - inference_started,
        )
    logger.info("Total elapsed=%.3fs", time.perf_counter() - started_total)


if __name__ == "__main__":
    main()
