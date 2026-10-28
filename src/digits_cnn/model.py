"""Small CPU CNN with validation checkpointing and honest linear comparison."""

import copy
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .common import figure, save_report


class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 10),
        )

    def forward(self, X):
        return self.layers(X)


def split_indices(y):
    fit, rest = train_test_split(np.arange(len(y)), test_size=0.4, stratify=y, random_state=42)
    valid, test = train_test_split(rest, test_size=0.5, stratify=y[rest], random_state=42)
    return fit, valid, test


def evaluate(y, probability):
    return {
        "accuracy": float(accuracy_score(y, probability.argmax(axis=1))),
        "macro_f1": float(f1_score(y, probability.argmax(axis=1), average="macro")),
        "log_loss": float(log_loss(y, probability, labels=np.arange(10))),
    }


def train(root):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    digits = load_digits()
    X = torch.tensor(digits.images[:, None] / 16.0, dtype=torch.float32)
    y = torch.tensor(digits.target, dtype=torch.long)
    fit, valid, test = split_indices(digits.target)
    loader = DataLoader(
        TensorDataset(X[fit], y[fit]),
        batch_size=64,
        shuffle=True,
        generator=torch.Generator().manual_seed(42),
    )
    model = DigitCNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()
    best_loss, checkpoint, best_epoch = float("inf"), None, 0
    history = []
    for epoch in range(1, 41):
        model.train()
        total = 0.0
        for batch, target in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch), target)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        model.eval()
        with torch.no_grad():
            valid_loss = float(loss_fn(model(X[valid]), y[valid]))
        history.append(
            {"epoch": epoch, "train_loss": total / len(fit), "validation_loss": valid_loss}
        )
        if valid_loss < best_loss:
            best_loss, best_epoch = valid_loss, epoch
            checkpoint = copy.deepcopy(model.state_dict())
    model.load_state_dict(checkpoint)
    model.eval()
    linear = make_pipeline(
        StandardScaler(), LogisticRegression(C=1, max_iter=2000, random_state=42)
    )
    linear.fit(digits.data[fit], digits.target[fit])
    with torch.no_grad():
        cnn_valid = model(X[valid]).softmax(dim=1).numpy()
        cnn_test = model(X[test]).softmax(dim=1).numpy()
    validation = {
        "scaled_logistic": evaluate(digits.target[valid], linear.predict_proba(digits.data[valid])),
        "cnn": evaluate(digits.target[valid], cnn_valid),
    }
    selected = min(validation, key=lambda name: validation[name]["log_loss"])
    results = {
        "scaled_logistic": evaluate(digits.target[test], linear.predict_proba(digits.data[test])),
        "cnn": evaluate(digits.target[test], cnn_test),
    }
    ConfusionMatrixDisplay.from_predictions(
        digits.target[test], cnn_test.argmax(axis=1), cmap="Blues"
    )
    plt.title("CNN test confusion matrix")
    figure(root, "evaluation.png")
    plt.figure(figsize=(7, 4))
    plt.plot([h["epoch"] for h in history], [h["train_loss"] for h in history], label="Train")
    plt.plot(
        [h["epoch"] for h in history], [h["validation_loss"] for h in history], label="Validation"
    )
    plt.axvline(best_epoch, ls="--", color="#64748b", label="Selected checkpoint")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy")
    plt.legend()
    figure(root, "learning_curve.png")
    errors = np.flatnonzero(cnn_test.argmax(axis=1) != digits.target[test])[:12]
    _, axes = plt.subplots(2, 6, figsize=(10, 4))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, index in zip(axes.ravel(), errors):
        ax.imshow(digits.images[test[index]], cmap="gray_r")
        ax.set_title(
            f"true {digits.target[test[index]]} / pred {cnn_test[index].argmax()}", fontsize=9
        )
    figure(root, "errors.png")
    result = save_report(
        root,
        {
            "raw_rows": len(y),
            "split": {
                "train": len(fit),
                "validation": len(valid),
                "test": len(test),
                "method": "stratified image split; writer IDs unavailable",
            },
            "selected": selected,
            "best_cnn_epoch": best_epoch,
            "parameters": sum(p.numel() for p in model.parameters()),
            "validation": validation,
            "test": results,
        },
    )
    torch.save(model.state_dict(), root / "artifacts/cnn.pt")
    import joblib

    joblib.dump(linear, root / "artifacts/linear.joblib")
    (root / "reports/history.json").write_text(json.dumps(history, indent=2))
    return result


def predict(root, payload):
    pixels = np.asarray(payload["pixels"], dtype=float)
    if (
        pixels.shape != (8, 8)
        or not np.isfinite(pixels).all()
        or (pixels < 0).any()
        or (pixels > 16).any()
    ):
        raise ValueError("pixels must be an 8x8 array of finite values in 0..16")
    model = DigitCNN()
    model.load_state_dict(
        torch.load(root / "artifacts/cnn.pt", map_location="cpu", weights_only=True)
    )
    model.eval()
    with torch.no_grad():
        probability = (
            model(torch.tensor(pixels[None, None] / 16.0, dtype=torch.float32))
            .softmax(dim=1)
            .numpy()[0]
        )
    return {
        "digit": int(probability.argmax()),
        "confidence": float(probability.max()),
        "probabilities": probability.tolist(),
        "model": "cnn",
        "note": "Softmax confidence is not a calibrated probability guarantee",
    }
