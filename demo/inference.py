"""
Shared inference for the waste-classification demo (MobileNetV2, 4 classes).

Class order matches `datasets.ImageFolder` on folder names sorted alphabetically:
glass_metal, paper, plastic, trash (same as training in this repo).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models

# Repo root
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.transforms import get_eval_transform  # noqa: E402

CLASS_NAMES = ["glass_metal", "paper", "plastic", "trash"]

DEFAULT_CHECKPOINT = (
    _ROOT
    / "outputs"
    / "models"
    / "mobilenet_v2_img224_bs32_ep10_lr0.0001_adam_freeze0_aug1_wloss0_4class_best.pth"
)

ASSISTANT_HINTS = {
    "glass_metal": (
        "If the item is predicted as glass/metal, place it in the **glass/metal recycling** stream "
    ),
    "paper": (
        "If the item is predicted as paper, place it in the **paper recycling bin** "
    ),
    "plastic": (
        "If the item is predicted as plastic, place it in the **plastic recycling bin** "
    ),
    "trash": (
        "If the item is predicted as trash, dispose of it in **general waste** unless local rules say otherwise. "
    ),
}


def project_root() -> Path:
    return _ROOT


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(num_classes: int = 4) -> nn.Module:
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def load_model(
    checkpoint_path: str | Path | None = None,
    device: torch.device | None = None,
) -> tuple[nn.Module, torch.device]:
    ckpt = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT
    if not ckpt.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt}\n"
            "Train Week 4 E1 (MobileNet fine-tune + CE) or pass a valid --checkpoint path."
        )
    dev = device or pick_device()
    model = build_model(len(CLASS_NAMES)).to(dev)
    state = torch.load(ckpt, map_location=dev)
    model.load_state_dict(state)
    model.eval()
    return model, dev


def preprocess_image(image: Image.Image, image_size: int = 224) -> torch.Tensor:
    """Return batch tensor (1, 3, H, W) on CPU."""
    tfm = get_eval_transform(image_size)
    return tfm(image.convert("RGB")).unsqueeze(0)


def predict_image(
    image: Image.Image,
    *,
    model: nn.Module | None = None,
    device: torch.device | None = None,
    image_size: int = 224,
) -> dict[str, Any]:
    """
    Run a single-image prediction.

    If ``model`` is None, loads DEFAULT_CHECKPOINT (lazy singleton not used; caller
    should load once for UI/CLI efficiency).
    """
    if model is None:
        model, device = load_model(device=device)
    else:
        if device is None:
            device = next(model.parameters()).device

    x = preprocess_image(image, image_size=image_size).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    probs_cpu = probs.detach().float().cpu()
    conf, idx = float(probs_cpu.max().item()), int(probs_cpu.argmax().item())
    pred_label = CLASS_NAMES[idx]

    prob_dict = {CLASS_NAMES[i]: float(probs_cpu[i]) for i in range(len(CLASS_NAMES))}

    top3_idx = torch.topk(probs_cpu, k=min(3, len(CLASS_NAMES))).indices.tolist()
    top3 = [(CLASS_NAMES[i], float(probs_cpu[i])) for i in top3_idx]

    return {
        "predicted_label": pred_label,
        "confidence": conf,
        "probabilities": prob_dict,
        "top3": top3,
        "assistant_hint": ASSISTANT_HINTS.get(pred_label, ""),
    }


def predict_image_json(image: Image.Image, **kwargs: Any) -> str:
    """JSON string for CLI / piping."""
    out = predict_image(image, **kwargs)
    # top3 as list of [name, p]
    serializable = {
        "predicted_label": out["predicted_label"],
        "confidence": out["confidence"],
        "probabilities": out["probabilities"],
        "top3": [{"label": a, "probability": b} for a, b in out["top3"]],
        "assistant_hint": out["assistant_hint"],
    }
    return json.dumps(serializable, indent=2, ensure_ascii=True)


def format_top3_markdown(top3: list[tuple[str, float]]) -> str:
    lines = ["**Top-3 predictions**", ""]
    for rank, (name, p) in enumerate(top3, start=1):
        lines.append(f"{rank}. `{name}`: **{p:.3f}**")
    return "\n".join(lines)
