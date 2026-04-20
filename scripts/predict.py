"""
CLI: single-image inference using the same pipeline as the Gradio demo.

Example:
  python scripts/predict.py --image path/to/waste.jpg
  python scripts/predict.py --image path/to/waste.jpg --checkpoint outputs/models/custom_best.pth
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PIL import Image

from demo.inference import (  # noqa: E402
    load_model,
    pick_device,
    predict_image,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="4-class waste prediction (MobileNetV2)")
    p.add_argument("--image", type=str, required=True, help="Path to an image file")
    p.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Optional path to .pth state_dict (default: Week 4 E1 MobileNet checkpoint)",
    )
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--json", action="store_true", help="Print full JSON to stdout")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    img_path = Path(args.image)
    if not img_path.is_file():
        raise FileNotFoundError(str(img_path))

    ckpt = args.checkpoint.strip() or None
    device = pick_device()
    model, _ = load_model(checkpoint_path=ckpt, device=device)

    image = Image.open(img_path).convert("RGB")
    out = predict_image(image, model=model, device=device, image_size=args.image_size)

    if args.json:
        payload = {
            "image": str(img_path),
            "predicted_label": out["predicted_label"],
            "confidence": out["confidence"],
            "probabilities": out["probabilities"],
            "top3": [{"label": a, "probability": b} for a, b in out["top3"]],
            "assistant_hint": out["assistant_hint"],
        }
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        return

    print(f"image: {img_path}")
    print(f"predicted: {out['predicted_label']} (confidence {out['confidence']:.4f})")
    print("probabilities:")
    for k, v in out["probabilities"].items():
        print(f"  {k}: {v:.4f}")
    print("top-3:")
    for rank, (name, p) in enumerate(out["top3"], start=1):
        print(f"  {rank}. {name}: {p:.4f}")
    print("assistant:")
    print(f"  {out['assistant_hint']}")


if __name__ == "__main__":
    main()
