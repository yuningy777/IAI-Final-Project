"""
Thin wrapper around ``demo.inference`` for quick CLI checks (same defaults as Gradio).

Prefer ``scripts/predict.py`` for structured output (probabilities + JSON).

Example:
  python scripts/demo_predict.py --image path/to/photo.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PIL import Image

from demo.inference import DEFAULT_CHECKPOINT, load_model, pick_device, predict_image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="4-class waste demo inference (MobileNetV2)")
    p.add_argument("--image", type=str, required=True, help="Path to an RGB image file")
    p.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Path to model state_dict (.pth); default is Week 4 E1 MobileNet checkpoint",
    )
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--topk", type=int, default=3)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    img_path = Path(args.image)
    if not img_path.exists():
        raise FileNotFoundError(str(img_path))

    ckpt = args.checkpoint.strip() or None
    device = pick_device()
    model, _ = load_model(checkpoint_path=ckpt, device=device)

    image = Image.open(img_path).convert("RGB")
    out = predict_image(image, model=model, device=device, image_size=args.image_size)

    topk = min(args.topk, len(out["top3"]))
    top = out["top3"][:topk]

    ckpt_resolved = Path(ckpt) if ckpt else DEFAULT_CHECKPOINT

    print(f"image: {img_path}")
    print(f"checkpoint: {ckpt_resolved}")
    print(f"device: {device}")
    print("top predictions:")
    for rank, (name, c) in enumerate(top, start=1):
        print(f"  {rank}. {name}: {c:.4f}")


if __name__ == "__main__":
    main()
