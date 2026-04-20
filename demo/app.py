"""
Lightweight Gradio web demo: upload an image, get class + probabilities + assistant hint.

Run from project root:
  pip install gradio
  python demo/app.py

Optional:
  python demo/app.py --checkpoint path/to/weights.pth --server_port 7860
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import gradio as gr
from PIL import Image

from demo.inference import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    format_top3_markdown,
    load_model,
    pick_device,
    predict_image,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gradio waste classification demo")
    p.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
        help="Path to MobileNetV2 state_dict (.pth)",
    )
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--server_name", type=str, default="127.0.0.1")
    p.add_argument("--server_port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="Create a temporary public Gradio link")
    return p.parse_args()


def make_predict_fn(checkpoint: str, image_size: int):
    device = pick_device()
    model, _ = load_model(checkpoint_path=checkpoint, device=device)

    def predict_ui(image: Image.Image | None):
        if image is None:
            return (
                "Upload an image to begin.",
                {},
                "_Upload an image first._",
                "_No prediction yet._",
            )

        out = predict_image(image, model=model, device=device, image_size=image_size)
        summary = (
            f"**Predicted class:** `{out['predicted_label']}`  \n"
            f"**Top-1 confidence:** {out['confidence']:.3f}"
        )
        probs = out["probabilities"]
        top3_md = format_top3_markdown(out["top3"])
        hint = out["assistant_hint"]
        hint_md = f"**Assistant**  \n{hint}" if hint else "_No hint available._"
        return summary, probs, top3_md, hint_md

    return predict_ui


def build_demo(checkpoint: str, image_size: int) -> gr.Blocks:
    predict_ui = make_predict_fn(checkpoint, image_size)

    with gr.Blocks(title="Waste Classification Assistant") as demo:
        gr.Markdown(
            "## Image-Based Waste Classification Assistant\n"
            "Upload a photo of waste. The model predicts one of: "
            "**glass_metal**, **paper**, **plastic**, **trash**.\n\n"
            "_Demo uses the MobileNetV2 model when the default checkpoint exists._"
        )

        with gr.Row():
            with gr.Column():
                image_in = gr.Image(type="pil", label="Upload image")
                predict_btn = gr.Button("Predict", variant="primary")
            with gr.Column():
                summary_out = gr.Markdown(label="Prediction")
                probs_out = gr.Label(label="All class scores", num_top_classes=4)
                top3_out = gr.Markdown(label="Top-3")
                hint_out = gr.Markdown(label="Assistant tip")

        predict_btn.click(
            fn=predict_ui,
            inputs=[image_in],
            outputs=[summary_out, probs_out, top3_out, hint_out],
        )

        image_in.change(
            fn=predict_ui,
            inputs=[image_in],
            outputs=[summary_out, probs_out, top3_out, hint_out],
        )

    return demo


def main() -> None:
    args = parse_args()
    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}\n"
            "Train the Week 4 E1 model or pass --checkpoint to a valid .pth file."
        )

    demo = build_demo(args.checkpoint, args.image_size)
    demo.launch(server_name=args.server_name, server_port=args.server_port, share=args.share)


if __name__ == "__main__":
    main()
