"""
Auto-detect UI elements in mobile game screenshots using Grounding DINO + SAM 2.

Pipeline:
  1. Grounding DINO finds UI elements by text description (zero-shot detection)
  2. SAM 2 takes each detected bounding box and produces a precise mask

Usage:
  python detect_ui.py --image my_screenshots/game.png
  python detect_ui.py --image my_screenshots/game.png --prompt "button. icon. menu panel. dialog."
  python detect_ui.py --image my_screenshots/game.png --box-threshold 0.3 --text-threshold 0.2

Options:
  --prompt           Text prompt describing UI elements to detect (period-separated)
  --box-threshold    Grounding DINO box confidence threshold (default: 0.35)
  --text-threshold   Grounding DINO text matching threshold (default: 0.25)
  --output-dir       Where to save results (default: ./output)
  --no-sam            Skip SAM 2 mask refinement, only show Grounding DINO boxes
"""

import argparse
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

DEFAULT_PROMPT = "button. text. icon. label. slider."

SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_small.pt"
SAM2_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_s.yaml"
GDINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect UI elements in mobile game screenshots with Grounding DINO + SAM 2"
    )
    parser.add_argument(
        "--image", required=True, help="Path to the screenshot image file"
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Period-separated text prompt for Grounding DINO (default: common UI element types)",
    )
    parser.add_argument(
        "--box-threshold",
        type=float,
        default=0.075,
        help="Grounding DINO box confidence threshold (default: 0.075). Lower = more detections.",
    )
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=0.1,
        help="Grounding DINO text matching threshold (default: 0.1). Lower = more detections.",
    )
    parser.add_argument(
        "--output-dir",
        default="./output",
        help="Directory to save output images (default: ./output)",
    )
    parser.add_argument(
        "--no-sam",
        action="store_true",
        help="Skip SAM 2 mask refinement, only show Grounding DINO bounding boxes.",
    )
    return parser.parse_args()


def setup_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device


def random_color(seed):
    rng = np.random.RandomState(seed)
    return rng.uniform(0.3, 0.95, size=3)


def draw_box_label(ax, x1, y1, x2, y2, color, label):
    rect = plt.Rectangle(
        (x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor="none"
    )
    ax.add_patch(rect)
    ax.text(
        x1, y1 - 4, label,
        fontsize=7, color="white", fontweight="bold",
        bbox=dict(facecolor=color, alpha=0.85, pad=1.5, edgecolor="none"),
    )


def draw_mask_overlay(ax, mask, color, alpha=0.4):
    mask = mask.astype(bool)
    h, w = mask.shape
    overlay = np.zeros((h, w, 4), dtype=np.float32)
    overlay[mask] = [*color, alpha]
    ax.imshow(overlay)

    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for contour in contours:
        contour = contour.squeeze(1)
        if len(contour) > 1:
            closed = np.vstack([contour, contour[0:1]])
            ax.plot(closed[:, 0], closed[:, 1], color=color, linewidth=1.5)


def load_grounding_dino(device):
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    print(f"Loading Grounding DINO ({GDINO_MODEL_ID})...")
    processor = AutoProcessor.from_pretrained(GDINO_MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_MODEL_ID).to(device)
    model.eval()
    return processor, model


def run_grounding_dino(pil_image, prompt, processor, model, device, box_threshold, text_threshold):
    inputs = processor(images=pil_image, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[pil_image.size[::-1]],
    )
    return results[0]


def load_sam2(device):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    print("Loading SAM 2 small model...")
    model = build_sam2(SAM2_MODEL_CFG, SAM2_CHECKPOINT, device=device)
    predictor = SAM2ImagePredictor(model)
    return predictor


def run_sam2(predictor, image_np, boxes, device):
    """Run SAM 2 on each detected box to get precise masks."""
    masks_out = []
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16):
        predictor.set_image(image_np)
        for box in boxes:
            box_np = box.cpu().numpy()
            masks, scores, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box_np[None, :],
                multimask_output=False,
            )
            masks_out.append(masks[0])
    return masks_out


def main():
    args = parse_args()

    if not os.path.isfile(args.image):
        print(f"Error: image not found: {args.image}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    device = setup_device()

    # Load image
    pil_image = Image.open(args.image).convert("RGB")
    image_np = np.array(pil_image)
    h, w = image_np.shape[:2]
    print(f"Image loaded: {args.image} ({w}x{h})")

    # --- Stage 1: Grounding DINO ---
    processor, gdino_model = load_grounding_dino(device)

    prompt = args.prompt.lower()
    if not prompt.endswith("."):
        prompt += "."
    print(f"Prompt: \"{prompt}\"")
    print(f"Thresholds: box={args.box_threshold}, text={args.text_threshold}")

    results = run_grounding_dino(
        pil_image, prompt, processor, gdino_model, device,
        args.box_threshold, args.text_threshold,
    )

    boxes = results["boxes"]       # tensor of shape (N, 4) in xyxy format
    scores = results["scores"]     # tensor of shape (N,)
    labels = results["labels"]     # list of N strings

    print(f"\nGrounding DINO found {len(boxes)} UI elements:")
    print(f"{'#':<4} {'Label':<20} {'Score':>6} {'Box (x1,y1,x2,y2)'}")
    print("-" * 65)
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        x1, y1, x2, y2 = box.tolist()
        print(f"{i:<4} {label:<20} {score.item():>6.3f} ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")

    if len(boxes) == 0:
        print("\nNo UI elements detected. Try lowering --box-threshold or adjusting --prompt.")
        sys.exit(0)

    # --- Stage 2: SAM 2 mask refinement ---
    sam_masks = None
    if not args.no_sam:
        predictor = load_sam2(device)
        print("\nRunning SAM 2 mask refinement...")
        sam_masks = run_sam2(predictor, image_np, boxes, device)
        print(f"Generated {len(sam_masks)} masks.")

    base_name = os.path.splitext(os.path.basename(args.image))[0]

    # --- Output 1: Bounding boxes with labels ---
    fig, ax = plt.subplots(1, 1, figsize=(w / 80, h / 80), dpi=150)
    ax.imshow(image_np)
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        x1, y1, x2, y2 = box.tolist()
        color = random_color(i * 7 + 3)
        draw_box_label(ax, x1, y1, x2, y2, color, f"{i} {label} ({score.item():.2f})")
    ax.set_title(f"Detected UI elements: {len(boxes)}", fontsize=12)
    ax.axis("off")
    fig.tight_layout(pad=0.5)
    boxes_path = os.path.join(args.output_dir, f"{base_name}_boxes.png")
    fig.savefig(boxes_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved bounding boxes: {boxes_path}")

    # --- Output 2: Masks + boxes (if SAM 2 ran) ---
    if sam_masks is not None:
        fig, ax = plt.subplots(1, 1, figsize=(w / 80, h / 80), dpi=150)
        ax.imshow(image_np)
        for i, (box, score, label, mask) in enumerate(zip(boxes, scores, labels, sam_masks)):
            x1, y1, x2, y2 = box.tolist()
            color = random_color(i * 7 + 3)
            draw_mask_overlay(ax, mask, color)
            draw_box_label(ax, x1, y1, x2, y2, color, f"{i} {label}")
        ax.set_title(f"Detected UI elements: {len(boxes)}", fontsize=12)
        ax.axis("off")
        fig.tight_layout(pad=0.5)
        masks_path = os.path.join(args.output_dir, f"{base_name}_masks.png")
        fig.savefig(masks_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved masks + boxes: {masks_path}")

    # --- Output 3: Side-by-side comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(w / 40, h / 80), dpi=150)

    axes[0].imshow(image_np)
    axes[0].set_title("Original", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(image_np)
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        x1, y1, x2, y2 = box.tolist()
        color = random_color(i * 7 + 3)
        if sam_masks is not None:
            draw_mask_overlay(axes[1], sam_masks[i], color, alpha=0.3)
        draw_box_label(axes[1], x1, y1, x2, y2, color, f"{i} {label}")
    axes[1].set_title(f"UI elements: {len(boxes)}", fontsize=11)
    axes[1].axis("off")

    fig.tight_layout(pad=1.0)
    compare_path = os.path.join(args.output_dir, f"{base_name}_compare.png")
    fig.savefig(compare_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison: {compare_path}")

    print(f"\nDone! Found {len(boxes)} UI elements.")


if __name__ == "__main__":
    main()
