"""
Post-process UI detection results into a hierarchical structure.

Pipeline:
  1. Text fragment merging (fix split text detections)
  4. Containment tree construction (parent-child hierarchy)
  7. JSON export with percentage-based coordinates

Usage:
  python post_process.py --image my_screenshots/game.png
  python post_process.py --image my_screenshots/game.png --no-sam --no-viz
  python post_process.py --image my_screenshots/game.png --overlap-threshold 0.80

As a module:
  from post_process import run_pipeline, export_json
  elements, roots = run_pipeline(boxes_np, scores_np, labels, masks, (h, w))
  json_str = export_json(roots, (h, w), "output/hierarchy.json")
"""

import argparse
import dataclasses
import json
import os
import sys
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class UIElement:
    id: int
    label: str
    box: np.ndarray                   # (4,) x1, y1, x2, y2 in pixels
    score: float
    mask: Optional[np.ndarray] = None  # boolean mask or None
    children: list["UIElement"] = dataclasses.field(default_factory=list)
    parent: Optional["UIElement"] = None


def elements_from_detections(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: list[str],
    masks: Optional[list[np.ndarray]] = None,
) -> list[UIElement]:
    elements = []
    for i in range(len(labels)):
        elements.append(UIElement(
            id=i,
            label=labels[i],
            box=boxes[i].copy(),
            score=float(scores[i]),
            mask=masks[i] if masks is not None else None,
        ))
    return elements


def reassign_ids(elements: list[UIElement]) -> None:
    for i, e in enumerate(elements):
        e.id = i


# ---------------------------------------------------------------------------
# Step 1: Text fragment merging
# ---------------------------------------------------------------------------

def _is_text_type(label: str) -> bool:
    return "text" in label.lower()


def _should_merge_horizontal(a: UIElement, b: UIElement,
                              gap_ratio: float, align_ratio: float,
                              height_ratio: float) -> bool:
    # ensure a is left of b
    if a.box[0] > b.box[0]:
        a, b = b, a
    gap = b.box[0] - a.box[2]
    if gap < 0:
        return False
    w_a = a.box[2] - a.box[0]
    w_b = b.box[2] - b.box[0]
    if gap > gap_ratio * min(w_a, w_b):
        return False
    h_a = a.box[3] - a.box[1]
    h_b = b.box[3] - b.box[1]
    if abs(a.box[1] - b.box[1]) > align_ratio * max(h_a, h_b):
        return False
    if abs(h_a - h_b) > height_ratio * max(h_a, h_b):
        return False
    return True


def merge_text_fragments(
    elements: list[UIElement],
    gap_ratio: float = 0.1,
    align_ratio: float = 0.2,
    height_ratio: float = 0.2,
) -> list[UIElement]:
    while True:
        merged_any = False
        text_indices = [i for i, e in enumerate(elements) if _is_text_type(e.label)]
        for ii in range(len(text_indices)):
            if merged_any:
                break
            for jj in range(ii + 1, len(text_indices)):
                a = elements[text_indices[ii]]
                b = elements[text_indices[jj]]
                should = _should_merge_horizontal(a, b, gap_ratio, align_ratio, height_ratio)
                if should:
                    merged_box = np.array([
                        min(a.box[0], b.box[0]),
                        min(a.box[1], b.box[1]),
                        max(a.box[2], b.box[2]),
                        max(a.box[3], b.box[3]),
                    ])
                    merged = UIElement(
                        id=a.id,
                        label=a.label,
                        box=merged_box,
                        score=max(a.score, b.score),
                        mask=None,
                    )
                    # remove both, add merged
                    to_remove = {text_indices[ii], text_indices[jj]}
                    elements = [e for idx, e in enumerate(elements) if idx not in to_remove]
                    elements.append(merged)
                    merged_any = True
                    break
        if not merged_any:
            break
    return elements


# ---------------------------------------------------------------------------
# Step 4: Containment tree construction
# ---------------------------------------------------------------------------

def _box_area(box: np.ndarray) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _intersection_area(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def build_containment_tree(
    elements: list[UIElement],
    overlap_threshold: float = 0.85,
    area_ratio: float = 1.5,
) -> list[UIElement]:
    # reset parent/children
    for e in elements:
        e.children = []
        e.parent = None

    # sort by area descending
    areas = {id(e): _box_area(e.box) for e in elements}
    sorted_elems = sorted(elements, key=lambda e: areas[id(e)], reverse=True)

    # for each element (smallest first), find smallest qualifying parent
    for b in reversed(sorted_elems):
        b_area = areas[id(b)]
        if b_area == 0:
            continue
        best_parent = None
        best_parent_area = float("inf")
        for a in sorted_elems:
            if a is b:
                continue
            a_area = areas[id(a)]
            if a_area < area_ratio * b_area:
                continue
            overlap = _intersection_area(a.box, b.box)
            if overlap >= overlap_threshold * b_area:
                if a_area < best_parent_area:
                    best_parent = a
                    best_parent_area = a_area
        if best_parent is not None:
            best_parent.children.append(b)
            b.parent = best_parent

    return [e for e in elements if e.parent is None]


# ---------------------------------------------------------------------------
# Step 7: JSON export
# ---------------------------------------------------------------------------

def element_to_dict(element: UIElement, image_w: int, image_h: int) -> dict:
    return {
        "id": f"node_{element.id}",
        "type": element.label,
        "confidence": round(float(element.score), 4),
        "bounds_pct": {
            "left": round(float(100.0 * element.box[0] / image_w), 1),
            "top": round(float(100.0 * element.box[1] / image_h), 1),
            "width": round(float(100.0 * (element.box[2] - element.box[0]) / image_w), 1),
            "height": round(float(100.0 * (element.box[3] - element.box[1]) / image_h), 1),
        },
        "children": [element_to_dict(c, image_w, image_h) for c in element.children],
    }


def export_json(
    roots: list[UIElement],
    image_hw: tuple[int, int],
    output_path: Optional[str] = None,
) -> str:
    h, w = image_hw
    tree = [element_to_dict(r, w, h) for r in roots]
    json_str = json.dumps(tree, indent=2)
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            f.write(json_str)
    return json_str


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _random_color(seed):
    rng = np.random.RandomState(seed)
    return rng.uniform(0.3, 0.95, size=3)


def draw_raw_boxes(
    image_np: np.ndarray,
    elements: list[UIElement],
    output_path: str,
) -> None:
    """Draw all detection boxes before any post-processing."""
    h, w = image_np.shape[:2]
    fig, ax = plt.subplots(1, 1, figsize=(w / 80, h / 80), dpi=150)
    ax.imshow(image_np)
    for e in elements:
        x1, y1, x2, y2 = e.box
        color = _random_color(e.id * 7 + 3)
        rect = plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            x1, y1 - 4,
            f"{e.id} {e.label} ({e.score:.2f})",
            fontsize=7, color="white", fontweight="bold",
            bbox=dict(facecolor=color, alpha=0.85, pad=1.5, edgecolor="none"),
        )
    ax.set_title(f"Raw detections: {len(elements)}", fontsize=12)
    ax.axis("off")
    fig.tight_layout(pad=0.5)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


DEPTH_COLORS = [
    (0.9, 0.2, 0.2),   # red
    (0.2, 0.8, 0.2),   # green
    (0.2, 0.4, 0.9),   # blue
    (0.9, 0.7, 0.1),   # yellow
    (0.8, 0.2, 0.8),   # magenta
    (0.1, 0.8, 0.8),   # cyan
]


def _draw_tree_boxes(ax, element: UIElement, depth: int = 0):
    color = DEPTH_COLORS[depth % len(DEPTH_COLORS)]
    x1, y1, x2, y2 = element.box
    rect = plt.Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=max(1, 2 - depth * 0.3),
        edgecolor=color, facecolor="none",
    )
    ax.add_patch(rect)
    ax.text(
        x1, y1 - 3,
        f"{element.id} {element.label} d{depth}",
        fontsize=6, color="white", fontweight="bold",
        bbox=dict(facecolor=color, alpha=0.85, pad=1, edgecolor="none"),
    )
    for child in element.children:
        _draw_tree_boxes(ax, child, depth + 1)


def draw_hierarchy(
    image_np: np.ndarray,
    roots: list[UIElement],
    output_path: str,
) -> None:
    h, w = image_np.shape[:2]
    fig, ax = plt.subplots(1, 1, figsize=(w / 80, h / 80), dpi=150)
    ax.imshow(image_np)
    for root in roots:
        _draw_tree_boxes(ax, root, depth=0)
    ax.set_title(f"UI Hierarchy ({_count_nodes(roots)} elements)", fontsize=12)
    ax.axis("off")
    fig.tight_layout(pad=0.5)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _count_nodes(roots: list[UIElement]) -> int:
    count = 0
    for r in roots:
        count += 1 + _count_nodes(r.children)
    return count


def print_tree(roots: list[UIElement], image_hw: tuple[int, int], indent: int = 0) -> None:
    h, w = image_hw
    for r in roots:
        prefix = "  " * indent
        left = 100.0 * r.box[0] / w
        top = 100.0 * r.box[1] / h
        bw = 100.0 * (r.box[2] - r.box[0]) / w
        bh = 100.0 * (r.box[3] - r.box[1]) / h
        print(f"{prefix}node_{r.id} [{r.label}] ({left:.1f}%, {top:.1f}%) {bw:.1f}x{bh:.1f}%  conf={r.score:.3f}")
        print_tree(r.children, image_hw, indent + 1)


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

def run_pipeline(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: list[str],
    masks: Optional[list[np.ndarray]],
    image_hw: tuple[int, int],
    overlap_threshold: float = 0.85,
) -> tuple[list[UIElement], list[UIElement], list[UIElement]]:
    """Returns (raw_elements, all_elements, root_nodes)."""
    elements = elements_from_detections(boxes, scores, labels, masks)
    raw_elements = list(elements)  # snapshot before processing
    print(f"  Input: {len(elements)} elements")

    elements = merge_text_fragments(elements)
    print(f"  After text merge: {len(elements)} elements")

    reassign_ids(elements)
    roots = build_containment_tree(elements, overlap_threshold=overlap_threshold)
    print(f"  Tree: {len(roots)} root nodes, {len(elements)} total")

    return raw_elements, elements, roots


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Post-process UI detections into a hierarchical structure"
    )
    parser.add_argument("--image", required=True, help="Path to the screenshot")
    parser.add_argument("--prompt", default=None,
                        help="Grounding DINO prompt (default: detect_ui default)")
    parser.add_argument("--box-threshold", type=float, default=0.075)
    parser.add_argument("--text-threshold", type=float, default=0.1)
    parser.add_argument("--output-dir", default="./output")
    parser.add_argument("--no-sam", action="store_true",
                        help="Skip SAM 2 mask refinement")
    parser.add_argument("--overlap-threshold", type=float, default=0.85,
                        help="Containment overlap threshold (default: 0.85)")
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip visualization output")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.image):
        print(f"Error: image not found: {args.image}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Run detection (import from detect_ui) ---
    from detect_ui import (
        setup_device, load_grounding_dino, run_grounding_dino,
        load_sam2, run_sam2, DEFAULT_PROMPT,
    )

    device = setup_device()
    pil_image = Image.open(args.image).convert("RGB")
    image_np = np.array(pil_image)
    h, w = image_np.shape[:2]
    print(f"Image: {args.image} ({w}x{h})")

    processor, gdino_model = load_grounding_dino(device)
    prompt = (args.prompt or DEFAULT_PROMPT).lower()
    if not prompt.endswith("."):
        prompt += "."
    print(f"Prompt: \"{prompt}\"")

    results = run_grounding_dino(
        pil_image, prompt, processor, gdino_model, device,
        args.box_threshold, args.text_threshold,
    )

    boxes_tensor = results["boxes"]
    scores_np = results["scores"].cpu().numpy()
    labels = results["labels"]
    boxes_np = boxes_tensor.cpu().numpy()

    print(f"Grounding DINO: {len(labels)} detections")

    masks = None
    if not args.no_sam:
        predictor = load_sam2(device)
        masks = run_sam2(predictor, image_np, boxes_tensor, device)
        print(f"SAM 2: {len(masks)} masks")

    # --- Post-process ---
    print("\nPost-processing:")
    raw_elements, elements, roots = run_pipeline(
        boxes_np, scores_np, labels, masks, (h, w),
        overlap_threshold=args.overlap_threshold,
    )

    # --- Export ---
    base = os.path.splitext(os.path.basename(args.image))[0]

    json_path = os.path.join(args.output_dir, f"{base}_hierarchy.json")
    export_json(roots, (h, w), json_path)
    print(f"\nSaved: {json_path}")

    if not args.no_viz:
        raw_path = os.path.join(args.output_dir, f"{base}_raw_boxes.png")
        draw_raw_boxes(image_np, raw_elements, raw_path)
        print(f"Saved: {raw_path}")

        viz_path = os.path.join(args.output_dir, f"{base}_hierarchy.png")
        draw_hierarchy(image_np, roots, viz_path)
        print(f"Saved: {viz_path}")

    print("\nHierarchy:")
    print_tree(roots, (h, w))


if __name__ == "__main__":
    main()
