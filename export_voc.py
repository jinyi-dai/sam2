"""
Export Grounding DINO detection boxes as Pascal VOC XML for use with labelImg.

Usage:
  python export_voc.py --image my_screenshots/game.png
  python export_voc.py --image my_screenshots/game.png --box-threshold 0.09 --text-threshold 0.09
  python export_voc.py --image my_screenshots/game.png --output-dir labeled/

The XML is saved next to the image by default (labelImg expects this).
Open labelImg → "Open Dir" → select the image folder to load and correct boxes.
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export Grounding DINO detections as Pascal VOC XML for labelImg"
    )
    parser.add_argument("--image", required=True, help="Path to the screenshot")
    parser.add_argument("--prompt", default=None,
                        help="Grounding DINO prompt (default: detect_ui default)")
    parser.add_argument("--box-threshold", type=float, default=0.09)
    parser.add_argument("--text-threshold", type=float, default=0.09)
    parser.add_argument("--output-dir", default=None,
                        help="Directory for XML output (default: same directory as image)")
    return parser.parse_args()


def build_voc_xml(image_path, image_w, image_h, boxes, scores, labels):
    """Build a Pascal VOC XML ElementTree from detection results."""
    annotation = ET.Element("annotation")

    folder = ET.SubElement(annotation, "folder")
    folder.text = os.path.basename(os.path.dirname(os.path.abspath(image_path)))

    filename = ET.SubElement(annotation, "filename")
    filename.text = os.path.basename(image_path)

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(image_w)
    ET.SubElement(size, "height").text = str(image_h)
    ET.SubElement(size, "depth").text = "3"

    for i in range(len(labels)):
        label = labels[i].strip() if labels[i] else "unknown"
        if not label:
            label = "unknown"
        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = label

        bndbox = ET.SubElement(obj, "bndbox")
        x1, y1, x2, y2 = boxes[i]
        ET.SubElement(bndbox, "xmin").text = str(max(0, int(round(x1))))
        ET.SubElement(bndbox, "ymin").text = str(max(0, int(round(y1))))
        ET.SubElement(bndbox, "xmax").text = str(min(image_w, int(round(x2))))
        ET.SubElement(bndbox, "ymax").text = str(min(image_h, int(round(y2))))

    return ET.ElementTree(annotation)


def main():
    args = parse_args()

    if not os.path.isfile(args.image):
        print(f"Error: image not found: {args.image}")
        sys.exit(1)

    from detect_ui import (
        setup_device, load_grounding_dino, run_grounding_dino, DEFAULT_PROMPT,
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
    print(f"Thresholds: box={args.box_threshold}, text={args.text_threshold}")

    results = run_grounding_dino(
        pil_image, prompt, processor, gdino_model, device,
        args.box_threshold, args.text_threshold,
    )

    boxes = results["boxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()
    labels = results["labels"]

    print(f"Detected {len(labels)} elements")

    if len(labels) == 0:
        print("No detections. Try lowering thresholds.")
        sys.exit(0)

    tree = build_voc_xml(args.image, w, h, boxes, scores, labels)

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.image))
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.image))[0]
    xml_path = os.path.join(output_dir, f"{base}.xml")

    ET.indent(tree, space="  ")
    tree.write(xml_path, encoding="unicode", xml_declaration=True)
    print(f"Saved: {xml_path}")


if __name__ == "__main__":
    main()
