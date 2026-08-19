import argparse
import json
import os
import cv2


def draw_bounding_boxes(
    json_path: str,
    img_dir: str,
    output_dir: str,
    box_color: tuple = (0, 0, 255),
    thickness: int = 3,
):
    """Parses JSON annotations and draws bounding boxes on corresponding page images.

    Args:
        json_path (str): Path to the input JSON annotation file.
        img_dir (str): Base directory containing the images.
        output_dir (str): Directory where annotated images will be saved.
        box_color (tuple): BGR color tuple for bounding boxes (default is Red).
        thickness (int): Thickness of bounding box borders.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    os.makedirs(output_dir, exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pages = data.get("pages", [])
    if not pages:
        print("No 'pages' found in JSON annotation file.")
        return

    for page in pages:
        rel_img_path = page.get("image_path")
        if not rel_img_path:
            continue

        # Build complete path to image
        full_img_path = os.path.join(img_dir, rel_img_path)

        if not os.path.exists(full_img_path):
            print(f"Warning: Image file not found at {full_img_path}, skipping.")
            continue

        # Load image
        img = cv2.imread(full_img_path)
        if img is None:
            print(f"Failed to load image: {full_img_path}")
            continue

        elements = page.get("elements", [])
        for elem in elements:
            bbox = elem.get("bbox")
            elem_type = elem.get("type", "unknown")

            # Check for bounding box coordinates [xmin, ymin, xmax, ymax]
            if bbox and len(bbox) == 4:
                xmin, ymin, xmax, ymax = map(int, bbox)

                # Draw bounding box
                cv2.rectangle(
                    img, (xmin, ymin), (xmax, ymax), box_color, thickness
                )

                # Draw label text above the box
                label = f"{elem_type}"
                font_scale = 0.8
                text_thickness = 2
                (w, h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness
                )

                # Draw background rectangle for readability
                cv2.rectangle(
                    img,
                    (xmin, max(0, ymin - h - 10)),
                    (xmin + w, max(0, ymin)),
                    box_color,
                    -1,
                )
                cv2.putText(
                    img,
                    label,
                    (xmin, max(15, ymin - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    text_thickness,
                    cv2.LINE_AA,
                )

        # Save annotated image
        output_filename = f"annotated_{os.path.basename(rel_img_path)}"
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, img)
        print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize JSON Bounding Boxes on Dataset Images"
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to the target JSON annotation file.",
    )
    parser.add_argument(
        "--img-dir",
        "-d",
        default=".",
        help="Root path containing image directory (default: current directory).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="output_visualizations",
        help="Path to directory where output images will be saved.",
    )

    args = parser.parse_args()
    draw_bounding_boxes(args.input, args.img_dir, args.output_dir)


if __name__ == "__main__":
    main()