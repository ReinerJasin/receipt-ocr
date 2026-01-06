import json
from PIL import Image, ImageDraw
from typing import Dict, Optional
from pathlib import Path

def draw_layout_boxes(
    image_path: str,
    json_path: str,
    output_path: str,
    color_map: Optional[Dict[str, str]] = None,
    show: bool = True
):

    """
    Draw layout detection bounding boxes on an image.

    Args:
        image_path (str): Path to input image
        json_path (str): Path to layout JSON result
        output_path (str): Path to save output image
        color_map (dict, optional): Mapping label -> color
        show (bool): Whether to display the image
    """

    # default colors
    if color_map is None:
        color_map = {
            "text": "blue",
            "image": "green",
            "table": "red"
        }

    # Loading JSON File
    with open(json_path, "r") as f:
        data = json.load(f)

    # print("JSON Loaded successfuly!")
    # print(data)

    boxes = data["layout_det_res"]["boxes"]

    # print("Bounding boxes extracted successfully!")
    # print(boxes)

    # Loading Image
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    # Draw bounding boxes by the coordinate
    for box in boxes:
        x1, y1, x2, y2 = box["coordinate"]
        label = box["label"]
        score = box["score"]

        # choose color
        color = color_map.get(label, "red")

        # draw rectangle
        draw.rectangle(
            [(x1, y1), (x2, y2)],
            outline=color,
            width=2
        )

        # label text
        text = f"{label} ({score:.2f})"

        # draw label slightly above box
        text_x = x1
        text_y = max(0, y1 - 14)

        draw.text(
            (text_x, text_y),
            text,
            fill=color
        )

    # Save & Show
    image.save(output_path)
    
    if show:
        image.show()

    return output_path


def main():
    """
    Local testing entry point
    """
    
    FILENAME = Path("sample_receipt_20260106_135400.jpeg")
    IMAGE_PATH = Path(f"input/{FILENAME}")
    JSON_PATH = Path(f"output/json/{FILENAME.stem}_res.json")
    OUTPUT_PATH = Path(f"output/bbox/{FILENAME}")

    print("Running local visualization test...")

    draw_layout_boxes(
        image_path=IMAGE_PATH,
        json_path=JSON_PATH,
        output_path=OUTPUT_PATH,
        show=True
    )

    print(f"Saved result to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
