"""
Command-line usage:
    python cli.py path/to/photo.jpg

Prints the inside/outside verdict for each detected person and saves an
annotated copy of the image (with boxes drawn) next to a name you choose.
"""

import argparse
import cv2
from detector import analyze_image, summarize


def main():
    parser = argparse.ArgumentParser(
        description="Detect whether people in a photo are inside or outside a car."
    )
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument(
        "-o", "--output", default="annotated_output.jpg",
        help="Where to save the annotated image (default: annotated_output.jpg)"
    )
    parser.add_argument(
        "-c", "--conf", type=float, default=0.35,
        help="Detection confidence threshold (default: 0.35)"
    )
    args = parser.parse_args()

    result = analyze_image(args.image, conf_threshold=args.conf)
    print(summarize(result["people"]))

    cv2.imwrite(args.output, result["annotated_image"])
    print(f"\nAnnotated image saved to: {args.output}")


if __name__ == "__main__":
    main()