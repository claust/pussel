"""Hand-label the puzzle's true corners in capture-dump photos.

The rectification metrics that matter most — quad IoU, corner error, and the
true aspect ratio the crop should have had — need a ground-truth quad. This
opens each image in a window, takes four clicks, and writes them to
`truth.json` in the dump directory, where `score_rectification.py` picks them
up automatically.

Usage (from the repo root):

    uv run python scripts/rectify_quality/label_quad.py <dump_dir>
    uv run python scripts/rectify_quality/label_quad.py <dump_dir> --all-frames
    uv run python scripts/rectify_quality/label_quad.py --image photo.jpg

Click the four corners of the puzzle picture (the artwork's rectangle, not the
cardboard's rounded outer edge) in order: top-left, top-right, bottom-right,
bottom-left. Keys: `u` undo the last point, `r` restart the image, `s` save and
advance (needs 4 points), `n` skip this image, `q` quit.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

Point = Tuple[float, float]

TRUTH_FILENAME = "truth.json"
WINDOW_NAME = "label puzzle corners"
DISPLAY_MAX_DIM = 1100
POINT_COLOR = (60, 200, 110)
LABELS = ("top-left", "top-right", "bottom-right", "bottom-left")


def load_display(path: Path) -> Tuple["np.ndarray", float]:
    """Load an image EXIF-corrected and downscaled for on-screen labelling.

    Args:
        path: The image file.

    Returns:
        A (BGR display image, scale) tuple, where scale maps display pixels
        back to original pixels.
    """
    with Image.open(path) as handle:
        rgb = np.asarray(ImageOps.exif_transpose(handle).convert("RGB"))
    height, width = rgb.shape[:2]
    scale = min(1.0, DISPLAY_MAX_DIM / max(width, height))
    if scale < 1.0:
        rgb = cv2.resize(rgb, (round(width * scale), round(height * scale)))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), scale


def render(base: "np.ndarray", points: List[Tuple[int, int]], name: str) -> "np.ndarray":
    """Draw the points collected so far plus the next-corner prompt.

    Args:
        base: The BGR display image.
        points: Display-space points clicked so far.
        name: The image's filename, shown in the prompt.

    Returns:
        An annotated copy.
    """
    canvas = base.copy()
    for index, point in enumerate(points):
        cv2.circle(canvas, point, 7, POINT_COLOR, -1)
        cv2.putText(
            canvas, str(index + 1), (point[0] + 10, point[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, POINT_COLOR, 2
        )
    if len(points) > 1:
        closed = len(points) == 4
        cv2.polylines(canvas, [np.array(points, dtype=np.int32)], isClosed=closed, color=POINT_COLOR, thickness=2)
    prompt = f"{name}: click {LABELS[len(points)]}" if len(points) < 4 else f"{name}: 's' save  'u' undo  'r' restart"
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(canvas, prompt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
    return canvas


def label_image(path: Path) -> Optional[List[Point]]:
    """Collect four corners for one image.

    Args:
        path: The image to label.

    Returns:
        Four normalized (x, y) corners, or None when the image was skipped or
        labelling was aborted.
    """
    base, _ = load_display(path)
    height, width = base.shape[:2]
    points: List[Tuple[int, int]] = []

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    try:
        while True:
            cv2.imshow(WINDOW_NAME, render(base, points, path.name))
            key = cv2.waitKey(20) & 0xFF
            if key == ord("q"):
                raise KeyboardInterrupt
            if key == ord("n"):
                return None
            if key == ord("u") and points:
                points.pop()
            if key == ord("r"):
                points.clear()
            if key == ord("s") and len(points) == 4:
                return [(x / width, y / height) for x, y in points]
    finally:
        cv2.destroyWindow(WINDOW_NAME)


def resolve_targets(arguments: argparse.Namespace) -> Tuple[List[Path], Path]:
    """Work out which images to label and where their labels belong.

    Args:
        arguments: The parsed command line.

    Returns:
        A (images, truth_path) tuple; images is empty when the dump directory
        holds none.
    """
    if arguments.image is not None:
        return [arguments.image], arguments.image.parent / TRUTH_FILENAME
    names = ["composite.jpg"]
    if arguments.all_frames:
        names.append("reference.jpg")
        names.extend(f"corner_{index}.jpg" for index in range(1, 5))
    images = [arguments.dump_dir / name for name in names if (arguments.dump_dir / name).exists()]
    return images, arguments.dump_dir / TRUTH_FILENAME


def load_payload(truth_path: Path) -> Tuple[Dict[str, object], Dict[str, object]]:
    """Read the existing labels file, or start a fresh one.

    Args:
        truth_path: Where labels are stored.

    Returns:
        A (payload, images-section) tuple; the second is a live reference into
        the first, so writing to it and re-serializing the payload persists it.
    """
    payload: Dict[str, object] = {"images": {}}
    if truth_path.exists():
        payload = json.loads(truth_path.read_text())
        payload.setdefault("images", {})
    existing = payload["images"]
    if not isinstance(existing, dict):
        raise ValueError(f"{truth_path} has a malformed 'images' section")
    return payload, existing


def main() -> int:
    """Entry point.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dump_dir", nargs="?", type=Path, help="Capture dump directory")
    parser.add_argument("--image", type=Path, help="Label a single loose image")
    parser.add_argument("--all-frames", action="store_true", help="Also label reference.jpg and corner_1..4.jpg")
    parser.add_argument("--relabel", action="store_true", help="Re-label images that already have a saved quad")
    arguments = parser.parse_args()
    if arguments.image is None and arguments.dump_dir is None:
        parser.error("give a dump directory or --image")

    images, truth_path = resolve_targets(arguments)
    if not images:
        print(f"No images found in {arguments.dump_dir}", file=sys.stderr)
        return 1
    payload, existing = load_payload(truth_path)

    try:
        for path in images:
            if path.name in existing and not arguments.relabel:
                print(f"{path.name}: already labelled (pass --relabel to redo)")
                continue
            corners = label_image(path)
            if corners is None:
                print(f"{path.name}: skipped")
                continue
            existing[path.name] = {"corners": [[round(x, 6), round(y, 6)] for x, y in corners]}
            truth_path.write_text(json.dumps(payload, indent=2) + "\n")
            print(f"{path.name}: saved")
    except KeyboardInterrupt:
        print("\nAborted; labels saved so far are kept.")
        return 0
    print(f"\nLabels in {truth_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
