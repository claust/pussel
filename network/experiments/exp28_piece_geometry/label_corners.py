#!/usr/bin/env python3
"""M2 hand-labeling tool: click the 4 true corners of real piece photos.

Interactive matplotlib tool. Shows each piece crop (with its rembg contour
overlaid when available) and waits for 4 mouse clicks marking the corners.
Autosaves to `--labels-file` after every piece, so progress is never lost.

Keys:
    u - undo the last click
    r - redo this piece (clear all 4 clicks and start over)
    s - skip this piece without labeling it
    q - quit and save

Usage:
    cd network
    uv run python experiments/exp28_piece_geometry/label_corners.py --limit 20
    uv run python experiments/exp28_piece_geometry/label_corners.py --puzzle bambi --background red_carpet
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
from common import PieceRecord, crop_with_margin, load_metadata

DEFAULT_DATASET_ROOT = Path("/Users/claus/Repos/pussel/network/datasets/north_star/v1")
DEFAULT_LABELS_FILE = Path(__file__).parent / "outputs" / "corner_labels.json"
DEFAULT_MARGIN_FRAC = 0.15


def _load_labels(labels_file: Path) -> Dict[str, Any]:
    """Load existing labels, if any.

    Args:
        labels_file: Path to the labels JSON file.

    Returns:
        The labels dict, keyed by piece_file. Empty dict if the file is absent.
    """
    if labels_file.exists():
        with open(labels_file, encoding="utf-8") as handle:
            return json.load(handle)
    return {}


def _save_labels(labels_file: Path, labels: Dict[str, Any]) -> None:
    """Persist labels to disk.

    Args:
        labels_file: Path to the labels JSON file.
        labels: The labels dict to save.
    """
    labels_file.parent.mkdir(parents=True, exist_ok=True)
    with open(labels_file, "w", encoding="utf-8") as handle:
        json.dump(labels, handle, indent=2)


def _load_contour(contours_dir: Optional[Path], record: PieceRecord) -> Optional[np.ndarray]:
    """Load a piece's saved rembg contour, if available.

    Args:
        contours_dir: The `outputs/contours` directory, or None to skip.
        record: The piece's metadata row.

    Returns:
        Nx2 contour array in original image coordinates, or None.
    """
    if contours_dir is None:
        return None
    path = contours_dir / record.puzzle_id / f"{record.piece_stem}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    contour = data["methods"].get("rembg", {}).get("contour")
    return np.array(contour) if contour else None


class LabelSession:
    """Drives one matplotlib figure through a sequence of pieces to label."""

    def __init__(
        self,
        dataset_root: Path,
        contours_dir: Optional[Path],
        records: List[PieceRecord],
        labels: Dict[str, Any],
        labels_file: Path,
    ) -> None:
        """Set up the labeling session.

        Args:
            dataset_root: The north_star dataset root.
            contours_dir: Optional `outputs/contours` directory for contour overlays.
            records: Pieces to label, in order.
            labels: The in-memory labels dict (mutated and autosaved as we go).
            labels_file: Where to autosave `labels`.
        """
        self.dataset_root = dataset_root
        self.contours_dir = contours_dir
        self.records = records
        self.labels = labels
        self.labels_file = labels_file
        self.index = 0
        self.clicks: List[List[float]] = []
        self.quit_requested = False

        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def run(self) -> None:
        """Run the labeling loop until all pieces are done or the user quits."""
        while self.index < len(self.records) and not self.quit_requested:
            self._render_current()
            plt.waitforbuttonpress()
            while self.fig.canvas.manager is not None and not self.quit_requested and len(self.clicks) < 4:
                if not plt.fignum_exists(self.fig.number):
                    self.quit_requested = True
                    break
                plt.waitforbuttonpress()
        plt.close(self.fig)

    def _render_current(self) -> None:
        """Draw the current piece crop, contour overlay, and any clicks so far."""
        record = self.records[self.index]
        image = cv2.imread(str(self.dataset_root / record.piece_file))
        crop, offset = crop_with_margin(image, record.bbox, margin_frac=DEFAULT_MARGIN_FRAC)
        self._offset = offset
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        self.ax.clear()
        self.ax.imshow(crop_rgb)
        self.ax.set_title(
            f"[{self.index + 1}/{len(self.records)}] {record.puzzle_id} r{record.row}c{record.col} "
            f"{record.background} -- click 4 corners (u=undo r=redo s=skip q=quit)"
        )

        contour = _load_contour(self.contours_dir, record)
        if contour is not None:
            local = contour - np.array(offset)
            self.ax.plot(local[:, 0], local[:, 1], color="cyan", linewidth=1, alpha=0.7)

        for x, y in self.clicks:
            self.ax.plot(x, y, "o", color="lime", markersize=8)

        self.fig.canvas.draw_idle()

    def _on_click(self, event: Any) -> None:
        """Record a corner click.

        Args:
            event: Matplotlib mouse event.
        """
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        self.clicks.append([event.xdata, event.ydata])
        self._render_current()
        if len(self.clicks) == 4:
            self._commit_piece()

    def _on_key(self, event: Any) -> None:
        """Handle keyboard shortcuts (undo/redo/skip/quit).

        Args:
            event: Matplotlib key event.
        """
        if event.key == "u" and self.clicks:
            self.clicks.pop()
            self._render_current()
        elif event.key == "r":
            self.clicks = []
            self._render_current()
        elif event.key == "s":
            self.clicks = []
            self.index += 1
            if self.index < len(self.records):
                self._render_current()
            else:
                self.quit_requested = True
        elif event.key == "q":
            self.quit_requested = True

    def _commit_piece(self) -> None:
        """Convert the 4 clicks to original-image coordinates, save, and advance."""
        record = self.records[self.index]
        offset_x, offset_y = self._offset
        original_coords = [[x + offset_x, y + offset_y] for x, y in self.clicks]
        self.labels[record.piece_file] = {"corners": original_coords, "background": record.background}
        _save_labels(self.labels_file, self.labels)
        print(f"Saved {record.piece_file} ({self.index + 1}/{len(self.records)})")

        self.clicks = []
        self.index += 1
        if self.index < len(self.records):
            self._render_current()
        else:
            self.quit_requested = True


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Hand-label the 4 true corners of north_star piece photos.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--contours-dir", type=Path, default=Path(__file__).parent / "outputs" / "contours")
    parser.add_argument("--labels-file", type=Path, default=DEFAULT_LABELS_FILE)
    parser.add_argument("--puzzle", type=str, default=None, help="Substring filter on puzzle_id")
    parser.add_argument("--background", type=str, default=None, help="Exact filter on background")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only-unlabeled", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    records = load_metadata(args.dataset_root)
    if args.puzzle:
        records = [r for r in records if args.puzzle in r.puzzle_id]
    if args.background:
        records = [r for r in records if r.background == args.background]

    labels = _load_labels(args.labels_file)
    if args.only_unlabeled:
        records = [r for r in records if r.piece_file not in labels]

    if args.limit is not None:
        records = records[: args.limit]

    if not records:
        print("Nothing to label (no pieces matched the filters, or all are already labeled).")
        return

    contours_dir = args.contours_dir if args.contours_dir.exists() else None
    session = LabelSession(args.dataset_root, contours_dir, records, labels, args.labels_file)
    session.run()
    print(f"\nLabeled {len(session.labels)} pieces total. Saved to {args.labels_file}")


if __name__ == "__main__":
    main()
