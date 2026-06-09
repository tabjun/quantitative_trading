"""Extract saved PNG outputs from a notebook into test/images.

This utility is intentionally narrow:
- parse notebook outputs only
- save `image/png` artifacts only
- do not generate markdown reports
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path


def extract_images_from_notebook(notebook_path: Path, output_dir: Path) -> int:
    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    base_name = notebook_path.stem
    extracted_count = 0

    for cell_idx, cell in enumerate(notebook.get("cells", []), start=1):
        if cell.get("cell_type") != "code":
            continue
        output_idx = 0
        for output in cell.get("outputs", []):
            data = output.get("data", {})
            if "image/png" not in data:
                continue
            output_idx += 1
            raw = data["image/png"]
            if isinstance(raw, list):
                raw = "".join(raw)
            raw = raw.replace("\n", "")
            image_bytes = base64.b64decode(raw)
            file_name = f"{base_name}_cell{cell_idx:03d}_{output_idx:02d}.png"
            (output_dir / file_name).write_bytes(image_bytes)
            extracted_count += 1
            print(f"Saved: {output_dir / file_name}")

    print(f"Extraction completed: {extracted_count} image(s)")
    return extracted_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract PNG outputs from notebook.")
    parser.add_argument(
        "notebook",
        nargs="?",
        default="test/models/2_time_series_advance_test.ipynb",
        help="Notebook path.",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="test/images",
        help="Output image directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extract_images_from_notebook(Path(args.notebook), Path(args.output_dir))


if __name__ == "__main__":
    main()
