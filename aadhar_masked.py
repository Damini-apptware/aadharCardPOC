#!/usr/bin/env python3
"""
Mask first 8 digits of every Aadhaar number in ALL images of a folder.
Also produce run_log.txt that summarises how many images were masked and how many
images contained no Aadhaar UID.

Usage:
    python3 mask_folder.py /path/to/images_folder
"""

import os
import re
import argparse
from datetime import datetime

import cv2
import pytesseract
from tqdm import tqdm

# -----------------------------------------------------------------------------
# 0) One‑time tweak (only needed if the Tesseract executable is **not** in PATH)
#     pytesseract.pytesseract.tesseract_cmd = r"/usr/local/bin/tesseract"
# -----------------------------------------------------------------------------

AADHAAR_RE = re.compile(r"\b(\d{4})\s(\d{4})\s(\d{4})\b")  # 1234 5678 9012

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def detect_aadhaar(text_blocks: list[str]):
    """Return list of (aadhaar_string, [indices_of_each_group])."""
    full_text = " ".join(text_blocks)
    matches = AADHAAR_RE.findall(full_text)
    results = []
    if matches:
        # Build word -> list[index] mapping once
        positions: dict[str, list[int]] = {}
        for idx, word in enumerate(text_blocks):
            positions.setdefault(word, []).append(idx)

        for m in matches:  # m = ('1234', '5678', '9012')
            groups = list(m)
            try:
                idxs = [positions[g].pop(0) for g in groups]
                results.append((" ".join(groups), idxs))
            except (KeyError, IndexError):
                # OCR mismatch, skip
                continue
    return results


def mask_block(img, bbox, label="XXXXXXXX"):
    """Draw white rectangle over bbox and write *label* centred inside."""
    x, y, w, h = bbox
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    thickness = 2
    (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
    cv2.putText(
        img,
        label,
        (x + (w - tw) // 2, y + (h + th) // 2),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )


# -----------------------------------------------------------------------------
# Core image‑level work
# -----------------------------------------------------------------------------

def process_one_image(in_path: str, out_path: str):
    """Process a single image and return (log_line, status).

    status ∈ {"masked", "not_found", "error"}
    """
    image = cv2.imread(in_path)
    if image is None:
        return f"{os.path.basename(in_path)} : ❌ could not load", "error"

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    data = pytesseract.image_to_data(rgb, output_type=pytesseract.Output.DICT)
    texts = data["text"]

    found_any = False
    aadhaar_last4 = ""  # for logging once per image

    for aadhaar, idxs in detect_aadhaar(texts):
        i1, i2 = idxs[:2]
        # bounding boxes of first two groups (8 digits)
        x1, y1, w1, h1 = (
            data["left"][i1],
            data["top"][i1],
            data["width"][i1],
            data["height"][i1],
        )
        x2, y2, w2, h2 = (
            data["left"][i2],
            data["top"][i2],
            data["width"][i2],
            data["height"][i2],
        )

        # combine two boxes so one rectangle covers the first 8 digits
        x = min(x1, x2)
        y = min(y1, y2)
        w = max(x1 + w1, x2 + w2) - x
        h = max(h1, h2)

        mask_block(image, (x, y, w, h), "XXXXXXXX")
        found_any = True
        aadhaar_last4 = aadhaar[-4:]

    if found_any:
        cv2.imwrite(out_path, image)
        return (
            f"{os.path.basename(in_path)} : ✅ masked …{aadhaar_last4}",
            "masked",
        )
    else:
        return (
            f"{os.path.basename(in_path)} : ⚠️  Aadhaar UID not found",
            "not_found",
        )


# -----------------------------------------------------------------------------
# Folder‑level driver
# -----------------------------------------------------------------------------

def iterate_folder(folder: str):
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    out_dir = os.path.join(folder, "masked")
    os.makedirs(out_dir, exist_ok=True)

    images = [f for f in sorted(os.listdir(folder)) if f.lower().endswith(valid_exts)]
    logs: list[str] = []

    # Counters
    masked_count = 0
    not_found_count = 0
    error_count = 0

    for fname in tqdm(images, desc="Processing"):
        in_path = os.path.join(folder, fname)
        out_path = os.path.join(out_dir, fname)
        log_line, status = process_one_image(in_path, out_path)
        logs.append(log_line)

        if status == "masked":
            masked_count += 1
        elif status == "not_found":
            not_found_count += 1
        else:
            error_count += 1

    # ---------------------------------------------------------------------
    # Write summary + per‑file log to run_log.txt
    # ---------------------------------------------------------------------
    with open(os.path.join(out_dir, "run_log.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"Run at {datetime.now()}\n")
        fh.write(f"Total images processed : {len(images)}\n")
        fh.write(f"Masked images          : {masked_count}\n")
        fh.write(f"Aadhaar UID not found  : {not_found_count}\n")
        fh.write(f"Images failed to load  : {error_count}\n\n")
        fh.write("=== Individual file log ===\n")
        fh.write("\n".join(logs))

    print(
        f"\nDone! {masked_count} masked, {not_found_count} without UID. "
        f"See {out_dir}/run_log.txt for details."
    )


# -----------------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mask Aadhaar numbers in every image of a folder and report summary counts"
    )
    parser.add_argument("folder", help="Folder containing images")
    args = parser.parse_args()
    iterate_folder(os.path.abspath(args.folder))

