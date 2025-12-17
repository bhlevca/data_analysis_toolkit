"""
Simple synthetic digit image generator for the Data Analysis Toolkit.

Generates PNG images of digits (0-9) with random rotations, scale, position,
background/noise and saves them into a specified output directory.

Usage (from repo root):
    python -m src.data_toolkit.image_data --output test_data/digits --n 500

This script avoids external fonts and uses Pillow's default font so it works
without extra system setup. For better-looking digits install additional fonts
and pass a font path.
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter


def _random_background(image_size: int):
    # subtle textured background using noise
    arr = np.uint8(np.random.normal(loc=240, scale=8, size=(image_size, image_size)))
    img = Image.fromarray(arr, mode='L').convert('RGB')
    return img


def _add_noise(img: Image.Image, intensity: float = 0.08):
    arr = np.array(img).astype(np.float32) / 255.0
    noise = np.random.normal(scale=intensity, size=arr.shape)
    arr = np.clip(arr + noise, 0.0, 1.0)
    res = Image.fromarray(np.uint8(arr * 255))
    return res


def generate_digit_images(
    output_dir: str | Path = "test_data/digits",
    n_images: int = 500,
    image_size: int = 128,
    classes: List[str] | None = None,
    seed: int | None = 42,
    save_predict_examples: int = 5,
) -> None:
    """Generate synthetic digit images and save them to `output_dir`.

    Args:
        output_dir: Base directory to save images. Will create subfolders `images/` and `predict_examples/`.
        n_images: Total number of generated images.
        image_size: Square image size in pixels.
        classes: List of class labels (default digits '0'..'9').
        seed: Random seed for reproducibility.
        save_predict_examples: Number of reserved images to keep aside for prediction testing.
    """
    random.seed(seed)
    np.random.seed(seed if seed is not None else None)

    if classes is None:
        classes = [str(i) for i in range(10)]

    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    predict_dir = output_dir / "predict_examples"
    images_dir.mkdir(parents=True, exist_ok=True)
    predict_dir.mkdir(parents=True, exist_ok=True)

    # Use default font; if user wants nicer fonts they can edit this file
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    records = []
    for i in range(n_images):
        label = random.choice(classes)
        # blank background
        bg = _random_background(image_size)

        draw = ImageDraw.Draw(bg)

        # choose font size relative to image
        font_size = random.randint(int(image_size * 0.4), int(image_size * 0.75))
        # try to use a truetype fallback if available
        try:
            pil_font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
        except Exception:
            pil_font = font

        text = label

        # measure text size and place roughly in center with jitter
        try:
            # preferred in newer Pillow: textbbox gives tight bbox
            bbox = draw.textbbox((0, 0), text, font=pil_font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
        except Exception:
            # fallback for older Pillow / font implementations
            try:
                w, h = pil_font.getsize(text)
            except Exception:
                # last resort: approximate
                w = int(image_size * 0.5)
                h = int(image_size * 0.6)
        x = (image_size - w) // 2 + random.randint(-int(image_size * 0.06), int(image_size * 0.06))
        y = (image_size - h) // 2 + random.randint(-int(image_size * 0.06), int(image_size * 0.06))

        # draw the digit in dark color
        fill = (random.randint(10, 30),) * 3
        draw.text((x, y), text, font=pil_font, fill=fill)

        # random rotation
        angle = random.uniform(-25, 25)
        img = bg.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=(255, 255, 255))

        # add blur sometimes
        if random.random() < 0.15:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.5)))

        # add noise
        img = _add_noise(img, intensity=random.uniform(0.02, 0.12))

        # optionally resize jitter
        if random.random() < 0.2:
            scale = random.uniform(0.9, 1.1)
            new_size = max(8, int(image_size * scale))
            img = img.resize((new_size, new_size), resample=Image.BILINEAR).resize((image_size, image_size))

        # final convert
        fname = f"img_{i:04d}_lbl_{label}.png"

        # save some examples to predict_examples folder
        if i < save_predict_examples:
            out_path = predict_dir / fname
            split = 'predict'
        else:
            out_path = images_dir / fname
            split = 'train'

        img.save(out_path)

        # record for CSV (relative path from output_dir)
        rel_path = str(out_path.relative_to(output_dir))
        records.append({'filename': rel_path, 'label': label, 'split': split})

    # write labels CSV
    import csv
    csv_path = output_dir / 'labels.csv'
    with open(csv_path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=['filename', 'label', 'split'])
        writer.writeheader()
        for r in records:
            writer.writerow(r)

    print(f"Generated {n_images} images under {output_dir} (images/ + predict_examples/) and wrote labels to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic digit images")
    parser.add_argument("--output", type=str, default="test_data/digits", help="Output directory")
    parser.add_argument("--n", type=int, default=500, help="Number of images to generate")
    parser.add_argument("--size", type=int, default=128, help="Image size (square)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    generate_digit_images(output_dir=args.output, n_images=args.n, image_size=args.size, seed=args.seed)
