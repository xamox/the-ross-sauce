#!/usr/bin/env python3
"""
Painterly image filter without OpenCV.
Usage:
    python paintify.py input.jpg -o output.png
"""
import argparse
import math
import random
import numpy as np
from PIL import Image, ImageOps, ImageDraw


def _to_array(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


def _to_image(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _box_blur(arr: np.ndarray, radius: int, passes: int = 2) -> np.ndarray:
    # Separable box blur; slower than OpenCV but dependency-free.
    blurred = arr
    for _ in range(passes):
        # Horizontal blur
        padded = np.pad(blurred, ((0, 0), (radius, radius), (0, 0)), mode="edge")
        cumsum = np.cumsum(padded, axis=1)
        blurred = (cumsum[:, 2 * radius :, :] - cumsum[:, :-2 * radius, :]) / (
            2 * radius
        )

        # Vertical blur
        padded = np.pad(blurred, ((radius, radius), (0, 0), (0, 0)), mode="edge")
        cumsum = np.cumsum(padded, axis=0)
        blurred = (cumsum[2 * radius :, :, :] - cumsum[:-2 * radius, :, :]) / (
            2 * radius
        )
    return blurred


def _convolve(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    k = kernel.shape[0] // 2
    padded = np.pad(img, ((k, k), (k, k)), mode="edge")
    out = np.zeros_like(img)
    for y in range(out.shape[0]):
        for x in range(out.shape[1]):
            region = padded[y : y + 2 * k + 1, x : x + 2 * k + 1]
            out[y, x] = np.sum(region * kernel)
    return out


def _sobel_edges(arr: np.ndarray) -> np.ndarray:
    gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
    ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    gx = _convolve(gray, kx)
    gy = _convolve(gray, ky)
    mag = np.sqrt(gx * gx + gy * gy)
    mag = mag / (mag.max() + 1e-6)
    return mag


def _sobel_gradients(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized Sobel gradients (gx, gy)."""
    gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
    ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    gx = _convolve(gray, kx)
    gy = _convolve(gray, ky)
    norm = np.sqrt(gx * gx + gy * gy) + 1e-6
    return gx / norm, gy / norm


def _posterize(arr: np.ndarray, levels: int = 8) -> np.ndarray:
    return np.floor(arr * levels + 0.5) / levels


def paintify(
    arr: np.ndarray,
    *,
    radius: int = 3,
    passes: int = 2,
    levels: int = 12,
    edge_scale: float = 1.5,
    edge_base: float = 0.1,
) -> np.ndarray:
    base = _box_blur(arr, radius=radius, passes=passes)
    colors = _posterize(base, levels=levels)
    edges = _sobel_edges(arr) * edge_scale
    edges = np.expand_dims(edges, axis=2)
    edge_mask = np.clip(1.0 - edges, 0, 1)
    painted = colors * edge_mask + (1 - edge_mask) * edge_base
    return painted


def _stroke_paint(
    base: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    *,
    strokes: int = 4000,
    min_radius: int = 2,
    max_radius: int = 6,
    scales: int = 3,
    opacity: float = 0.65,
) -> np.ndarray:
    """Layer painterly strokes that follow local structure."""
    h, w, _ = base.shape
    base_img = _to_image(base).convert("RGBA")
    base_rgb = base_img.convert("RGB")
    overlay = Image.new("RGBA", (w, h))
    draw = ImageDraw.Draw(overlay, "RGBA")

    radii = np.linspace(max_radius, min_radius, num=max(scales, 1))
    strokes_per_scale = max(strokes // max(scales, 1), 1)

    for radius in radii:
        r_int = max(1, int(round(radius)))
        for _ in range(strokes_per_scale):
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)
            angle = math.atan2(gy[y, x], gx[y, x]) + math.pi / 2.0
            angle += random.uniform(-0.5, 0.5)
            length = r_int * (1.2 + random.random() * 1.5)
            x2 = x + length * math.cos(angle)
            y2 = y + length * math.sin(angle)

            color = base_rgb.getpixel((x, y))
            draw.line(
                [(x, y), (x2, y2)],
                fill=color + (int(255 * opacity),),
                width=r_int,
                joint="curve",
            )

    composited = Image.alpha_composite(base_img, overlay).convert("RGB")
    return _to_array(composited)


def main() -> None:
    parser = argparse.ArgumentParser(description="Painterly filter without OpenCV.")
    parser.add_argument("input", help="Input image (jpg/png)")
    parser.add_argument(
        "-o", "--output", default="painted.png", help="Output image path"
    )
    parser.add_argument("--radius", type=int, default=3, help="Blur radius")
    parser.add_argument(
        "--passes", type=int, default=2, help="Number of box-blur passes"
    )
    parser.add_argument(
        "--levels", type=int, default=12, help="Posterize color levels"
    )
    parser.add_argument(
        "--edge-scale",
        type=float,
        default=1.5,
        help="Edge darkening strength (higher = darker lines)",
    )
    parser.add_argument(
        "--edge-base",
        type=float,
        default=0.1,
        help="Base tone for fully darkened edges (0-1)",
    )
    parser.add_argument(
        "--strokes",
        type=int,
        default=4000,
        help="Approximate number of brush strokes to overlay",
    )
    parser.add_argument(
        "--stroke-min-radius",
        type=int,
        default=2,
        help="Smallest stroke width",
    )
    parser.add_argument(
        "--stroke-max-radius",
        type=int,
        default=6,
        help="Largest stroke width",
    )
    parser.add_argument(
        "--stroke-scales",
        type=int,
        default=3,
        help="Number of stroke size scales",
    )
    parser.add_argument(
        "--stroke-opacity",
        type=float,
        default=0.65,
        help="Stroke alpha (0-1)",
    )
    parser.add_argument(
        "--no-strokes",
        action="store_true",
        help="Disable brush stroke overlay",
    )
    args = parser.parse_args()

    img = ImageOps.exif_transpose(Image.open(args.input))
    arr = _to_array(img)
    painted = paintify(
        arr,
        radius=args.radius,
        passes=args.passes,
        levels=args.levels,
        edge_scale=args.edge_scale,
        edge_base=args.edge_base,
    )
    if not args.no_strokes and args.strokes > 0 and args.stroke_opacity > 0:
        gx, gy = _sobel_gradients(arr)
        painted = _stroke_paint(
            painted,
            gx,
            gy,
            strokes=args.strokes,
            min_radius=args.stroke_min_radius,
            max_radius=args.stroke_max_radius,
            scales=args.stroke_scales,
            opacity=args.stroke_opacity,
        )

    out = _to_image(painted)
    out.save(args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
