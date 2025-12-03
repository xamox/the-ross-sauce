Painterly image filter using only Pillow and NumPy. Converts JPEG/PNG into a brushy, posterized painting with optional stroke overlay.

## Quick start
```bash
pip install pillow numpy
python paintify.py input.jpg -o painted.png
```

## Useful flags
- `--radius/--passes` blur strength (default 3 / 2)
- `--levels` posterize levels (default 12; lower = flatter color)
- `--edge-scale` edge darkening (default 1.5)
- `--edge-base` darkest edge tone (default 0.1)
- `--strokes` number of strokes (default 4000; `--no-strokes` to disable)
- `--stroke-min-radius/--stroke-max-radius` stroke widths
- `--stroke-scales` how many sizes to layer
- `--stroke-opacity` stroke alpha (0-1)

## Examples
Soft painterly look:
```bash
python paintify.py input.jpg -o painted.png \
  --radius 4 --passes 3 --levels 10 --edge-scale 1.2 \
  --strokes 5000 --stroke-max-radius 8 --stroke-opacity 0.65
```

Chunkier strokes:
```bash
python paintify.py input.jpg -o painted.png \
  --strokes 7000 --stroke-min-radius 3 --stroke-max-radius 12 \
  --stroke-scales 4 --stroke-opacity 0.75
```
