# Longshot

Smart screenshot stitching tool that automatically detects overlapping regions and merges multiple long screenshots using the longest common substring algorithm.

## Features

- Automatic overlap detection
- Smart scrollbar exclusion
- Wildcard pattern support
- High quality output

## Usage

```bash
# Install dependencies
uv sync

# Stitch images matching pattern
uv run python main.py "IMG_627FF0035451-*.jpeg"
uv run python main.py "screenshot-*.png"

# Custom scrollbar ignore pixels
uv run python main.py "page-*.jpg" --ignore-pixels 30

# Specify output filename
uv run python main.py "IMG_*.png" --output "result.png"
```

Output files default to `{prefix}-concat.{extension}` format.
