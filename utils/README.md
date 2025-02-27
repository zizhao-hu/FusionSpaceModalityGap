# Generation Comparison Utilities

This folder contains utilities for comparing images across different generations of your Stable Diffusion training.

## Scripts

### 1. `compare_generations.py`

This script allows you to select a single image and compare it across different generations.

#### Usage:

```bash
# Basic usage - opens a file dialog to select an image
python utils/compare_generations.py --cfg_scale 7.0

# Specify an image path directly
python utils/compare_generations.py --cfg_scale 7.0 --image_path data/coco/sd_to_sd_cfg_7_gen_0/COCO_train2014_000000000001.jpg

# Specify an output path for the comparison image
python utils/compare_generations.py --cfg_scale 7.0 --output my_comparison.png
```

### 2. `compare_generations_grid.py`

This script provides more advanced functionality, including:
- Creating a grid of multiple images across generations
- Randomly selecting images that exist across generations

#### Usage:

```bash
# Basic usage - opens a file dialog to select an image
python utils/compare_generations_grid.py --cfg_scale 7.0

# Create a grid of multiple images (opens a file dialog to select multiple images)
python utils/compare_generations_grid.py --cfg_scale 7.0 --grid

# Automatically select 5 random images that exist across generations
python utils/compare_generations_grid.py --cfg_scale 7.0 --random

# Select 10 random images
python utils/compare_generations_grid.py --cfg_scale 7.0 --random --num_random 10

# Specify an output path for the comparison image
python utils/compare_generations_grid.py --cfg_scale 7.0 --random --output my_grid_comparison.png
```

### 3. `compare_gen.py`

This script combines and enhances the functionality of the previous scripts, allowing you to:
- Compare images by their index (0-99) in the sorted directory listing
- Generate new images from custom captions using saved model checkpoints
- Create comparison grids of multiple images

#### Usage:

```bash
# Compare a single image by index
python utils/compare_gen.py --cfg_scale 7.0 --index 5

# Compare multiple images in a grid
python utils/compare_gen.py --cfg_scale 7.0 --indices 0 10 20 30 40

# Generate new images from a custom caption using all available model generations
python utils/compare_gen.py --cfg_scale 7.0 --caption "a photo of a cat wearing a hat"

# Generate images from a caption with specific generations only
python utils/compare_gen.py --cfg_scale 7.0 --caption "a photo of a cat wearing a hat" --generations 0 1 3

# Use a specific seed for deterministic generation
python utils/compare_gen.py --cfg_scale 7.0 --caption "a photo of a cat wearing a hat" --seed 42

# Specify an output directory for generated images
python utils/compare_gen.py --cfg_scale 7.0 --caption "a photo of a cat wearing a hat" --output_dir my_generated_images

# Specify maximum generation to check
python utils/compare_gen.py --cfg_scale 7.0 --index 5 --max_gen 5

# Specify an output path for the comparison image
python utils/compare_gen.py --cfg_scale 7.0 --index 5 --output my_index_comparison.png
```

## Parameters

All scripts accept the following common parameters:
- `--cfg_scale`: The CFG scale used for generation (default: 7.0)
- `--output`: Output path for the comparison image (optional)

Script-specific parameters:

- `compare_generations.py` and `compare_generations_grid.py`:
  - `--image_path`: Path to an image to compare (optional)
  - `--grid`: Create a grid of multiple images (grid script only)
  - `--random`: Select random images that exist across generations (grid script only)
  - `--num_random`: Number of random images to select (default: 5, grid script only)

- `compare_gen.py`:
  - `--index`: Index of the image to compare (0-99)
  - `--indices`: Multiple indices to compare in a grid (e.g., `--indices 0 10 20 30`)
  - `--caption`: Custom caption to generate new images with across all generations
  - `--generations`: Specific generations to use (default: all available)
  - `--seed`: Random seed for generation (default: random)
  - `--max_gen`: Maximum generation number to check (default: 10)
  - `--output_dir`: Directory to save generated images (default: "generated_comparisons")

## Examples

1. To compare a specific image across all generations with CFG scale 7.0:
   ```bash
   python utils/compare_generations.py --cfg_scale 7.0 --image_path data/coco/sd_to_sd_cfg_7_gen_0/COCO_train2014_000000000001.jpg
   ```

2. To create a grid of 3 random images across all generations:
   ```bash
   python utils/compare_generations_grid.py --cfg_scale 7.0 --random --num_random 3
   ```

3. To manually select multiple images for comparison:
   ```bash
   python utils/compare_generations_grid.py --cfg_scale 7.0 --grid
   ```

4. To compare the 5th image (index 4) across all generations:
   ```bash
   python utils/compare_gen.py --cfg_scale 7.0 --index 4
   ```

5. To create a grid comparing images at indices 0, 25, 50, and 75:
   ```bash
   python utils/compare_gen.py --cfg_scale 7.0 --indices 0 25 50 75
   ```

6. To generate images from a custom caption using all model generations:
   ```bash
   python utils/compare_gen.py --cfg_scale 7.0 --caption "a beautiful landscape with mountains and a lake"
   ```

7. To generate images with a fixed seed for reproducibility:
   ```bash
   python utils/compare_gen.py --cfg_scale 7.0 --caption "a portrait of a woman with blue eyes" --seed 12345
   ``` 