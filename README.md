# FusionSpaceModalityGap

## Training Stable Diffusion 1.4 with Multiple Generations

The `train_sd1_4.py` script allows you to train and generate images using Stable Diffusion 1.4 across multiple generations. Each generation is trained on the images from the previous generation, creating an iterative refinement process.

### Prerequisites

- Python 3.7+
- PyTorch
- Hugging Face `diffusers` library
- CUDA-capable GPU with sufficient memory (16GB+ recommended)
- MSCOCO dataset with captions

### Installation

```bash
pip install torch diffusers transformers datasets evaluate torchmetrics clip
```

### Usage

Default configuration (recommended for full training):
```bash
python train_sd1_4.py
```
This will run with:
- 100 images per generation
- 5 training epochs
- 50 sampling steps
- CFG scale of 1.0
- 3 target generations

Custom configuration:
```bash
python train_sd1_4.py \
    --cfg_scale 1.0 \          # Classifier-free guidance scale
    --target_gen 3 \           # Target generation to reach
    --steps 50 \              # Number of inference steps for generation
    --num_images 100 \        # Number of images per generation (except gen_0)
    --epochs 5                # Number of training epochs
```

### Parameters

- `--cfg_scale` (float, default=1.0): Classifier-free guidance scale for training and generation
- `--target_gen` (int, default=3): Target generation number to reach
- `--steps` (int, default=50): Number of inference steps for image generation
- `--num_images` (int, default=100): Number of images to generate for each generation (Note: Generation 0 always uses 100 images for proper training)
- `--epochs` (int, default=5): Number of training epochs per generation

### Process Overview

1. **Generation 0**: 
   - Always generates 100 images using the base SD 1.4 model
   - These images are used to train Generation 1

2. **Subsequent Generations**:
   - Trains on images from the previous generation
   - Generates the specified number of images (controlled by `--num_images`)
   - Each generation uses the same CFG scale and number of steps

### Directory Structure

The script creates the following directory structure:
```
data/coco/
├── sd_to_sd_cfg_{cfg_scale}_steps_{steps}_gen_0/    # Generation 0 images
├── sd_to_sd_cfg_{cfg_scale}_steps_{steps}_gen_1/    # Generation 1 images
└── ...                                              # Subsequent generations

models/
├── sd_to_sd_cfg_{cfg_scale}_steps_{steps}_gen_1/    # Generation 1 model
├── sd_to_sd_cfg_{cfg_scale}_steps_{steps}_gen_2/    # Generation 2 model
└── ...                                              # Subsequent generations
```

### Examples

1. Default full training run (recommended):
```bash
python train_sd1_4.py
```

2. Modifying CFG scale only:
```bash
python train_sd1_4.py --cfg_scale 7.5
```

3. Extended training with more generations:
```bash
python train_sd1_4.py --target_gen 5
```

4. Quick test/debug run:
```bash
python train_sd1_4.py --target_gen 1 --num_images 1 --epochs 1
```

5. Custom training configuration:
```bash
python train_sd1_4.py --cfg_scale 7.5 --target_gen 5 --steps 100 --epochs 10
```

### Notes

- Generation 0 always uses 100 images regardless of the `--num_images` parameter to ensure proper training
- Each generation is trained on all images from the previous generation
- Models and images are saved after each generation
- The script includes automatic evaluation metrics (FID, IS, CLIP score)
- Training can be resumed from the last completed generation