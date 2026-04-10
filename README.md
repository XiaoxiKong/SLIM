# SLIM: Stable Latent Integration for Robust Watermark in Diffusion Model

SLIM embeds invisible, extractable watermarks into Stable Diffusion generated images. A compact encoder maps a secret bit string into a latent perturbation, which is injected into the UNet during denoising via LoRA-adapted residuals. A lightweight decoder then recovers the secret from the generated image.


Training is split into two stages:

- Stage 1: train `SecretEncoder` and `SecretDecoder` in latent space.
- Stage 2: freeze Stage 1 and fine-tune the Stable Diffusion UNet with LoRA so watermark injection integrates into denoising.

## Important Notes Before Reproducing

- This repository includes a locally modified `diffusers/` implementation. The UNet changes inside this local copy are part of the method and must be used as-is.
- Do not install another `diffusers` package over the repository code and expect identical behavior. When you run scripts from the project root, Python will import the local `diffusers/` folder first.
- `xformers` is optional. Training will continue without it, but memory usage will be higher.
- For strict reproducibility, keep `secret_len`, LoRA rank, base model, prompt file, and random seed fixed across runs.

## Environment

### Option 1: Conda

```bash
conda env create -f environment.yml
conda activate slim
```

### Option 2: pip

```bash
pip install -r requirements.txt
```

Test the three entry points:

```bash
python scripts/first_stage_train.py --help
python scripts/second_stage_finetune.py --help
python scripts/inference.py --help
```

## Base Model

All scripts expect a Stable Diffusion checkpoint compatible with the v1.5 architecture.

- Recommended model ID: `runwayml/stable-diffusion-v1-5`
- You may also pass a local path with the standard HuggingFace Diffusers folder layout.

## Project Layout

```text
SLIM/
├── cfg/
│   └── mountain.json
├── ckpt/
│   ├── sd1.5/
│   └── sd2.1/
├── cldm/
├── diffusers/
├── utils_f/
│   ├── models.py
│   └── noise_layers/
├── scripts/
│   ├── custom_pipeline.py
│   ├── second_stage_finetune.py
│   ├── first_stage_train.py
│   ├── inference.py
│   └── training_cfg.py
├── utils.py
├── VQ4_mir.yaml
├── environment.yml
└── requirements.txt
```

## Stage 1: Train SecretEncoder and SecretDecoder

Run from the repository root:

```bash
python scripts/first_stage_train.py \
  --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
  --dataset /path/to/train_images \
  --output_dir ./output/stage1 \
  --epochs 40 \
  --bit_num 48 \
  --batch_size 1 \
  --lr 1e-3
```

Arguments:

| Argument | Default | Description |
|---|---:|---|
| `--pretrained_model_name_or_path` | required | SD model ID or local model path |
| `--dataset` | required | Directory containing `.png` / `.jpg` training images |
| `--output_dir` | `checkpoints-slim-stage1` | Output directory |
| `--epochs` | `40` | Training epochs |
| `--batch_size` | `1` | Batch size |
| `--bit_num` | `48` | Secret length |
| `--resume_from_ckpt` | `None` | Resume from a previous Stage 1 checkpoint |
| `--warmup` / `--no-warmup` | enabled | Warmup phase |
| `--fixinit` / `--no-fixinit` | enabled | Zero-image initialization at training start |
| `--random_aug` / `--no-random_aug` | enabled | Random flip/rotation augmentation |
| `--lr` | `1e-3` | Learning rate |

Outputs:

- checkpoints: `<output_dir>/checkpoints/state_dict_<epoch>.pth`
- preview images: `<output_dir>/log_images/`
- tensorboard logs: `<output_dir>/logs/`

Each Stage 1 checkpoint contains:

- `sec_encoder`
- `sec_decoder`

## Stage 2: LoRA Fine-tuning

Stage 2 requires:

- a Stage 1 checkpoint
- an image-text dataset
- a training config JSON

Example:

```bash
python scripts/second_stage_finetune.py \
  --cfg cfg/mountain.json \
  --secret_models_path ./output/stage1/checkpoints/state_dict_39.pth
```

### Training Config

`cfg/mountain.json` is a template. At minimum, update:

```json
{
  "pretrained_model_name_or_path": "runwayml/stable-diffusion-v1-5",
  "data_root_path": "/path/to/images",
  "data_json_file": "/path/to/annotations.json",
  "output_dir": "output/checkpoints",
  "rank": 50,
  "secret_len": 48,
  "n_epochs": 30,
  "train_batch_size": 1,
  "learning_rate_lora": 1e-4
}
```

Notes:

- `secret_len` must match the Stage 1 checkpoint.
- `rank` controls the LoRA rank and should be kept fixed between training and inference.
- `enable_xformers_memory_efficient_attention` can be set to `false` if `xformers` is unavailable.

### Annotation File Format

The loader expects a JSON file with this structure:

```json
[
  [
    {
      "image_file": "relative/path/to/image.jpg",
      "text": "caption for this image"
    }
  ]
]
```

Where:

- `image_file` is resolved relative to `data_root_path`
- `text` is the prompt/caption used for conditioning

### Stage 2 Outputs

Checkpoints are saved under:

```text
<output_dir>/checkpoint-<global_step>/
```

Each checkpoint directory contains:

- `lora_weights.pth`
- `training_state.pth`

## Inference

Generate watermarked images and verify extraction accuracy:

```bash
python scripts/inference.py \
  --model_id runwayml/stable-diffusion-v1-5 \
  --checkpoint_path ./output/stage1/checkpoints/xxxx.pth \
  --lora_path ./output/checkpoints/checkpoint-500/lora_weights.pth \
  --prompt_file /path/to/prompts.json \
  --output_dir ./output/inference \
  --start 0 \
  --end 100 \
  --bit_num 48 \
  --seed 1234
```

Arguments:

| Argument | Default | Description |
|---|---:|---|
| `--model_id` | `runwayml/stable-diffusion-v1-5` | Base SD model ID or local path |
| `--checkpoint_path` | `None` | Stage 1 checkpoint |
| `--lora_path` | `None` | Stage 2 LoRA weights |
| `--prompt_file` | `None` | Prompt JSON file or HF dataset name |
| `--output_dir` | `output` | Output directory |
| `--start` | `0` | Start index |
| `--end` | `5` | End index |
| `--bit_num` | `48` | Secret length |
| `--num_images` | `1` | Images per prompt |
| `--guidance_scale` | `7.5` | CFG guidance scale |
| `--num_inference_steps` | `25` | Denoising steps |
| `--image_length` | `512` | Output size |
| `--seed` | `None` | Fixed seed. Set this for reproducibility |

If `--prompt_file` is omitted, the script falls back to a short built-in demo prompt list.

### Prompt File Formats

Accepted JSON formats:

```json
["prompt one", "prompt two"]
```

```json
[{"caption": "prompt one"}, {"caption": "prompt two"}]
```

```json
{"annotations": [{"caption": "prompt one"}]}
```

## Reproducibility Checklist

To make your release reproducible for other readers, publish the following together:

1. The exact Stage 1 checkpoint used for Stage 2 training.
2. The final Stage 2 `lora_weights.pth`.
3. The exact `cfg/*.json` used for training.
4. The prompt file used for evaluation.
5. The fixed random seed used for the reported inference results.
6. A short note describing the dataset source and any filtering/splitting rules.
7. The commit hash of this repository.

For paper-level reproduction, also report:

- Stage 1 training epochs
- Stage 2 training epochs
- LoRA rank
- secret length
- image resolution
- inference scheduler and number of steps

## Common Issues

- `ModuleNotFoundError: mmcv`
  The repository now includes a fallback implementation for the small subset used at import time, so `mmcv` is not required for the default training and inference scripts.

- `xformers is not available`
  Training can continue without `xformers`. Set `enable_xformers_memory_efficient_attention` to `false` or keep the current fallback behavior.

- Out-of-memory during Stage 2
  Reduce `train_batch_size`, lower image resolution, disable xformers-only assumptions, or use gradient accumulation.

- Random-looking results during inference
  Make sure `--checkpoint_path`, `--lora_path`, `--bit_num`, and base model all match the training run.

## Citation

```bibtex
@ARTICLE{11449245,
  author={Kong, Xiaoxi and Chen, Pengdi and Li, Bin and Yuan, Jieyu and Cai, Zhanchuan and Wu, Hao and Liang, Lifeng},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={SLIM: Stable Latent Integration for Robust Watermark in Diffusion Model}, 
  year={2026},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TCSVT.2026.3676184}}

```
