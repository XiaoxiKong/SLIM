# SLIM: Stable Latent Integration for Robust Watermark in Diffusion Model

SLIM embeds invisible watermarks into Stable Diffusion generated images. A secret encoder maps a bit string into a latent perturbation, injected into the UNet via LoRA-adapted residuals during the final denoising steps. A decoder recovers the secret from the generated image.

## Setup

```bash
conda env create -f environment.yml
conda activate slim
```

Or with pip:

```bash
pip install -r requirements.txt
```

> The repo includes a locally modified `diffusers/` folder. Run all scripts from the project root so Python imports this local copy.

## Usage

### Stage 1 — Train Encoder/Decoder

```bash
python scripts/first_stage_train.py \
  --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
  --dataset /path/to/images \
  --output_dir ./output/stage1 \
  --bit_num 48
```

Checkpoints are saved as `<output_dir>/checkpoints/state_dict_<epoch>.pth`, each containing `sec_encoder` and `sec_decoder` keys.

### Stage 2 — LoRA Fine-tuning

Edit `cfg/mountain.json` to set your paths and hyperparameters, then:

```bash
python scripts/second_stage_finetune.py \
  --cfg cfg/mountain.json \
  --secret_models_path ./output/stage1/checkpoints/xxxx.pth
```

The annotation JSON expected by the dataloader:

```json
[[{"image_file": "relative/path.jpg", "text": "caption"}]]
```

LoRA weights are saved under `<output_dir>/checkpoint-<step>/lora_weights.pth`.

### Inference

```bash
python scripts/inference.py \
  --model_id runwayml/stable-diffusion-v1-5 \
  --checkpoint_path ./output/stage1/checkpoints/xxx.pth \
  --lora_path ./output/checkpoint-xxx/lora_weights.pth \
  --output_dir ./output/inference \
  --bit_num 48
```

If `--prompt_file` is omitted, a short built-in demo prompt list is used.

## Citation

```bibtex
@ARTICLE{11449245,
  author={Kong, Xiaoxi and Chen, Pengdi and Li, Bin and Yuan, Jieyu and Cai, Zhanchuan and Wu, Hao and Liang, Lifeng},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  title={SLIM: Stable Latent Integration for Robust Watermark in Diffusion Model},
  year={2026},
  pages={1-1},
  doi={10.1109/TCSVT.2026.3676184}}
```
