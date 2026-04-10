import argparse
import io
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm
from diffusers import DPMSolverMultistepScheduler

from custom_pipeline import WatermarkPipeline
from inference import SecretDecoder, SecretEncoder
from utils import get_dataset


DEFAULT_PROMPTS = [
    "A cinematic mountain landscape at sunrise with golden mist and detailed pine trees.",
    "A clean studio product photo of a glass bottle with soft reflections and white background.",
    "An oil painting of a harbor at dusk with dramatic clouds and warm lantern light.",
]


def tensor_to_pil(image_tensor):
    image_tensor = torch.clamp(image_tensor.detach().cpu().float(), 0.0, 1.0)
    return transforms.ToPILImage()(image_tensor)


def pil_to_decoder_tensor(image, device, dtype):
    tensor = transforms.ToTensor()(image).to(device=device, dtype=dtype)
    return tensor * 2.0 - 1.0


def decode_and_score(decoder, image_tensor, secret_bits):
    logits = decoder(image_tensor.unsqueeze(0))
    pred_bits = torch.argmax(logits, dim=2)
    acc = 1.0 - torch.abs(pred_bits - secret_bits).sum().float() / secret_bits.numel()

    batch_indices = torch.arange(secret_bits.shape[0], device=logits.device)[:, None]
    bit_indices = torch.arange(secret_bits.shape[1], device=logits.device)[None, :]
    target_idx = secret_bits.long()
    target_logits = logits[batch_indices, bit_indices, target_idx]
    other_logits = logits[batch_indices, bit_indices, 1 - target_idx]
    score = (target_logits - other_logits).mean()
    return logits, acc.item(), score.item()


def quantile_threshold(negative_scores, fpr_target):
    sorted_scores = np.sort(np.asarray(negative_scores, dtype=np.float64))
    if sorted_scores.size == 0:
        return float("inf")
    quantile = 1.0 - fpr_target
    index = int(np.ceil(quantile * sorted_scores.size)) - 1
    index = min(max(index, 0), sorted_scores.size - 1)
    return float(sorted_scores[index])


def compute_tpr_at_fpr(positive_scores, negative_scores, fpr_target=0.001):
    threshold = quantile_threshold(negative_scores, fpr_target)
    positive_scores = np.asarray(positive_scores, dtype=np.float64)
    negative_scores = np.asarray(negative_scores, dtype=np.float64)
    tpr = float(np.mean(positive_scores >= threshold)) if positive_scores.size else 0.0
    fpr = float(np.mean(negative_scores >= threshold)) if negative_scores.size else 0.0
    return threshold, tpr, fpr


def build_attack_specs():
    return {
        "identity": {"name": "identity"},
        "jpeg_50": {"name": "jpeg", "quality": 50},
        "jpeg_30": {"name": "jpeg", "quality": 30},
        "rotate_5": {"name": "rotate", "degrees": 5},
        "center_crop_0.9": {"name": "center_crop", "scale": 0.9},
        "gaussian_blur": {"name": "gaussian_blur", "radius": 1.5},
        "gaussian_noise": {"name": "gaussian_noise", "std": 8.0},
        "brightness": {"name": "brightness", "factor": 1.15},
        "color": {"name": "color", "factor": 1.2},
    }


def apply_attack(image, spec, seed):
    attack_name = spec["name"]
    if attack_name == "identity":
        return image.copy()

    if attack_name == "jpeg":
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=spec["quality"])
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

    if attack_name == "rotate":
        return image.rotate(spec["degrees"], resample=Image.Resampling.BICUBIC)

    if attack_name == "center_crop":
        width, height = image.size
        scale = spec["scale"]
        crop_w = int(width * scale)
        crop_h = int(height * scale)
        left = max((width - crop_w) // 2, 0)
        top = max((height - crop_h) // 2, 0)
        cropped = image.crop((left, top, left + crop_w, top + crop_h))
        return cropped.resize((width, height), Image.Resampling.BICUBIC)

    if attack_name == "gaussian_blur":
        return image.filter(ImageFilter.GaussianBlur(radius=spec["radius"]))

    if attack_name == "gaussian_noise":
        rng = np.random.default_rng(seed)
        image_np = np.asarray(image).astype(np.float32)
        noise = rng.normal(0.0, spec["std"], image_np.shape)
        noised = np.clip(image_np + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noised)

    if attack_name == "brightness":
        return ImageEnhance.Brightness(image).enhance(spec["factor"])

    if attack_name == "color":
        return ImageEnhance.Color(image).enhance(spec["factor"])

    raise ValueError(f"Unknown attack: {attack_name}")


def load_prompt_dataset(prompt_file):
    if prompt_file and os.path.exists(prompt_file):
        return get_dataset(prompt_file)
    return DEFAULT_PROMPTS, None


def get_prompt(dataset, prompt_key, index):
    if prompt_key:
        return dataset[index][prompt_key]
    return dataset[index]


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weight_dtype = torch.float16 if device == "cuda" else torch.float32

    scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_id, subfolder="scheduler")
    pipe = WatermarkPipeline.from_pretrained(
        args.model_id,
        scheduler=scheduler,
        torch_dtype=weight_dtype,
        safety_checker=None,
    ).to(device)

    if not args.checkpoint_path or not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError("--checkpoint_path must point to a valid Stage 1 checkpoint for robustness evaluation.")
    if not args.lora_path or not os.path.exists(args.lora_path):
        raise FileNotFoundError("--lora_path must point to a valid Stage 2 LoRA checkpoint for robustness evaluation.")

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    encoder = SecretEncoder(args.bit_num).to(device=device, dtype=weight_dtype)
    encoder.load_state_dict(checkpoint["sec_encoder"])
    encoder.eval()

    decoder = SecretDecoder(args.bit_num).to(device=device, dtype=weight_dtype)
    decoder.load_state_dict(checkpoint["sec_decoder"])
    decoder.eval()

    dataset, prompt_key = load_prompt_dataset(args.prompt_file)
    num_samples = min(args.end, len(dataset))
    start_idx = min(args.start, num_samples)
    if start_idx >= num_samples:
        raise ValueError("Start index is out of bounds or dataset is empty.")

    attack_specs = build_attack_specs()
    selected_attacks = [name.strip() for name in args.attacks.split(",") if name.strip()]
    unknown_attacks = [name for name in selected_attacks if name not in attack_specs]
    if unknown_attacks:
        raise ValueError(f"Unknown attacks requested: {unknown_attacks}")

    os.makedirs(args.output_dir, exist_ok=True)

    metrics = {
        attack_name: {
            "positive_acc": [],
            "positive_scores": [],
            "negative_scores": [],
        }
        for attack_name in selected_attacks
    }

    sample_records = []
    base_seed = args.seed if args.seed is not None else 1234

    for index in tqdm(range(start_idx, num_samples), desc="Evaluating attacks"):
        prompt = get_prompt(dataset, prompt_key, index)
        sample_seed = base_seed + index
        secret_generator = torch.Generator(device=device).manual_seed(sample_seed)
        secret_bits = torch.randint(0, 2, (1, args.bit_num), generator=secret_generator, device=device, dtype=weight_dtype)

        with torch.no_grad():
            _, eps = encoder(torch.zeros(1, 4, 64, 64, device=device, dtype=weight_dtype), secret_bits)
            eps = eps * 0.18215

            wm_generator = torch.Generator(device=device).manual_seed(sample_seed)
            clean_generator = torch.Generator(device=device).manual_seed(sample_seed)

            watermarked = pipe(
                prompt,
                num_images_per_prompt=1,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                height=args.image_length,
                width=args.image_length,
                generator=wm_generator,
                eps=eps,
                output_type="pt",
                impuwater=1,
                lora_path=args.lora_path,
            ).images[0]

            clean = pipe(
                prompt,
                num_images_per_prompt=1,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                height=args.image_length,
                width=args.image_length,
                generator=clean_generator,
                output_type="pt",
                impuwater=0,
                lora_path=None,
            ).images[0]

        watermarked_pil = tensor_to_pil(watermarked)
        clean_pil = tensor_to_pil(clean)

        sample_result = {"index": index, "prompt": prompt, "seed": sample_seed, "attacks": {}}

        for attack_name in selected_attacks:
            attack_seed = sample_seed * 100 + selected_attacks.index(attack_name)
            attacked_positive = apply_attack(watermarked_pil, attack_specs[attack_name], attack_seed)
            attacked_negative = apply_attack(clean_pil, attack_specs[attack_name], attack_seed)

            positive_tensor = pil_to_decoder_tensor(attacked_positive, device, weight_dtype)
            negative_tensor = pil_to_decoder_tensor(attacked_negative, device, weight_dtype)

            with torch.no_grad():
                _, positive_acc, positive_score = decode_and_score(decoder, positive_tensor, secret_bits)
                _, _, negative_score = decode_and_score(decoder, negative_tensor, secret_bits)

            metrics[attack_name]["positive_acc"].append(positive_acc)
            metrics[attack_name]["positive_scores"].append(positive_score)
            metrics[attack_name]["negative_scores"].append(negative_score)

            sample_result["attacks"][attack_name] = {
                "acc": positive_acc,
                "positive_score": positive_score,
                "negative_score": negative_score,
            }

            if args.save_attacked_images and index == start_idx:
                attacked_positive.save(os.path.join(args.output_dir, f"{attack_name}_sample_{index}_pos.png"))
                attacked_negative.save(os.path.join(args.output_dir, f"{attack_name}_sample_{index}_neg.png"))

        sample_records.append(sample_result)

    summary = {
        "fpr_target": args.fpr_target,
        "num_samples": num_samples - start_idx,
        "attacks": {},
    }

    print(f"Evaluated {num_samples - start_idx} samples")
    for attack_name in selected_attacks:
        attack_metrics = metrics[attack_name]
        threshold, tpr, actual_fpr = compute_tpr_at_fpr(
            attack_metrics["positive_scores"],
            attack_metrics["negative_scores"],
            fpr_target=args.fpr_target,
        )
        mean_acc = float(np.mean(attack_metrics["positive_acc"])) if attack_metrics["positive_acc"] else 0.0

        summary["attacks"][attack_name] = {
            "acc": mean_acc,
            "tpr_at_fpr": tpr,
            "actual_fpr": actual_fpr,
            "threshold": threshold,
            "num_positive": len(attack_metrics["positive_scores"]),
            "num_negative": len(attack_metrics["negative_scores"]),
        }

        print(
            f"[{attack_name}] "
            f"ACC={mean_acc:.4f} "
            f"TPR@{args.fpr_target * 100:.3f}%FPR={tpr:.4f} "
            f"(actual FPR={actual_fpr:.6f}, threshold={threshold:.4f})"
        )

    results = {
        "summary": summary,
        "samples": sample_records,
    }
    results_path = os.path.join(args.output_dir, "robustness_results.json")
    with open(results_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
    print(f"Saved results to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate watermark robustness under image attacks.")
    parser.add_argument("--model_id", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--checkpoint_path", required=True,
                        help="Stage 1 checkpoint containing sec_encoder and sec_decoder.")
    parser.add_argument("--lora_path", required=True,
                        help="Stage 2 LoRA weights for watermark generation.")
    parser.add_argument("--prompt_file", default=None,
                        help="Prompt JSON file or dataset name. Falls back to built-in prompts if omitted.")
    parser.add_argument("--output_dir", default="output/robustness_eval")
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=100, type=int)
    parser.add_argument("--image_length", default=512, type=int)
    parser.add_argument("--bit_num", default=48, type=int)
    parser.add_argument("--guidance_scale", default=7.5, type=float)
    parser.add_argument("--num_inference_steps", default=25, type=int)
    parser.add_argument("--seed", default=1234, type=int,
                        help="Base seed used for generation, secret sampling, and deterministic attacks.")
    parser.add_argument("--fpr_target", default=0.001, type=float,
                        help="Target false positive rate. 0.001 means 0.1%% FPR.")
    parser.add_argument("--attacks", default="identity,jpeg_50,jpeg_30,rotate_5,center_crop_0.9,gaussian_blur,gaussian_noise,brightness,color",
                        help="Comma-separated attack names.")
    parser.add_argument("--save_attacked_images", action="store_true",
                        help="Save the first attacked positive/negative pair for each attack.")
    args = parser.parse_args()
    main(args)
