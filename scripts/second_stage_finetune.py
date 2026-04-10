# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fine-tuning script for Stable Diffusion for text2image with support for LoRA and Steganography Injection."""

import argparse
import logging
import math
import os
import random
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
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from peft import LoraConfig
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_xformers_available
from omegaconf import OmegaConf
from PIL import Image
import torchvision.models.efficientnet as efficientnet
from torchvision.models.efficientnet import EfficientNet_B1_Weights

# Assuming these are available in the project structure
try:
    from training_cfg import load_training_config
    from ldm.util import instantiate_from_config
except ImportError:
    def load_training_config(path):
        raise ImportError("training_cfg module not found")
    def instantiate_from_config(config):
        raise ImportError("ldm.util module not found")
import json

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.24.0.dev0")

logger = get_logger(__name__, log_level="INFO")

def save_checkpoint(optimizer, scheduler, epoch, iter, loss, save_path):
    checkpoint = {
        'epoch': epoch,
        'iter': iter,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, save_path)

def load_checkpoint(optimizer, scheduler, load_path):
    checkpoint = torch.load(load_path)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    iter = checkpoint['iter']
    loss = checkpoint.get('loss', None)
    print(f"Checkpoint loaded from {load_path}")
    return optimizer, scheduler, epoch, iter, loss

def save_lora_parameters(model, filepath):
    lora_params = {}
    for name, param in model.named_parameters():
        if 'lora' in name and param.requires_grad:
            lora_params[name] = param.detach().cpu()
    torch.save(lora_params, filepath)
    print(f"LoRA parameters saved to {filepath}")

def load_lora_parameters(model, filepath):
    lora_params = torch.load(filepath)
    model_state = model.state_dict()
    for name, param in lora_params.items():
        if name in model_state:
            model_state[name].copy_(param)
        else:
            print(f"Parameter {name} not found in model. Skipping.")
    model.load_state_dict(model_state)
    print(f"LoRA parameters loaded from {filepath}")

def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

def conv_nd(dims, *args, **kwargs):
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

class View(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)

class Repeat(nn.Module):
    def __init__(self, *sizes):
        super(Repeat, self).__init__()
        self.sizes = sizes

    def forward(self, x):
        return x.repeat(1, *self.sizes)

class SecretEncoder(nn.Module):
    def __init__(self, secret_len, base_res=32, resolution=64) -> None:
        super().__init__()
        log_resolution = int(np.log2(resolution))
        log_base = int(np.log2(base_res))
        self.secret_len = secret_len
        self.secret_scaler = nn.Sequential(
            nn.Linear(secret_len, base_res * base_res),
            nn.SiLU(),
            View(-1, 1, base_res, base_res),
            Repeat(4, 1, 1),
            nn.Upsample(scale_factor=(2 ** (log_resolution - log_base), 2 ** (log_resolution - log_base))),
            zero_module(conv_nd(2, 4, 4, 3, padding=1))
        )

    def encode(self, x):
        x = self.secret_scaler(x)
        return x

    def forward(self, x, c):
        # x: [B, C, H, W], c: [B, secret_len]
        c = self.encode(c)
        c = F.interpolate(
            c, size=(x.shape[2], x.shape[3]), mode='bilinear'
        )
        x = x + c
        return x, c

class SecretDecoder(nn.Module):
    def __init__(self, output_size=64):
        super(SecretDecoder, self).__init__()
        self.output_size = output_size
        self.model = efficientnet.efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V1)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, output_size * 2, bias=True)

    def forward(self, x):
        x = F.interpolate(
            x, size=(512, 512), mode='bilinear'
        )
        decoded = self.model(x).view(-1, self.output_size, 2)
        return decoded

class MyDataset(torch.utils.data.Dataset):
    """
    Custom dataset for loading images and captions from a JSON file.
    Expected JSON format:
    [
        [{"image_file": "path/to/image.jpg", "text": "caption"}],
        ...
    ]
    """
    def __init__(self, json_file, tokenizer, tokenizer_2, size=1024, center_crop=True, t_drop_rate=0.05,
                 i_drop_rate=0.05, ti_drop_rate=0.05, image_root_path=""):
        super().__init__()
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.size = size
        self.center_crop = center_crop
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate
        self.image_root_path = image_root_path

        self.data = json.load(open(json_file))

        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        self.clip_image_processor = CLIPImageProcessor()

    def __getitem__(self, idx):
        item = self.data[idx][0]
        text = item["text"]
        image_file = item["image_file"]

        image_path = os.path.join(self.image_root_path, image_file)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        raw_image = Image.open(image_path)

        original_width, original_height = raw_image.size
        original_size = torch.tensor([original_height, original_width])

        image_tensor = self.transform(raw_image.convert("RGB"))
        delta_h = image_tensor.shape[1] - self.size
        delta_w = image_tensor.shape[2] - self.size

        if self.center_crop:
            top = delta_h // 2
            left = delta_w // 2
        else:
            top = np.random.randint(0, delta_h + 1)
            left = np.random.randint(0, delta_w + 1)

        image = transforms.functional.crop(
            image_tensor, top=top, left=left, height=self.size, width=self.size
        )
        crop_coords_top_left = torch.tensor([top, left])

        clip_image = self.clip_image_processor(images=raw_image, return_tensors="pt").pixel_values

        # Conditional dropout
        drop_image_embed = 0
        rand_num = random.random()
        if rand_num < self.i_drop_rate:
            drop_image_embed = 1
        elif rand_num < (self.i_drop_rate + self.t_drop_rate):
            text = ""
        elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
            text = ""
            drop_image_embed = 1
        if not text:
            text = "A lone traveler walking through a misty forest at dawn, the sun's rays barely breaking through the dense trees, creating a magical atmosphere."

        text_input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids

        text_input_ids_2 = self.tokenizer_2(
            text,
            max_length=self.tokenizer_2.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids

        return {
            "image": image,
            "text": text,
            "text_input_ids": text_input_ids,
            "text_input_ids_2": text_input_ids_2,
            "clip_image": clip_image,
            "drop_image_embed": drop_image_embed,
            "original_size": original_size,
            "crop_coords_top_left": crop_coords_top_left,
            "target_size": torch.tensor([self.size, self.size]),
        }

    def __len__(self):
        return len(self.data)

def collate_fn(data):
    images = torch.stack([example["image"] for example in data])
    texts = [example["text"] for example in data]
    text_input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)
    text_input_ids_2 = torch.cat([example["text_input_ids_2"] for example in data], dim=0)
    clip_images = torch.cat([example["clip_image"] for example in data], dim=0)
    drop_image_embeds = [example["drop_image_embed"] for example in data]
    original_size = torch.stack([example["original_size"] for example in data])
    crop_coords_top_left = torch.stack([example["crop_coords_top_left"] for example in data])
    target_size = torch.stack([example["target_size"] for example in data])

    return {
        "images": images,
        "texts": texts,
        "text_input_ids": text_input_ids,
        "text_input_ids_2": text_input_ids_2,
        "clip_images": clip_images,
        "drop_image_embeds": drop_image_embeds,
        "original_size": original_size,
        "crop_coords_top_left": crop_coords_top_left,
        "target_size": target_size,
    }

def main():
    parser = argparse.ArgumentParser(description="Fine-tuning script for Stable Diffusion with LoRA and Steganography.")
    parser.add_argument("--cfg", type=str, default="cfg/mountain.json", help="Path to the training configuration JSON file.")
    parser.add_argument("--secret_models_path", type=str, required=True,
                        help="Path to the pretrained secret encoder/decoder state dict (.pth file from Stage 1).")
    parser.add_argument("--data_dir", type=str, default=None, help="Dataset directory path.")
    parser.add_argument("--data_root_path", type=str, default=None, help="Root directory of training images.")
    parser.add_argument("--data_test_path", type=str, default=None, help="Validation image directory path.")
    parser.add_argument("--data_json_file", type=str, default=None, help="Path to dataset annotation json.")
    args = parser.parse_args()

    cfg = load_training_config(args.cfg)
    if args.data_dir is not None:
        cfg.data_dir = args.data_dir
    if args.data_root_path is not None:
        cfg.data_root_path = args.data_root_path
    if args.data_test_path is not None:
        cfg.data_test_path = args.data_test_path
    if args.data_json_file is not None:
        cfg.data_json_file = args.data_json_file
    logging_dir = Path(cfg.log_dir)

    config = OmegaConf.load(cfg.config)
    if cfg.secret_len <= 0:
        cfg.secret_len = config.model.params.control_config.params.secret_len
    config.model.params.control_config.params.secret_len = cfg.secret_len
    config.model.params.loss_config.params.max_image_weight_ratio = cfg.max_image_weight_ratio

    accelerator_project_config = ProjectConfiguration(project_dir=cfg.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
        log_with="tensorboard",
        project_config=accelerator_project_config,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        import diffusers
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        import diffusers
        diffusers.utils.logging.set_verbosity_error()

    if cfg.seed is not None:
        set_seed(cfg.seed)

    if accelerator.is_main_process:
        if cfg.output_dir is not None:
            os.makedirs(cfg.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="unet")
    unetB = UNet2DConditionModel.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="unet")

    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unetB.requires_grad_(False)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    unet_lora_config = LoraConfig(
        r=cfg.rank,
        lora_alpha=2 * cfg.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0", "conv*", "proj", "proj_in"],
    )

    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    unetB.to(accelerator.device, dtype=weight_dtype)

    unet.add_adapter(unet_lora_config)

    if cfg.mixed_precision == "fp16":
        for param in unet.parameters():
            if param.requires_grad:
                param.data = param.to(torch.float32)

    if cfg.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. Please update to at least 0.0.17."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            logger.warning("xformers is not available; continuing without memory efficient attention.")

    # Fix: use list() so lora_layers is not exhausted after optimizer construction
    lora_layers = list(filter(lambda p: p.requires_grad, unet.parameters()))
    optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(
        lora_layers,
        lr=cfg.learning_rate_lora,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        weight_decay=cfg.adam_weight_decay,
        eps=cfg.adam_epsilon,
    )

    train_dataset = MyDataset(cfg.data_json_file, tokenizer=tokenizer, tokenizer_2=tokenizer,
                              size=cfg.resolution, image_root_path=cfg.data_root_path)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=cfg.train_batch_size,
        num_workers=cfg.dataloader_num_workers,
    )

    # Load pretrained watermark encoder/decoder from Stage 1 checkpoint
    pretrain_dict = torch.load(args.secret_models_path)
    Encoder = SecretEncoder(cfg.secret_len).to(accelerator.device, dtype=weight_dtype)
    Encoder.load_state_dict(pretrain_dict['sec_encoder'])
    Encoder.requires_grad_(False)

    Decoder = SecretDecoder(cfg.secret_len).to(accelerator.device, dtype=weight_dtype)
    Decoder.load_state_dict(pretrain_dict['sec_decoder'])
    Decoder.requires_grad_(False)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)
    max_train_steps = cfg.n_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        cfg.lr_scheduler_name,
        optimizer=optimizer
    )

    unet, unetB, Encoder, Decoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, unetB, Encoder, Decoder, optimizer, train_dataloader, lr_scheduler
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    total_batch_size = cfg.train_batch_size * accelerator.num_processes * cfg.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {cfg.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    global_step = 0
    first_epoch = 0

    if cfg.resume_from_checkpoint:
        if cfg.resume_from_checkpoint != "latest":
            path = os.path.basename(cfg.resume_from_checkpoint)
        else:
            dirs = os.listdir(cfg.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(f"Checkpoint '{cfg.resume_from_checkpoint}' does not exist. Starting a new training run.")
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(cfg.output_dir, path))
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    # Optional: load pretrained LoRA weights (set cfg.pretrained_lora_path in training config)
    if getattr(cfg, 'pretrained_lora_path', None):
        load_lora_parameters(unet, cfg.pretrained_lora_path)

    # Optional: load optimizer/scheduler state (set cfg.optimizer_checkpoint_path in training config)
    if getattr(cfg, 'optimizer_checkpoint_path', None):
        optimizer, lr_scheduler, first_epoch, initial_global_step, _ = load_checkpoint(
            optimizer, lr_scheduler, cfg.optimizer_checkpoint_path
        )
        global_step = initial_global_step
        first_epoch = first_epoch + 1

    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, num_train_epochs):
        train_loss = 0.0
        for batch_step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                with torch.no_grad():
                    latents = vae.encode(batch["images"].to(dtype=weight_dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    encoder_hidden_states = text_encoder(batch["text_input_ids"].to(accelerator.device))[0]

                # Generate secret data
                secret_data = torch.randint(0, 2, (latents.shape[0], cfg.secret_len)).to(accelerator.device)
                _, eps = Encoder(latents, secret_data.float())
                eps = eps * vae.config.scaling_factor

                # Sample noise
                noise = torch.randn_like(latents)
                if cfg.noise_offset:
                    noise += cfg.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )

                # Only train in the last 1/5 of timesteps (low-noise region, small t),
                # aligned with inference which only injects watermark in the final denoising steps.
                t_max = noise_scheduler.config.num_train_timesteps // 5  # e.g. 200 for 1000-step scheduler
                t = torch.randint(0, t_max, (1,), device=latents.device).item()
                timesteps = torch.full((latents.shape[0],), t, device=latents.device, dtype=torch.long)

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                latent_model_input = noise_scheduler.scale_model_input(noisy_latents, timestep=timesteps)

                # Teacher model prediction (frozen unetB)
                target = unetB(latent_model_input, timesteps, encoder_hidden_states).sample.detach()

                # Student model prediction with watermark injection
                unetoutput = unet(latent_model_input, timesteps, encoder_hidden_states=encoder_hidden_states,
                                  down_intrablock_additional_residuals=eps)
                model_pred0 = unetoutput.sample

                mean_loss = F.mse_loss(model_pred0.float(), target.float(), reduction='mean')

                
                model_pred_step = noise_scheduler.step(model_pred0, t, latent_model_input).prev_sample
                model_pred_step = model_pred_step * (1 / vae.config.scaling_factor)
                vae_param = next(vae.parameters())
                model_pred_step = model_pred_step.to(device=vae_param.device, dtype=vae_param.dtype)
                image_recw = vae.decode(model_pred_step).sample

                decoded_message = Decoder(image_recw)

                labels = F.one_hot(secret_data.long(), num_classes=2).float()
                secret_loss = F.binary_cross_entropy_with_logits(decoded_message, labels)

                loss = mean_loss + secret_loss

                decoded_messages = torch.argmax(decoded_message, dim=2)
                bit_acc = 1 - torch.abs(decoded_messages - secret_data).sum().float() / (cfg.secret_len * cfg.train_batch_size)

                avg_loss = accelerator.gather(loss.repeat(cfg.train_batch_size)).mean()
                train_loss += avg_loss.item() / cfg.gradient_accumulation_steps

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(lora_layers, cfg.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                accelerator.log({"bit_acc": bit_acc}, step=global_step)
                train_loss = 0.0

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if accelerator.sync_gradients and global_step > 0 and global_step % cfg.checkpointing_steps == 0:
                if accelerator.is_main_process:
                    save_path = os.path.join(cfg.output_dir, f'checkpoint-{global_step}')
                    os.makedirs(save_path, exist_ok=True)
                    save_lora_parameters(unet, os.path.join(save_path, 'lora_weights.pth'))
                    save_checkpoint(optimizer, lr_scheduler, epoch, global_step, loss,
                                    os.path.join(save_path, 'training_state.pth'))

            if global_step >= max_train_steps:
                break

    if accelerator.is_main_process and global_step > 0:
        save_path = os.path.join(cfg.output_dir, f'checkpoint-{global_step}')
        os.makedirs(save_path, exist_ok=True)
        save_lora_parameters(unet, os.path.join(save_path, 'lora_weights.pth'))
        save_checkpoint(optimizer, lr_scheduler, num_train_epochs - 1, global_step, loss,
                        os.path.join(save_path, 'training_state.pth'))

    accelerator.end_training()

if __name__ == "__main__":
    main()
