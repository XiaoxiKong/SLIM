import torch
import argparse
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from utils import *
import numpy as np
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
import torchvision.models.efficientnet as efficientnet
from torchvision.models.efficientnet import EfficientNet_B1_Weights
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from custom_pipeline import WatermarkPipeline
def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
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
        # We assume x has shape (N, C, H, W) and sizes is (H', W')
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
            # chx16x16 -> chx256x256
            zero_module(conv_nd(2, 4, 4, 3, padding=1))
        )  # secret len -> ch x res x res

    def encode(self, x):
        x = self.secret_scaler(x)
        return x

    def forward(self, x, c):
        # x: [B, C, H, W], c: [B, secret_len]
        c = self.encode(c)
        c = F.interpolate(
            c, size=(64, 64), mode='bilinear'
        )
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

def main(args):
    # load diffusion model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_id, subfolder='scheduler')
    pipe = WatermarkPipeline.from_pretrained(
        args.model_id,
        scheduler=scheduler,
        torch_dtype=torch.float16,
        safety_checker=None
        )
    pipe = pipe.to(device)
    
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        pretrain_dict = torch.load(args.checkpoint_path)
        Encoder = SecretEncoder(args.bit_num).to(device, dtype=torch.float16)
        Encoder.load_state_dict(pretrain_dict['sec_encoder'])
        Decoder = SecretDecoder(args.bit_num).to(device, dtype=torch.float16)
        Decoder.load_state_dict(pretrain_dict['sec_decoder'])
    else:
        print("Warning: Checkpoint path not provided or does not exist. Initializing random weights (for testing only).")
        Encoder = SecretEncoder(args.bit_num).to(device, dtype=torch.float16)
        Decoder = SecretDecoder(args.bit_num).to(device, dtype=torch.float16)

    if args.prompt_file and os.path.exists(args.prompt_file):
        dataset, prompt_key = get_dataset(args.prompt_file)
    else:
        print("Using default prompts list.")
        # Minimal default prompts
        dataset = [
            "A cartoon character with vibrant colors, wearing a magical costume, surrounded by floating sparkles.",
            "A serene landscape featuring a sunset over rolling hills, with flowers blooming in the foreground.",
            "A whimsical forest filled with fairy tale creatures, bright mushrooms, and colorful foliage."
        ]
        prompt_key = None
    
    w_dir = args.output_dir
    os.makedirs(w_dir, exist_ok=True)
    acc_all = []
    
    # Standard torchvision ToPILImage doesn't handle the [-1, 1] -> [0, 1] conversion we need for diffusion outputs
    # so we'll do it manually like the original script
    to_pil = transforms.ToPILImage()

    # Determine loop range based on dataset size
    num_samples = min(args.end, len(dataset))
    start_idx = min(args.start, num_samples)

    if start_idx >= num_samples:
        print("Start index is out of bounds or dataset is empty.")
        return

    for i in tqdm(range(start_idx, num_samples)):
        secret_data = torch.randint(0, 2, (1, args.bit_num), device=device, dtype=torch.float16)
        _, eps = Encoder(torch.zeros(1, 4, 64, 64, device=device, dtype=torch.float16), secret_data)

        seed = args.seed if args.seed is not None else torch.randint(0, 2**32 - 1, (1,)).item()
        
        if prompt_key:
            current_prompt = dataset[i][prompt_key]
        else:
            current_prompt = dataset[i]
            
        print(f"Generating: {current_prompt}")

        ### generation
        outputs_w = pipe(
            current_prompt,
            num_images_per_prompt=args.num_images,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            height=args.image_length,
            width=args.image_length,
            generator=torch.Generator(device=device).manual_seed(seed),
            eps=eps*0.18215,
            output_type="pt", #original output_type="pt", 
            impuwater = 1,  # Custom parameter for watermarking
            lora_path = args.lora_path,  # Path to LoRA weights
            )
        orig_image_w = outputs_w.images[0]  # Tensor [C, H, W] in range [0, 1] (already denormalized by VaeImageProcessor)
        
        # Save image - clamp to valid range only (postprocess already handled [-1,1] -> [0,1])
        img_clamped = torch.clamp(orig_image_w, 0.0, 1.0)
        pil_img = to_pil(img_clamped)
        save_path = os.path.join(w_dir, f'sample_{i}.png')
        pil_img.save(save_path)

        # Verify watermark from saved image 
        # Read back image
        img_read = Image.open(save_path).convert('RGB')
        # Convert to tensor [0, 1]
        img_read_tensor = transforms.ToTensor()(img_read).to(device, dtype=torch.float16)
        # Normalize to [-1, 1] for the decoder
        img_read_tensor = img_read_tensor * 2.0 - 1.0
        
        decoded_message = Decoder(img_read_tensor.unsqueeze(0))
        decoded_messages = torch.argmax(decoded_message, dim=2)
        acc = 1 - torch.abs(decoded_messages - secret_data).sum().float() / (args.bit_num * 1)
        acc_all.append(acc.cpu())

    if acc_all:
        print(f"Mean Accuracy: {np.mean(acc_all)}")
    else:
        print("No images generated.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='diffusion watermark')
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=5, type=int)
    parser.add_argument('--image_length', default=512, type=int)
    parser.add_argument('--model_id', default="D:/code/stable-diffusion-v1-5",
                        help="Path to local model directory or HuggingFace model ID")
    # parser.add_argument('--checkpoint_path', default="D:/code/DiffusersExample-main/LoRA/ckpt/412/state_dict_39.pth",
    #                     help="Path to Stage 1 checkpoint (.pth file with 'sec_encoder' and 'sec_decoder' keys)")
    parser.add_argument('--checkpoint_path', default="D:/code/DiffusersExample-main/LoRA/ckpt/412/both_backbone/pretained_latentwm.pth",
                        help="Path to Stage 1 checkpoint (.pth file with 'sec_encoder' and 'sec_decoder' keys)")
    # parser.add_argument('--lora_path', default="D:/code/DiffusersExample-main/LoRA/ckpt/412/SD1.5_LORA50_WS_512_48.pth",
    #                     help="Path to LoRA weights for the watermarked UNet (.pth file)")
    parser.add_argument('--lora_path', default="D:\code\DiffusersExample-main\LoRA\ckpt\\412\\both_backbone\\both20.pth",
                        help="Path to LoRA weights for the watermarked UNet (.pth file)")
    parser.add_argument('--output_dir', default="output", help="Directory to save generated images")
    parser.add_argument('--num_images', default=1, type=int)
    parser.add_argument('--guidance_scale', default=7.5, type=float)
    parser.add_argument('--num_inference_steps', default=25, type=int)
    parser.add_argument('--prompt_file', default='D:\Image\\annotations\\transformed_data.json',
                        help="Path to prompt file (.json) or HuggingFace dataset name. If not provided, uses built-in demo prompts.")
    parser.add_argument('--bit_num', default=48, type=int,
                        help="Number of secret bits to embed in the watermark (must match the trained checkpoint)")
    parser.add_argument('--seed', default=None, type=int,
                        help="Random seed for image generation. If not set, a random seed is used per image.")

    args = parser.parse_args()

    import time
    start_time = time.time()
    main(args)
    end_time = time.time()
    print(f"RUNNING TIME: {end_time - start_time:.4f}s")
