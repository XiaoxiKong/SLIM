import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import os
import random
import glob
import PIL
from PIL import Image
from tqdm.auto import tqdm
import numpy as np
import argparse

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torch.utils.checkpoint
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
import gc
import transformers
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    UNet2DConditionModel,
)
import lpips
import matplotlib.pyplot as plt
from utils_f.models import *
from utils_f.misc import torch_to_pil
from utils_f.noise_layers.noiser import Noiser
import torchsummary

def rand_rotation_matrix(deflection=1.0, randnums=None):
    """
    Creates a random rotation matrix.

    deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random
    rotation. Small deflection => small perturbation.
    randnums: 3 random numbers in the range [0, 1]. If `None`, they will be auto-generated.
    """
    # from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c

    if randnums is None:
        randnums = np.random.uniform(size=(3,))

    theta, phi, z = randnums

    theta = theta * 2.0*deflection*np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0*np.pi  # For direction of pole deflection.
    z = z * 2.0*deflection  # For magnitude of pole deflection.

    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.

    r = np.sqrt(z)
    Vx, Vy, Vz = V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
        )

    st = np.sin(theta)
    ct = np.cos(theta)
    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))

    # Construct the rotation matrix  ( V Transpose(V) - I ) R.
    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M

def sigmoid(x):
    return 1 - 1 / (1 + torch.exp(-35*x))

def get_cdf(x_samples, full_range):
    return sigmoid((x_samples[:,None]-full_range[None,:])).sum(0)/len(x_samples)


def sliced_wasserstein(loss, pixels_gen, pixels_ref, num_slices=10, use_differenatable_historgam_matching=False):
        device = pixels_gen.device
        for slice_idx in range(num_slices):
            R_trans = rand_rotation_matrix(deflection=1.0)
            R_trans = torch.Tensor(R_trans)
            R_trans = R_trans.to(device).to(pixels_gen.dtype)
            pixels_gen_rotated = pixels_gen.T@R_trans
            pixels_ref_rotated = pixels_ref.T@R_trans
            for dim_idx in range(3):
                x_samples_3d_rotated_slice = pixels_gen_rotated[:,dim_idx]
                y_samples_3d_rotated_slice = pixels_ref_rotated[:,dim_idx]

                if not use_differenatable_historgam_matching:
                    rand_idxes_len = min(len(x_samples_3d_rotated_slice), len(y_samples_3d_rotated_slice))
                    rand_idxes = np.random.randint(0, rand_idxes_len, rand_idxes_len)
                    x_samples_3d_rotated_slice = x_samples_3d_rotated_slice[rand_idxes]
                    y_samples_3d_rotated_slice = y_samples_3d_rotated_slice[rand_idxes]

                    x_samples_3d_rotated_slice = torch.sort(x_samples_3d_rotated_slice).values
                    y_samples_3d_rotated_slice = torch.sort(y_samples_3d_rotated_slice).values
                    loss += torch.mean(torch.abs(x_samples_3d_rotated_slice-y_samples_3d_rotated_slice))
                else:
                    min_range = min(
                        x_samples_3d_rotated_slice.min().item(),
                        y_samples_3d_rotated_slice.min().item()
                    )
                    max_range = max(
                        x_samples_3d_rotated_slice.max().item(),
                        y_samples_3d_rotated_slice.max().item()
                    )
                    grid_size = 400
                    full_range = torch.linspace(min_range-0.05, max_range+0.05, grid_size)
                    full_range = full_range.to(device)
                    x_cdf = get_cdf(x_samples_3d_rotated_slice, full_range)
                    y_cdf = get_cdf(y_samples_3d_rotated_slice, full_range)
                    loss += torch.mean(torch.abs(x_cdf-y_cdf))
        # Fix: each slice accumulates 3 dims, so normalize by num_slices * 3
        return loss / (num_slices * 3)


# ---------------- Explicit XL pipeline forward call
class LatentDistributionLosses:
    """
    提供不同的潜在空间分布一致性损失函数，用于保证水印编码的latent B与
    Stable Diffusion的标准正态分布潜在空间保持一致
    """

    @staticmethod
    def kl_divergence_loss(latent_B):
        """
        计算latent B与标准正态分布N(0, I)之间的KL散度

        Args:
            latent_B (torch.Tensor): 水印编码的潜在向量

        Returns:
            torch.Tensor: KL散度损失值
        """
        batch_size = latent_B.shape[0]
        mean_B = torch.mean(latent_B, dim=[1, 2, 3])
        var_B = torch.var(latent_B, dim=[1, 2, 3], unbiased=False)

        mean_target = torch.zeros_like(mean_B)
        var_target = torch.ones_like(var_B)

        # KL(N(μ_B, σ²_B) || N(0, 1))
        kl_div = torch.log(torch.sqrt(var_target) / torch.sqrt(var_B)) + \
                (var_B + (mean_B - mean_target)**2) / (2 * var_target) - 0.5

        return torch.mean(kl_div)

    @staticmethod
    def mean_var_loss(latent_B, mean_weight=1.0, var_weight=1.0):
        """
        计算更简单的均值-方差损失，确保latent B接近均值为0、方差为1的分布
        """
        mean_B = torch.mean(latent_B, dim=[1, 2, 3])
        var_B = torch.var(latent_B, dim=[1, 2, 3], unbiased=False)

        mean_loss = torch.mean(mean_B**2)
        std_B = torch.sqrt(var_B)
        var_loss = torch.mean((std_B - 1.0)**2)

        total_loss = mean_weight * mean_loss + var_weight * var_loss
        return total_loss

    @staticmethod
    def moment_matching_loss(latent_A, latent_B, num_moments=4):
        """
        通过匹配多阶矩来确保latent B与latent A具有相似的分布
        """
        loss = 0.0
        for k in range(1, num_moments + 1):
            moment_A = torch.mean(latent_A**k, dim=[1, 2, 3])
            moment_B = torch.mean(latent_B**k, dim=[1, 2, 3])
            loss += torch.mean((moment_A - moment_B)**2)
        return loss

    @staticmethod
    def wasserstein_loss(latent_A, latent_B, n_projections=1000, device='cuda'):
        """
        使用Wasserstein距离的近似估计来衡量两个分布的相似性
        """
        flat_A = latent_A.reshape(latent_A.size(0), -1)
        flat_B = latent_B.reshape(latent_B.size(0), -1)

        dim = flat_A.size(1)
        projections = torch.randn(n_projections, dim, device=device)
        projections = F.normalize(projections, dim=1)

        proj_A = torch.matmul(flat_A, projections.t())
        proj_B = torch.matmul(flat_B, projections.t())

        proj_A_sorted, _ = torch.sort(proj_A, dim=0)
        proj_B_sorted, _ = torch.sort(proj_B, dim=0)

        wasserstein_dist = torch.mean(torch.abs(proj_A_sorted - proj_B_sorted))
        return wasserstein_dist


def do_sw_guidance(device, latents, pixels_ref, decode_latents, sw_debug):
    """Compute sliced Wasserstein loss between generated and reference pixels."""
    loss = 0
    pixels_ref = (pixels_ref / 2 + 0.5).clamp(0, 1)
    pixels_ref = pixels_ref.squeeze(0).reshape(3, -1)

    image = decode_latents(latents)
    image = (image / 2 + 0.5).clamp(0, 1)
    pixels_gen = image.squeeze(0).reshape(3, -1)
    loss = sliced_wasserstein(loss, pixels_gen, pixels_ref)

    return latents, loss


WINDOW_SIZE = 32
KERNEL = torch.ones((1, 1, WINDOW_SIZE, WINDOW_SIZE), dtype=torch.float32) / (WINDOW_SIZE**2)

def PRVL_loss(img1, img2):
    global KERNEL
    diff = torch.abs(img1 - img2)
    diff_combined = torch.mean(diff, dim=1, keepdim=True)
    if KERNEL.device != diff_combined.device:
        KERNEL = KERNEL.to(diff_combined.device)
    diff_sum = F.conv2d(diff_combined, KERNEL, padding=WINDOW_SIZE//2).squeeze(0)
    max_diff = torch.max(diff_sum)
    return max_diff

def base_augment(image):
    if random.random() > 0.5:
        image = torch.flip(image, dims=[-1])
    image = torch.rot90(image, k=random.randint(0, 3), dims=[-2, -1])
    return image

class traindataset(Dataset):
    def __init__(self, root, random_aug=True):
        self.root = root
        self.image_files = glob.glob(root + "/*.png") + glob.glob(root + "/*.jpg")
        self.random_aug = random_aug

    def __len__(self):
        return len(self.image_files)

    def process(self, image):
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = image.resize((512, 512), resample=PIL.Image.Resampling.BICUBIC)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = torch.from_numpy(image).permute(2, 0, 1)
        if self.random_aug and random.random() > 0.5:
            image = base_augment(image)
        return image

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image = Image.open(img_name)
        image = self.process(image)
        return image


def main(args):

    train_loader = torch.utils.data.DataLoader(
        traindataset(args.dataset, args.random_aug),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    vae = vae.cuda()
    vae.requires_grad_(False)

    def decode_latents(latents):
        image = vae.decode(latents).sample
        return image

    sec_encoder = SecretEncoder(args.bit_num).cuda()
    sec_decoder = SecretDecoder(output_size=args.bit_num).cuda()
    torchsummary.summary(sec_decoder, (3, 512, 512))

    loss_fn_alex = lpips.LPIPS(net='vgg').cuda()
    loss_fn_alex.requires_grad_(False)

    noise_config = ['Identity','Jpeg','CropandResize','GaussianBlur','GaussianNoise','ColorJitter']
    posibilities = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    noiser = Noiser(noise_config, posibilities, device='cuda')

    current_epoch = 0
    if args.resume_from_ckpt is not None:
        print(f"Resuming from checkpoint: {args.resume_from_ckpt}")
        models = torch.load(args.resume_from_ckpt)
        sec_encoder.load_state_dict(models['sec_encoder'])
        sec_decoder.load_state_dict(models['sec_decoder'])
        current_epoch = int(args.resume_from_ckpt.split('_')[-1].split('.')[0])
        print(f"Resumed from epoch {current_epoch}")

    optimizer = optim.AdamW([
        {'params': sec_encoder.parameters()},
        {'params': sec_decoder.parameters()}
    ], lr=args.lr, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.8)

    writer = SummaryWriter(args.output_dir + '/logs')

    def gen_combined_latents(latents, wm_latent, scale=1.0):
        cornerfy_aug = random.choice([True, False, False, False])  # 1/4 chance

        height, width = wm_latent.shape[2], wm_latent.shape[3]
        height_scale, width_scale = (random.uniform(1.0, 2.0), random.uniform(1.0, 2.0)) if cornerfy_aug else (1.0, 1.0)
        if cornerfy_aug:
            wm_template = F.interpolate(torch.zeros_like(latents), scale_factor=(height_scale, width_scale), mode='bilinear')
            wm_template[:,:,:height//2, :width//2] =  wm_latent[:,:,:height//2, :width//2]
            wm_template[:,:,:height//2, -width//2:] = wm_latent[:,:,:height//2, -width//2:]
            wm_template[:,:,-height//2:,:width//2] =  wm_latent[:,:,-height//2:,:width//2]
            wm_template[:,:,-height//2:,-width//2:] = wm_latent[:,:,-height//2:,-width//2:]
            wm_template = F.interpolate(wm_template, size=(height, width), mode='bilinear')
        else:
            wm_template = wm_latent

        watermarked_latents = latents + wm_template * scale
        return watermarked_latents

    all_iter = args.epochs * len(train_loader)

    pbar = tqdm(total=all_iter)
    iterations = 0
    warmup = args.warmup
    fixinit = args.fixinit

    zero_batch = torch.zeros(args.batch_size, 3, 512, 512).cuda()
    for epoch in range(current_epoch, current_epoch + args.epochs):
        sec_encoder.train()
        sec_decoder.train()

        msgloss_10buffer = []
        for batch_idx, oimage in enumerate(train_loader):

            optimizer.zero_grad()
            oimage = oimage.cuda()
            latents = vae.encode(oimage).latent_dist.sample().detach()

            msg = torch.randint(0, 2, (args.batch_size, args.bit_num)).cuda()
            _, wm_latent = sec_encoder(latents, msg.float())

            if warmup:
                watermarked_latents = gen_combined_latents(latents, wm_latent, scale=0.03)
            else:
                watermarked_latents = gen_combined_latents(latents, wm_latent)
            clean_image = decode_latents(latents).detach()

            watermarked_latents, ws_loss = do_sw_guidance(
                watermarked_latents.device, watermarked_latents, clean_image,
                decode_latents=decode_latents, sw_debug=False
            )
            watermarked_image = decode_latents(watermarked_latents)
            lpips_loss = loss_fn_alex(clean_image, watermarked_image).mean()
            prvl_loss = PRVL_loss(clean_image, watermarked_image)
            

            if epoch - current_epoch > 12 or args.resume_from_ckpt is not None:
                watermarked_image = noiser([watermarked_image,None],[0.4, 0.1, 0.2, 0.05, 0.1, 0.15])[0]
            else:
                watermarked_image = noiser([watermarked_image,None],[0.6, 0., 0.4, 0., 0., 0.])[0]

            reveal_output = sec_decoder(watermarked_image)

            labels = F.one_hot(msg, num_classes=2).float()
            msgloss = F.binary_cross_entropy_with_logits(reveal_output, labels.cuda())

            if len(msgloss_10buffer) == 10:
                msgloss_10buffer.pop(0)
            msgloss_10buffer.append(msgloss.item())

            # when message loss < 0.1 for 10 consecutive batches, consider model warmed up
            if len(msgloss_10buffer) == 10 and sum(msgloss_10buffer) / 10 < 0.1:
                warmup = False
                

            if warmup:
                loss = msgloss
            else:
                if epoch - current_epoch > 10 or args.resume_from_ckpt is not None:
                    loss = lpips_loss * 5 + msgloss * 1.0 + ws_loss * 100
                elif epoch - current_epoch > 6:
                    loss = lpips_loss + msgloss
                else:
                    loss = msgloss

            loss.backward()
            optimizer.step()

            pbar.update(1)
            iterations += 1
            print("lpips_loss: %.4f, msgloss: %.4f, prvl_loss: %.4f, ws_loss: %.4f, loss: %.4f" % (lpips_loss, msgloss, prvl_loss, ws_loss, loss))

            writer.add_scalar('Loss/train', loss, iterations)
            writer.add_scalar('Loss/lpips_loss', lpips_loss, iterations)
            writer.add_scalar('Loss/prvl_loss', prvl_loss, iterations)
            writer.add_scalar('Loss/msgloss', msgloss, iterations)
            writer.add_scalar('Loss/ws_loss', ws_loss, iterations)

        watermarked_image_pil = torch_to_pil(watermarked_image)[0]
        watermarked_image_pil.save(f"{args.output_dir}/log_images/watermarked_{epoch}_{batch_idx}.png")

        # Validation
        sec_encoder.eval()
        sec_decoder.eval()
        with torch.no_grad():
            msg_val = torch.randint(0, 2, (args.batch_size, args.bit_num)).cuda()
            # Fix: use same embedding logic as training (extract wm_latent, then combine)
            _, wm_latent_val = sec_encoder(latents, msg_val.float())
            watermarked_latent_val = latents + wm_latent_val  # scale=1.0, no augmentation for eval
            watermarked = decode_latents(watermarked_latent_val)
            decoded_msg = sec_decoder(watermarked)
            decoded_msg = torch.argmax(decoded_msg, dim=2)
            acc = 1 - torch.abs(decoded_msg - msg_val).sum().float() / (args.bit_num * args.batch_size)
            print(f"Epoch {epoch}: acc {acc}")
            writer.add_scalar('Accuracy/train', acc, epoch)

        print(f"Epoch {epoch}: loss {loss}, lpips_loss {lpips_loss}, msgloss {msgloss}, prvl_loss {prvl_loss}")

        scheduler.step()

        torch.save({
            'sec_decoder': sec_decoder.state_dict(),
            'sec_encoder': sec_encoder.state_dict(),
        }, f"{args.output_dir}/checkpoints/state_dict_{epoch}.pth")

    writer.close()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--pretrained_model_name_or_path', type=str, required=True,
                           help='Path to pretrained Stable Diffusion model (e.g. runwayml/stable-diffusion-v1-5)')
    argparser.add_argument('--dataset', type=str, required=True,
                           help='Path to training image directory')
    argparser.add_argument('--output_dir', type=str, default='checkpoints-slim-stage1')
    argparser.add_argument('--epochs', type=int, default=40)
    argparser.add_argument('--batch_size', type=int, default=1)
    argparser.add_argument('--bit_num', type=int, default=48)
    argparser.add_argument('--resume_from_ckpt', type=str, default=None)
    argparser.add_argument('--warmup', action=argparse.BooleanOptionalAction, default=True,
                           help='Use warmup phase (use --no-warmup to disable)')
    argparser.add_argument('--fixinit', action=argparse.BooleanOptionalAction, default=True,
                           help='Use zero image at start of training (use --no-fixinit to disable)')
    argparser.add_argument('--random_aug', action=argparse.BooleanOptionalAction, default=True,
                           help='Use random augmentation (use --no-random_aug to disable)')
    argparser.add_argument('--lr', type=float, default=1e-3)
    args = argparser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        os.makedirs(args.output_dir + '/logs')
        os.makedirs(f'{args.output_dir}/checkpoints')
        os.makedirs(f'{args.output_dir}/log_images')

    main(args)
