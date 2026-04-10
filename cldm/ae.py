import numpy as np
import einops
import torch
import torch as th
import torch.nn as nn
from torch.nn import functional as thf
import pytorch_lightning as pl
import torchvision
from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)
from contextlib import contextmanager, nullcontext
from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config, default
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.modules.ema import LitEma
from ldm.modules.distributions.distributions import normal_kl, DiagonalGaussianDistribution
from ldm.modules.diffusionmodules.model import Encoder
import lpips
from kornia import color
import torch.nn.functional as F
try:
    from mmcv.cnn import ConvModule, Scale
except ImportError:
    class Scale(nn.Module):
        def __init__(self, scale=1.0):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(float(scale)))

        def forward(self, x):
            return x * self.scale

    class ConvModule(nn.Sequential):
        def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            bias=True,
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=None,
            **kwargs,
        ):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)]
            if norm_cfg:
                norm_type = norm_cfg.get("type", "").upper()
                if norm_type == "BN":
                    layers.append(nn.BatchNorm2d(out_channels))
                elif norm_type == "GN":
                    num_groups = norm_cfg.get("num_groups", 32)
                    layers.append(nn.GroupNorm(num_groups=num_groups, num_channels=out_channels))
            if act_cfg:
                act_type = act_cfg.get("type", "").upper()
                if act_type == "RELU":
                    layers.append(nn.ReLU(inplace=act_cfg.get("inplace", True)))
                elif act_type == "LEAKYRELU":
                    layers.append(nn.LeakyReLU(inplace=act_cfg.get("inplace", True)))
                elif act_type == "SILU":
                    layers.append(nn.SiLU())
            super().__init__(*layers)
import torch
import torch.nn as nn
from cldm.two_way_transformer import TwoWayTransformer, PositionEmbeddingRandom
import torch.nn.functional as F


class AttentionBlock(nn.Module):
    def __init__(self, dim):
        super(AttentionBlock, self).__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v):
        query = self.query(q)
        key = self.key(k)
        value = self.value(v)

        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / query.size(-1) ** 0.5
        attn_weights = self.softmax(attn_scores)

        return torch.matmul(attn_weights, value)


class FeatureProcessingBlock(nn.Module):
    def __init__(self, input_dim=4, conv_out_channels=4):
        super(FeatureProcessingBlock, self).__init__()
        self.self_attention = AttentionBlock(input_dim)
        self.cross_attention1 = AttentionBlock(input_dim)
        self.cross_attention2 = AttentionBlock(input_dim)
        self.conv = nn.Conv2d(input_dim, conv_out_channels, kernel_size=3, padding=1)

    def forward(self, f, v):
        bs, c, h, w = f.shape

        # Flatten the spatial dimensions for attention
        f_flat = f.view(bs, c, -1).permute(0, 2, 1)  # (bs, h*w, c)
        v_flat = v.view(bs, c, -1).permute(0, 2, 1)  # (bs, h*w, c)

        # Self-attention on f
        f_self_attended = self.self_attention(f_flat, f_flat, f_flat)

        # Cross-attention with f_self_attended as query and v as key/value
        f1_flat = self.cross_attention1(f_self_attended, v_flat, v_flat)

        # Reshape f1 back to (bs, c, h, w) for convolution
        f1 = f1_flat.permute(0, 2, 1).view(bs, c, h, w)

        # Apply convolution
        f2 = self.conv(f1)

        # Flatten f2 for second cross-attention
        f2_flat = f2.view(bs, f2.size(1), -1).permute(0, 2, 1)  # (bs, h*w, conv_out_channels)

        # Cross-attention with v as query and f2 as key/value
        v1_flat = self.cross_attention2(v_flat, f2_flat, f2_flat)

        # Reshape v1 back to (bs, c, h, w)
        v1 = v1_flat.permute(0, 2, 1).view(bs, c, h, w)

        return f2, v1



class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()

        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x



class PAM(nn.Module):
    """Position Attention Module (PAM)
    Args:
        in_channels (int): Input channels of key/query feature.
        channels (int): Output channels of key/query transform.
    """

    def __init__(self, in_channels, channels):
        super().__init__(
            key_in_channels=in_channels,
            query_in_channels=in_channels,
            channels=channels,
            out_channels=in_channels,
            share_key_query=False,
            query_downsample=None,
            key_downsample=None,
            key_query_num_convs=1,
            key_query_norm=False,
            value_out_num_convs=1,
            value_out_norm=False,
            matmul_norm=False,
            with_out=False,
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=None)

        self.gamma = Scale(0)

    def forward(self, x):
        """Forward function."""
        out = super().forward(x, x)

        out = self.gamma(out) + x
        return out


class CAM(nn.Module):
    """Channel Attention Module (CAM)"""

    def __init__(self):
        super().__init__()
        self.gamma = Scale(0)

    def forward(self, x):  # (8,512,60,60)
        """Forward function."""
        batch_size, channels, height, width = x.size()
        proj_query = x.view(batch_size, channels, -1)  # (8,512,3600)
        proj_key = x.view(batch_size, channels, -1).permute(0, 2, 1)  # (8,3600,512)
        energy = torch.bmm(proj_query, proj_key)  # (8,512,512), 每个通道hw的特征图和所有其它通道的特征图求点积
        energy_new = torch.max(
            energy, -1, keepdim=True)[0].expand_as(
            energy) - energy  # (8,512,1)->(8,512,512) - (8,512,512), 文章中好像没有这一步，每个注意力都减去最大注意力值
        attention = F.softmax(energy_new, dim=-1)  # (8,512,512), 每个通道和所有其它通道注意力的和为1,值越大说明这两个通道之间的关联性越大
        proj_value = x.view(batch_size, channels, -1)  # (8,512,3600)

        out = torch.bmm(attention, proj_value)  # (8,512,3600)
        # attention是512x512的每一行和为1，表示一个通道和其它所有通道的注意力（相关性）
        # 以attention的第一行与proj_value的第一列求点积得到out第一个点[0,0]的值为例，表示proj_value中所有通道[0,0]点的值根据该通道与其它所有通道的注意力加权求和。
        # 输出out中某个通道某个像素点的值等于proj_value中这个点所有通道的值根据这个通道与其它所有通道的注意力进行加权求和得到。
        out = out.view(batch_size, channels, height, width)  # (8,512,60,60)

        out = self.gamma(out) + x
        return out

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class View(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)

class SEnet(nn.Module):
    def __init__(self,channels=320,ratio=16):
        super(SEnet, self).__init__()
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        # 经过两次全连接层，一次较小，一次还原
        self.fc=nn.Sequential(
            nn.Linear(channels,channels//ratio,False),
            nn.ReLU(),
            nn.Linear(channels//ratio, channels, False),
            nn.Sigmoid()
        )
    def forward(self,x,y):
        b,c,_,_=x.size() #取出batch size和通道数
        # b,c,w,h->b,c,1,1->b,c 以便进行全连接
        avg=self.avgpool(x).view(b,c)
        #b,c->b,c->b,c,1,1 以便进行线性加权
        fc=self.fc(avg).view(b,c,1,1)

        return fc*y

class DW_Encoder(nn.Module):

    def __init__(self, message_length, blocks=2, channels=64, attention=None):
        super(DW_Encoder, self).__init__()

        self.conv1 = ConvBlock(3, 16, blocks=blocks)
        self.down1 = Down(16, 32, blocks=blocks)
        self.down2 = Down(32, 64, blocks=blocks)
        self.down3 = Down(64, 128, blocks=blocks)

        self.down4 = Down(128, 256, blocks=blocks)

        self.up3 = UP(256, 128)
        self.linear3 = nn.Linear(message_length, message_length * message_length)
        self.Conv_message3 = ConvBlock(1, channels, blocks=blocks)
        self.att3 = ResBlock(128 * 2 + channels, 128, blocks=blocks, attention=attention)

        self.up2 = UP(128, 64)
        self.linear2 = nn.Linear(message_length, message_length * message_length)
        self.Conv_message2 = ConvBlock(1, channels, blocks=blocks)
        self.att2 = ResBlock(64 * 2 + channels, 64, blocks=blocks, attention=attention)

        self.up1 = UP(64, 32)
        self.linear1 = nn.Linear(message_length, message_length * message_length)
        self.Conv_message1 = ConvBlock(1, channels, blocks=blocks)
        self.att1 = ResBlock(32 * 2 + channels, 32, blocks=blocks, attention=attention)

        self.up0 = UP(32, 16)
        self.linear0 = nn.Linear(message_length, message_length * message_length)
        self.Conv_message0 = ConvBlock(1, channels, blocks=blocks)
        self.att0 = ResBlock(16 * 2 + channels, 16, blocks=blocks, attention=attention)

        self.Conv_1x1 = nn.Conv2d(16 + 3, 3, kernel_size=1, stride=1, padding=0)

        self.message_length = message_length

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])


    def forward(self, feature, watermark):
        d0 = self.conv1(watermark)

        u0 = torch.cat((d0, u0), dim=1)
        u0 = self.att0(u0)

        image = self.Conv_1x1(torch.cat((x, u0), dim=1))

        forward_image = image.clone().detach()
        '''read_image = torch.zeros_like(forward_image)

        for index in range(forward_image.shape[0]):
            single_image = ((forward_image[index].clamp(-1, 1).permute(1, 2, 0) + 1) / 2 * 255).add(0.5).clamp(0, 255).to('cpu', torch.uint8).numpy()
            im = Image.fromarray(single_image)
            read = np.array(im, dtype=np.uint8)
            read_image[index] = self.transform(read).unsqueeze(0).to(image.device)

        gap = read_image - forward_image'''
        gap = forward_image.clamp(-1, 1) - forward_image

        return image + gap


class SecretEncoder3(nn.Module):
    def __init__(self, secret_len, base_res=16, resolution=64) -> None:
        super().__init__()
        log_resolution = int(np.log2(resolution))
        log_base = int(np.log2(base_res))
        self.secret_len = secret_len
        self.secret_scaler = nn.Sequential(
            nn.Linear(secret_len, base_res*base_res*3),
            nn.SiLU(),
            View(-1, 3, base_res, base_res),
            nn.Upsample(scale_factor=(2**(log_resolution-log_base), 2**(log_resolution-log_base))),  # chx16x16 -> chx256x256
            conv_nd(2, 3, 4, 3, padding=1)#zero_module()
        )  # secret len -> ch x res x res
    
    def copy_encoder_weight(self, ae_model):
        # misses, ignores = self.load_state_dict(ae_state_dict, strict=False)
        return None

    def encode(self, x):

        x = self.secret_scaler(x)

        return x
    
    def forward(self, x, c):
        # x: [B, C, H, W], c: [B, secret_len]
        c = self.encode(c)
        return c, None


class SecretEncoder4(nn.Module):
    """same as SecretEncoder3 but with ch as input"""
    def __init__(self, secret_len, ch=3, base_res=16, resolution=64) -> None:
        super().__init__()
        log_resolution = int(np.log2(resolution))
        log_base = int(np.log2(base_res))
        self.secret_len = secret_len
        self.secret_scaler = nn.Sequential(
            nn.Linear(secret_len, base_res*base_res*ch),
            nn.SiLU(),
            View(-1, ch, base_res, base_res),
            nn.Upsample(scale_factor=(2**(log_resolution-log_base), 2**(log_resolution-log_base))),  # chx16x16 -> chx256x256
            zero_module(conv_nd(2, ch, ch, 3, padding=1))
        )  # secret len -> ch x res x res
    
    def copy_encoder_weight(self, ae_model):
        # misses, ignores = self.load_state_dict(ae_state_dict, strict=False)
        return None

    def encode(self, x):
        x = self.secret_scaler(x)
        return x
    
    def forward(self, x, c):
        # x: [B, C, H, W], c: [B, secret_len]
        c = self.encode(c)
        return c, None
    
class SecretEncoder6(nn.Module):
    """join img emb with secret emb"""
    def __init__(self, secret_len, ch=3, base_res=16, resolution=64, emode='c3') -> None:
        super().__init__()
        assert emode in ['c3', 'c2', 'm3']
        
        if emode == 'c3':  # c3: concat c and x each has ch channels
            secret_ch = ch 
            join_ch = 2*ch
        elif emode == 'c2':  # c2: concat c (2) and x ave (1)
            secret_ch = 2
            join_ch = ch
        elif emode == 'm3':  # m3: multiply c (ch) and x (ch)
            secret_ch = ch
            join_ch = ch       
        
        # m3: multiply c (ch) and x ave (1)
        log_resolution = int(np.log2(resolution))
        log_base = int(np.log2(base_res))
        self.secret_len = secret_len
        self.emode = emode
        self.resolution = resolution
        self.secret_scaler = nn.Sequential(
            nn.Linear(secret_len, base_res*base_res*secret_ch),
            nn.SiLU(),
            View(-1, secret_ch, base_res, base_res),
            nn.Upsample(scale_factor=(2**(log_resolution-log_base), 2**(log_resolution-log_base))),  # chx16x16 -> chx256x256
        )  # secret len -> ch x res x res
        self.join_encoder = nn.Sequential(
            conv_nd(2, join_ch, join_ch, 3, padding=1),
            nn.SiLU(),
            conv_nd(2, join_ch, ch, 3, padding=1),
            nn.SiLU(),
            conv_nd(2, ch, ch, 3, padding=1),
            nn.SiLU()
        )
        self.out_layer = zero_module(conv_nd(2, ch, ch, 3, padding=1))
    
    def copy_encoder_weight(self, ae_model):
        # misses, ignores = self.load_state_dict(ae_state_dict, strict=False)
        return None

    def encode(self, x):
        x = self.secret_scaler(x)
        return x
    
    def forward(self, x, c):
        # x: [B, C, H, W], c: [B, secret_len]
        c = self.encode(c)
        if self.emode == 'c3':
            x = torch.cat([x, c], dim=1)
        elif self.emode == 'c2':
            x = torch.cat([x.mean(dim=1, keepdim=True), c], dim=1)
        elif self.emode == 'm3':
            x = x * c
        dx = self.join_encoder(x)
        dx = self.out_layer(dx)
        return dx, None
        
class SecretEncoder5(nn.Module):
    """same as SecretEncoder3 but with ch as input"""
    def __init__(self, secret_len, ch=3, base_res=16, resolution=64, joint=False) -> None:
        super().__init__()
        log_resolution = int(np.log2(resolution))
        log_base = int(np.log2(base_res))
        self.secret_len = secret_len
        self.joint = joint
        self.resolution = resolution
        self.secret_scaler = nn.Sequential(
            nn.Linear(secret_len, base_res*base_res*ch),
            nn.SiLU(),
            View(-1, ch, base_res, base_res),
            nn.Upsample(scale_factor=(2**(log_resolution-log_base), 2**(log_resolution-log_base))),  # chx16x16 -> chx256x256
        )  # secret len -> ch x res x res
        if joint:
            self.join_encoder = nn.Sequential(
                conv_nd(2, 2*ch, 2*ch, 3, padding=1),
                nn.SiLU(),
                conv_nd(2, 2*ch, ch, 3, padding=1),
                nn.SiLU()
            )
        self.out_layer = zero_module(conv_nd(2, ch, ch, 3, padding=1))
    
    def copy_encoder_weight(self, ae_model):
        # misses, ignores = self.load_state_dict(ae_state_dict, strict=False)
        return None

    def encode(self, x):
        x = self.secret_scaler(x)
        return x
    
    def forward(self, x, c):
        # x: [B, C, H, W], c: [B, secret_len]
        c = self.encode(c)
        if self.joint:
            x = thf.interpolate(x, size=(self.resolution, self.resolution), mode="bilinear", align_corners=False, antialias=True)
            c = self.join_encoder(torch.cat([x, c], dim=1))
        c = self.out_layer(c)
        return c, None


class SecretEncoder2(nn.Module):
    def __init__(self, secret_len, embed_dim, ddconfig, ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 ema_decay=None,
                 learn_logvar=False) -> None:
        super().__init__()
        log_resolution = int(np.log2(ddconfig.resolution))
        self.secret_len = secret_len
        self.learn_logvar = learn_logvar
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.encoder.conv_out = zero_module(self.encoder.conv_out)
        self.embed_dim = embed_dim

        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))

        if monitor is not None:
            self.monitor = monitor

        self.secret_scaler = nn.Sequential(
            nn.Linear(secret_len, 32*32*ddconfig.out_ch),
            nn.SiLU(),
            View(-1, ddconfig.out_ch, 32, 32),
            nn.Upsample(scale_factor=(2**(log_resolution-5), 2**(log_resolution-5))),  # chx16x16 -> chx256x256
            # zero_module(conv_nd(2, ddconfig.out_ch, ddconfig.out_ch, 3, padding=1))
        )  # secret len -> ch x res x res
        # out_resolution = ddconfig.resolution//(len(ddconfig.ch_mult)-1)
        # self.out_layer = zero_module(conv_nd(2, ddconfig.out_ch, ddconfig.out_ch, 3, padding=1))

        self.use_ema = ema_decay is not None
        if self.use_ema:
            self.ema_decay = ema_decay
            assert 0. < ema_decay < 1.
            self.model_ema = LitEma(self, decay=ema_decay)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)


    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        misses, ignores = self.load_state_dict(sd, strict=False)
        print(f"[SecretEncoder] Restored from {path}, misses: {misses}, ignores: {ignores}")

    def copy_encoder_weight(self, ae_model):
        # misses, ignores = self.load_state_dict(ae_state_dict, strict=False)
        return None
        self.encoder.load_state_dict(ae_model.encoder.state_dict())
        self.quant_conv.load_state_dict(ae_model.quant_conv.state_dict())

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")
    
    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    def encode(self, x):
        h = self.encoder(x)
        posterior = h
        return posterior
    
    def forward(self, x, c):
        # x: [B, C, H, W], c: [B, secret_len]
        c = self.secret_scaler(c)
        x = torch.cat([x, c], dim=1)
        z = self.encode(x)
        # z = self.out_layer(z)
        return z, None

class SecretEncoder(nn.Module):
    def __init__(self, secret_len, embed_dim, ddconfig, ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 ema_decay=None,
                 learn_logvar=False) -> None:
        super().__init__()
        log_resolution = int(np.log2(ddconfig.resolution))
        self.secret_len = secret_len
        self.learn_logvar = learn_logvar
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.embed_dim = embed_dim

        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))

        if monitor is not None:
            self.monitor = monitor

        self.use_ema = ema_decay is not None
        if self.use_ema:
            self.ema_decay = ema_decay
            assert 0. < ema_decay < 1.
            self.model_ema = LitEma(self, decay=ema_decay)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        self.secret_scaler = nn.Sequential(
            nn.Linear(secret_len, 32*32*ddconfig.out_ch),
            nn.SiLU(),
            View(-1, ddconfig.out_ch, 32, 32),
            nn.Upsample(scale_factor=(2**(log_resolution-5), 2**(log_resolution-5))),  # chx16x16 -> chx256x256
            zero_module(conv_nd(2, ddconfig.out_ch, ddconfig.out_ch, 3, padding=1))
        )  # secret len -> ch x res x res
        # out_resolution = ddconfig.resolution//(len(ddconfig.ch_mult)-1)
        self.out_layer = zero_module(conv_nd(2, ddconfig.out_ch, ddconfig.out_ch, 3, padding=1))

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        misses, ignores = self.load_state_dict(sd, strict=False)
        print(f"[SecretEncoder] Restored from {path}, misses: {misses}, ignores: {ignores}")

    def copy_encoder_weight(self, ae_model):
        # misses, ignores = self.load_state_dict(ae_state_dict, strict=False)
        self.encoder.load_state_dict(ae_model.encoder.state_dict())
        self.quant_conv.load_state_dict(ae_model.quant_conv.state_dict())

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")
    
    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior
    
    def forward(self, x, c):
        # x: [B, C, H, W], c: [B, secret_len]
        c = self.secret_scaler(c)
        x = x + c
        posterior = self.encode(x)
        z = posterior.sample()
        z = self.out_layer(z)
        return z, posterior


class ControlAE(pl.LightningModule):
    def __init__(self,
                 first_stage_key,
                 first_stage_config,
                 control_key,
                 control_config,
                 decoder_config,
                 loss_config,
                 noise_config='__none__',
                 use_ema=False,
                 secret_warmup=False,
                 scale_factor=1.,
                 ckpt_path="__none__",
                 ):
        super().__init__()
        self.scale_factor = scale_factor
        self.control_key = control_key
        self.first_stage_key = first_stage_key
        self.ae = instantiate_from_config(first_stage_config)
        self.control = instantiate_from_config(control_config)
        self.decoder = instantiate_from_config(decoder_config)
        if noise_config != '__none__':
            print('Using noise')
            self.noise = instantiate_from_config(noise_config)
        # copy weights from first stage
        self.control.copy_encoder_weight(self.ae)
        # freeze first stage
        self.ae.eval()
        self.ae.train = disabled_train
        for p in self.ae.parameters():
            p.requires_grad = False

        self.loss_layer = instantiate_from_config(loss_config)

        # early training phase
        # self.fixed_input = True
        self.fixed_x = None
        self.fixed_img = None
        self.fixed_input_recon = None
        self.fixed_control = None
        self.register_buffer("fixed_input", torch.tensor(True))

        # secret warmup
        self.secret_warmup = secret_warmup
        self.secret_baselen = 2
        self.secret_len = control_config.params.secret_len
        if self.secret_warmup:
            assert self.secret_len == 2**(int(np.log2(self.secret_len)))

        self.use_ema = use_ema
        if self.use_ema:
            print('Using EMA')
            self.control_ema = LitEma(self.control)
            self.decoder_ema = LitEma(self.decoder)
            print(f"Keeping EMAs of {len(list(self.control_ema.buffers()) + list(self.decoder_ema.buffers()))}.")

        if ckpt_path != '__none__':
            self.init_from_ckpt(ckpt_path, ignore_keys=[])

    def get_warmup_secret(self, old_secret):
        # old_secret: [B, secret_len]
        # new_secret: [B, secret_len]
        if self.secret_warmup:
            bsz = old_secret.shape[0]
            nrepeats = self.secret_len // self.secret_baselen
            new_secret  = torch.zeros((bsz, self.secret_baselen), dtype=torch.float).random_(0, 2).repeat_interleave(nrepeats, dim=1)
            return new_secret.to(old_secret.device)
        else:
            return old_secret
        
    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.control_ema.store(self.control.parameters())
            self.decoder_ema.store(self.decoder.parameters())
            self.control_ema.copy_to(self.control)
            self.decoder_ema.copy_to(self.decoder)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.control_ema.restore(self.control.parameters())
                self.decoder_ema.restore(self.decoder.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.control_ema(self.control)
            self.decoder_ema(self.decoder)

    def compute_loss(self, pred, target):
        # return thf.mse_loss(pred, target, reduction="none").mean(dim=(1, 2, 3))
        lpips_loss = self.lpips_loss(pred, target).mean(dim=[1,2,3])
        pred_yuv = color.rgb_to_yuv((pred + 1) / 2)
        target_yuv = color.rgb_to_yuv((target + 1) / 2)
        yuv_loss = torch.mean((pred_yuv - target_yuv)**2, dim=[2,3])
        yuv_loss = 1.5*torch.mm(yuv_loss, self.yuv_scales).squeeze(1)
        return lpips_loss + yuv_loss

    def forward(self, x, image, c):
        if self.control.__class__.__name__ == 'SecretEncoder6':
            eps, posterior = self.control(x, c)
        else:
            eps, posterior = self.control(image, c)
        return x + eps, posterior

    @torch.no_grad()
    def get_input(self, batch, return_first_stage=False, bs=None):
        image = batch[self.first_stage_key]
        control = batch[self.control_key]
        control = self.get_warmup_secret(control)
        if bs is not None:
            image = image[:bs]
            control = control[:bs]
        else:
            bs = image.shape[0]
        # encode image 1st stage
        image = einops.rearrange(image, "b h w c -> b c h w").contiguous()
        x = self.encode_first_stage(image).detach()
        image_rec = self.decode_first_stage(x).detach()
        
        # check if using fixed input (early training phase)
        # if self.training and self.fixed_input:
        if self.fixed_input:
            if self.fixed_x is None:  # first iteration
                print('[TRAINING] Warmup - using fixed input image for now!')
                self.fixed_x = x.detach().clone()[:bs]
                self.fixed_img = image.detach().clone()[:bs]
                self.fixed_input_recon = image_rec.detach().clone()[:bs]
                self.fixed_control = control.detach().clone()[:bs]  # use for log_images with fixed_input option only
            x, image, image_rec = self.fixed_x, self.fixed_img, self.fixed_input_recon
        
        out = [x, control]
        if return_first_stage:
            out.extend([image, image_rec])
        return out

    def decode_first_stage(self, z):
        z = 1./self.scale_factor * z
        image_rec = self.ae.decode(z)
        return image_rec
    
    def encode_first_stage(self, image):
        encoder_posterior = self.ae.encode(image)
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z

    def shared_step(self, batch):
        x, c, img, _ = self.get_input(batch, return_first_stage=True)
        # import pdb; pdb.set_trace()
        x, posterior = self(x, img, c)
        image_rec = self.decode_first_stage(x)
        # resize
        if img.shape[-1] > 256:
            img =  thf.interpolate(img, size=(256, 256), mode='bilinear', align_corners=False).detach()
            image_rec =  thf.interpolate(image_rec, size=(256, 256), mode='bilinear', align_corners=False)
        if hasattr(self, 'noise') and self.noise.is_activated():
            image_rec_noised = self.noise(image_rec, self.global_step, p=0.9)
        else:
            image_rec_noised = image_rec
        pred = self.decoder(image_rec_noised)

        loss, loss_dict = self.loss_layer(img, image_rec, posterior, c, pred, self.global_step)
        bit_acc = loss_dict["bit_acc"]

        bit_acc_ = bit_acc.item()

        if (bit_acc_ > 0.98) and (not self.fixed_input) and self.noise.is_activated():
            self.loss_layer.activate_ramp(self.global_step)

        if (bit_acc_ > 0.95) and (not self.fixed_input):  # ramp up image loss at late training stage
            if hasattr(self, 'noise') and (not self.noise.is_activated()):
                self.noise.activate(self.global_step) 
        
        # if (bit_acc_ > 0.95) and (not self.fixed_input) and self.secret_warmup:
        #     if self.secret_baselen == self.secret_len:  # warm up done
        #         self.secret_warmup = False
        #     else:
        #         print(f'[TRAINING] secret length warmup: {self.secret_baselen} -> {self.secret_baselen*2}')
        #         self.secret_baselen *= 2

        if (bit_acc_ > 0.9) and self.fixed_input:  # execute only once
            print(f'[TRAINING] High bit acc ({bit_acc_}) achieved, switch to full image dataset training.')
            self.fixed_input = ~self.fixed_input
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)
        loss_dict = {f"train/{key}": val for key, val in loss_dict.items()}
        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)
        
        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)
        # if self.use_scheduler:
        #     lr = self.optimizers().param_groups[0]['lr']
        #     self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, loss_dict_no_ema = self.shared_step(batch)
        loss_dict_no_ema = {f"val/{key}": val for key, val in loss_dict_no_ema.items() if key != 'img_lw'}
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch)
            loss_dict_ema = {'val/' + key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
    
    @torch.no_grad()
    def log_images(self, batch, fixed_input=False, **kwargs):
        log = dict()
        if fixed_input and self.fixed_img is not None:
            x, c, img, img_recon = self.fixed_x, self.fixed_control, self.fixed_img, self.fixed_input_recon
        else:
            x, c, img, img_recon = self.get_input(batch, return_first_stage=True)
        x, _ = self(x, img, c)
        image_out = self.decode_first_stage(x)
        if hasattr(self, 'noise') and self.noise.is_activated():
            img_noise = self.noise(image_out, self.global_step, p=1.0)
            log['noised'] = img_noise
        log['input'] = img
        log['output'] = image_out
        log['recon'] = img_recon
        return log
    
    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control.parameters()) + list(self.decoder.parameters())
        optimizer = torch.optim.AdamW(params, lr=lr)
        return optimizer
    



