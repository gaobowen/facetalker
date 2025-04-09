import sys
sys.path.append("loss")

import torch
import torch.nn as nn
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
# from musetalk.loss.discriminator import MultiScaleDiscriminator,DiscriminatorFullModel
import vgg_face as vgg_face

class HSLoss(torch.nn.Module):
    def __init__(self, hue_weight: float = 1.0, sat_weight: float = 0.5, light_weight: float = 0.5):
        super().__init__()
        self.hue_weight = hue_weight
        self.sat_weight = sat_weight
        self.light_weight = light_weight
    
    def rgb_to_hsl(self, image: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        将 RGB 图像转换为 HSL 颜色空间
        Args:
            image: Tensor of shape [B, C, H, W], 输入范围需为 [-1, 1]
            eps: 防止除零的小常数
        Returns:
            hsl: Tensor of shape [B, C, H, W], H 范围 [0, 360), S/L 范围 [0, 1]
        """
        # 输入是 [-1, 1]，先归一化        
        image = (image + 1) / 2  # 映射 [-1, 1] → [0, 1]
        image = image.clamp(0, 1)

        r, g, b = image[:, 0, ...], image[:, 1, ...], image[:, 2, ...]
        max_val, max_idx = torch.max(image, dim=1)  # [B, H, W]
        min_val, _ = torch.min(image, dim=1)        # [B, H, W]
        delta = max_val - min_val

        # 计算明度 L
        l = (max_val + min_val) / 2  # [B, H, W]

        # 计算饱和度 S
        s = torch.zeros_like(max_val)
        mask = delta != 0
        s[mask] = delta[mask] / (1 - torch.abs(2 * l[mask] - 1) + eps)

        # 计算色相 H
        h = torch.zeros_like(max_val)
        mask_r = (max_idx == 0) & mask  # 最大值是 R 通道
        mask_g = (max_idx == 1) & mask  # 最大值是 G 通道
        mask_b = (max_idx == 2) & mask  # 最大值是 B 通道

        # R 通道主导的色相
        h[mask_r] = ((g - b)[mask_r] / (delta[mask_r] + eps)) % 6
        # G 通道主导的色相
        h[mask_g] = ((b - r)[mask_g] / (delta[mask_g] + eps) + 2)
        # B 通道主导的色相
        h[mask_b] = ((r - g)[mask_b] / (delta[mask_b] + eps) + 4)
        h = (h * 60) % 360  # 转换为角度 [0, 360)

        # 组合 HSL
        hsl = torch.stack([h, s, l], dim=1)
        return hsl

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算预测图和目标图的 HSL 空间损失
        Args:
            pred:   Tensor of shape [B, C, H, W], 输入范围需为 [-1, 1]
            target: Tensor of shape [B, C, H, W], 输入范围需与 pred 一致
        Returns:
            loss: 加权后的 HSL 损失
        """
        pred_hsl = self.rgb_to_hsl(pred)
        target_hsl = self.rgb_to_hsl(target)

        # 拆分 H, S, L 分量
        h_pred, s_pred, l_pred = pred_hsl[:, 0, ...], pred_hsl[:, 1, ...], pred_hsl[:, 2, ...]
        h_target, s_target, l_target = target_hsl[:, 0, ...], target_hsl[:, 1, ...], target_hsl[:, 2, ...]

        # 色相损失（处理 360° 环状特性）
        h_diff = torch.abs(h_pred - h_target)
        h_loss = torch.min(h_diff, 360 - h_diff).mean()

        # 饱和度和明度损失（L1）
        s_loss = torch.nn.functional.l1_loss(s_pred, s_target)
        l_loss = torch.nn.functional.l1_loss(l_pred, l_target)

        # 加权总损失
        total_loss = (
            self.hue_weight * h_loss +
            self.sat_weight * s_loss +
            self.light_weight * l_loss)
        
        return total_loss


class Interpolate(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
        super(Interpolate, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, input):
        return F.interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners)

def set_requires_grad(net, requires_grad=False):
    if net is not None:
        for param in net.parameters():
            param.requires_grad = requires_grad

def initialize_vgg(cfg, device):
    """Initialize VGG model"""
    if cfg.loss_params.vgg_loss > 0:
        vgg_IN = vgg_face.Vgg19().to(device)
        pyramid = vgg_face.ImagePyramide(
            cfg.loss_params.pyramid_scale, 3).to(device)
        vgg_IN.eval()
        downsampler = Interpolate(
            size=(224, 224), mode='bilinear', align_corners=False).to(device)
        return vgg_IN, pyramid, downsampler
    return None, None, None


if __name__ == "__main__":
    cfg = OmegaConf.load("config/stage1.yaml")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pyramid_scale = [1, 0.5, 0.25, 0.125]
    vgg_IN = vgg_face.Vgg19().to(device)
    pyramid = vgg_face.ImagePyramide(cfg.loss_params.pyramid_scale, 3).to(device)
    vgg_IN.eval()
    downsampler = Interpolate(size=(224, 224), mode='bilinear', align_corners=False)
    img_hw = 320
    image = torch.rand(8, 3, img_hw, img_hw).to(device)
    image_pred = torch.rand(8, 3, img_hw, img_hw).to(device)
    pyramide_real = pyramid(downsampler(image))
    pyramide_generated = pyramid(downsampler(image_pred))
    

    loss_IN = 0
    for scale in cfg.loss_params.pyramid_scale:
        x_vgg = vgg_IN(pyramide_generated['prediction_' + str(scale)])
        y_vgg = vgg_IN(pyramide_real['prediction_' + str(scale)])
        for i, weight in enumerate(cfg.loss_params.vgg_layer_weight):
            value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean() 
            loss_IN += weight * value
    loss_IN /= sum(cfg.loss_params.vgg_layer_weight)  # 对vgg不同层取均值，金字塔loss是每层叠
    # 多尺度的图像loss能够关注到图片的不同细节，比如低分辨率的图像更关注轮廓，高分辨率图像更关注细节。
    print('loss_IN ==>', loss_IN)

    #print(cfg.model_params.discriminator_params)

    # discriminator = MultiScaleDiscriminator(**cfg.model_params.discriminator_params).to(device)
    # discriminator_full = DiscriminatorFullModel(discriminator)
    # disc_scales = cfg.model_params.discriminator_params.scales
    # # Prepare optimizer and loss function
    # optimizer_D = optim.AdamW(discriminator.parameters(), 
    #                             lr=cfg.discriminator_train_params.lr, 
    #                             weight_decay=cfg.discriminator_train_params.weight_decay,
    #                             betas=cfg.discriminator_train_params.betas,
    #                             eps=cfg.discriminator_train_params.eps)
    # scheduler_D = CosineAnnealingLR(optimizer_D, 
    #                                 T_max=cfg.discriminator_train_params.epochs, 
    #                                 eta_min=1e-6)

    # discriminator.train()

    # set_requires_grad(discriminator, False)

    # loss_G = 0.
    # discriminator_maps_generated = discriminator(pyramide_generated)
    # discriminator_maps_real = discriminator(pyramide_real)

    # for scale in disc_scales:
    #     key = 'prediction_map_%s' % scale
    #     value = ((1 - discriminator_maps_generated[key]) ** 2).mean()
    #     loss_G += value

    # print(loss_G)
