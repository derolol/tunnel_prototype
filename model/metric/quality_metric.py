from typing import Any
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

class PSNR(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input_tensor, target_tensor) -> Any:
        B, C, H, W = input_tensor.shape
        mse = F.mse_loss(input_tensor, target_tensor, reduction='none')
        mse = mse.view(B, -1).mean(dim=1)
        return 10 * torch.log10(1 / mse)

class PSNRB(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input_tensor, target_tensor) -> Any:
        
        total = 0

        B, C, H, W = input_tensor.shape

        for c in range(C):
            input_c = input_tensor[:, c : c + 1, :, :]
            target_c = target_tensor[:, c : c + 1, :, :]
            
            mse = F.mse_loss(input_c, target_c, reduction='none')
            bef = self.blocking_effect_factor(input_c)

            mse = mse.view(B, -1).mean(dim=1)
            total += 10 * torch.log10(1 / (mse + bef))

        return total / C

    def blocking_effect_factor(self, im):
        B, C, H, W = im.shape
        block_size = 8

        block_horizontal_positions = torch.arange(7, W - 1, 8)
        block_vertical_positions = torch.arange(7, H - 1, 8)

        horizontal_block_difference = ((im[:, :, :, block_horizontal_positions] - im[:, :, :, block_horizontal_positions + 1])**2).sum(3).sum(2).sum(1)
        vertical_block_difference = ((im[:, :, block_vertical_positions, :] - im[:, :, block_vertical_positions + 1, :])**2).sum(3).sum(2).sum(1)

        nonblock_horizontal_positions = np.setdiff1d(torch.arange(0, W - 1), block_horizontal_positions)
        nonblock_vertical_positions = np.setdiff1d(torch.arange(0, H - 1), block_vertical_positions)

        horizontal_nonblock_difference = ((im[:, :, :, nonblock_horizontal_positions] - im[:, :, :, nonblock_horizontal_positions + 1])**2).sum(3).sum(2).sum(1)
        vertical_nonblock_difference = ((im[:, :, nonblock_vertical_positions, :] - im[:, :, nonblock_vertical_positions + 1, :])**2).sum(3).sum(2).sum(1)

        n_boundary_horiz = H * (W // block_size - 1)
        n_boundary_vert = W * (H // block_size - 1)
        boundary_difference = (horizontal_block_difference + vertical_block_difference) / (n_boundary_horiz + n_boundary_vert)


        n_nonboundary_horiz = H * (W - 1) - n_boundary_horiz
        n_nonboundary_vert = W * (H - 1) - n_boundary_vert
        nonboundary_difference = (horizontal_nonblock_difference + vertical_nonblock_difference) / (n_nonboundary_horiz + n_nonboundary_vert)

        scaler = np.log2(block_size) / np.log2(min([H, W]))
        bef = scaler * (boundary_difference - nonboundary_difference)

        bef[boundary_difference <= nonboundary_difference] = 0
        
        return bef

class SSIM(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input_tensor, target_tensor) -> Any:
        
        B, C, H, W = input_tensor.shape
        total = 0
        
        for c in range(C):
            input_c = input_tensor[:, c : c + 1, :, :]
            target_c = target_tensor[:, c : c + 1, :, :]
            total += self.ssim_single(input_c, target_c)

        return total / C

    def ssim_single(self, input_tensor, target_tensor):
        
        B, C, H, W = input_tensor.shape
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        kernel = torch.ones(1, 1, 8, 8) / 64
        kernel = kernel.to(input_tensor.device)

        mu_i = F.conv2d(input_tensor, kernel)
        mu_t = F.conv2d(target_tensor, kernel)

        var_i = F.conv2d(input_tensor ** 2, kernel) - mu_i ** 2
        var_t = F.conv2d(target_tensor ** 2, kernel) - mu_t ** 2
        cov_it = F.conv2d(target_tensor * input_tensor, kernel) - mu_i * mu_t

        a = (2 * mu_i * mu_t + C1) * (2 * cov_it + C2)
        b = (mu_i ** 2 + mu_t ** 2 + C1) * (var_i + var_t + C2)
        ssim_blocks = a / b
        
        return ssim_blocks.view(B, -1).mean(dim=1)

