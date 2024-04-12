# ---------------------------------------------------------------
# Ref:
# [1]Code: https://github.com/bubbliiiing/segformer-pytorch
# ---------------------------------------------------------------
import os
import math
import warnings
import numpy as np
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    r"""
    Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

#--------------------------------------#
#   Gelu激活函数的实现
#   利用近似的数学公式
#--------------------------------------#
class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x,3))))

class OverlapPatchEmbed(nn.Module):
    '''
    通过Conv2D实现图像patch编码
    '''
    
    def __init__(self, in_channels=3, patch_size=7, stride=4, embed_dim=768):
    
        super().__init__()
    
        patch_size  = (patch_size, patch_size)

        self.proj = nn.Conv2d(in_channels=in_channels, out_channels=embed_dim,
                              kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        
        self.norm = nn.LayerNorm(normalized_shape=embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):

        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W

class Attention(nn.Module):
    '''    
    Attention机制
    将输入的特征qkv特征进行划分,首先生成query, key, value。query是查询向量、key是键向量、v是值向量。
    然后利用 查询向量query 叉乘 转置后的键向量key,这一步可以通俗的理解为,利用查询向量去查询序列的特征,获得序列每个部分的重要程度score。
    然后利用 score 叉乘 value,这一步可以通俗的理解为,将序列每个部分的重要程度重新施加到序列的值上去。
    
    在segformer中,为了减少计算量,首先对特征图进行了浓缩,所有特征层都压缩到原图的1/32。
    当输入图片为512, 512时,Block1的特征图为128, 128,此时就先将特征层压缩为16, 16。
    在Block1的Attention模块中,相当于将8x8个特征点进行特征浓缩,浓缩为一个特征点。
    然后利用128x128个查询向量对16x16个键向量与值向量进行查询。尽管键向量与值向量的数量较少,但因为查询向量的不同,依然可以获得不同的输出。
    '''
    
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim        = dim
        self.num_heads  = num_heads
        head_dim        = dim // num_heads
        self.scale      = qk_scale or head_dim ** -0.5

        self.q          = nn.Linear(dim, dim, bias=qkv_bias)
        
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr     = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm   = nn.LayerNorm(dim)
        self.kv         = nn.Linear(dim, dim * 2, bias=qkv_bias)
        
        self.attn_drop  = nn.Dropout(attn_drop)
        
        self.proj       = nn.Linear(dim, dim)
        self.proj_drop  = nn.Dropout(proj_drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        # bs, 16384, 32 => bs, 16384, 32 => bs, 16384, 8, 4 => bs, 8, 16384, 4
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            # bs, 16384, 32 => bs, 32, 128, 128
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            # bs, 32, 128, 128 => bs, 32, 16, 16 => bs, 256, 32
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            # bs, 256, 32 => bs, 256, 64 => bs, 256, 2, 8, 4 => 2, bs, 8, 256, 4
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # bs, 8, 16384, 4 @ bs, 8, 4, 256 => bs, 8, 16384, 256 
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # bs, 8, 16384, 256  @ bs, 8, 256, 4 => bs, 8, 16384, 4 => bs, 16384, 32
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # bs, 16384, 32 => bs, 16384, 32
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
    
    def forward_cross(self, x, query, H, W):
        B, N, C = x.shape
        # bs, 16384, 32 => bs, 16384, 32 => bs, 16384, 8, 4 => bs, 8, 16384, 4
        q = self.q(query).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            # bs, 16384, 32 => bs, 32, 128, 128
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            # bs, 32, 128, 128 => bs, 32, 16, 16 => bs, 256, 32
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            # bs, 256, 32 => bs, 256, 64 => bs, 256, 2, 8, 4 => 2, bs, 8, 256, 4
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # bs, 8, 16384, 4 @ bs, 8, 4, 256 => bs, 8, 16384, 256 
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # bs, 8, 16384, 256  @ bs, 8, 256, 4 => bs, 8, 16384, 4 => bs, 16384, 32
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # bs, 16384, 32 => bs, 16384, 32
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob       = 1 - drop_prob
    shape           = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor   = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)
    
class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x
    
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=GELU, drop=0.):
        super().__init__()
        out_features    = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1    = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act    = act_layer()
        
        self.fc2    = nn.Linear(hidden_features, out_features)
        
        self.drop   = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1      = norm_layer(dim)
        self.query_norm1 = norm_layer(dim)
        
        self.attn       = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio
        )
        self.norm2      = norm_layer(dim)
        self.mlp        = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

        self.drop_path  = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x
    
    def forward_cross(self, x, query, H, W):
        x = x + self.drop_path(self.attn.forward_cross(self.norm1(x), self.query_norm1(query), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
class VAE(nn.Module):
    '''
    VAE for 64x64 face generation. The hidden dimensions can be tuned.
    '''
    def __init__(self, in_channels=32, input_size=128, latent_dim=128) -> None:

        super().__init__()

        self.latent_dim = latent_dim

        # encoder
        modules = []
        hidden_num = 3
        prev_channels = in_channels
        for i in range(hidden_num):
            cur_channels = prev_channels * 2
            modules.append(
                nn.Sequential(
                    nn.Conv2d(prev_channels,
                              cur_channels,
                              kernel_size=3,
                              stride=2,
                              padding=1),
                    nn.BatchNorm2d(cur_channels),
                    nn.ReLU()))
            prev_channels = cur_channels
            input_size //= 2        
        
        self.encoder = nn.Sequential(*modules)

        self.mean_linear = nn.Linear(in_features=(prev_channels * input_size * input_size),
                                     out_features=latent_dim)
        self.var_linear = nn.Linear(in_features=(prev_channels * input_size * input_size),
                                    out_features=latent_dim)
        
        # decoder
        self.decoder_input_chw = (prev_channels, input_size, input_size)
        self.decoder_projection = nn.Linear(in_features=latent_dim,
                                            out_features=(prev_channels * input_size * input_size))
        modules = []
        for i in range(hidden_num):
            cur_channels = prev_channels // 2
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(prev_channels,
                                       cur_channels,
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(cur_channels),
                    nn.ReLU()))
            prev_channels = cur_channels
        modules.append(
            nn.Sequential(
                nn.Conv2d(cur_channels,
                          cur_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1),
                nn.BatchNorm2d(cur_channels),
                nn.ReLU()))
        
        self.decoder = nn.Sequential(*modules)

    def forward(self, x):
        
        encoded = self.encoder(x)
        encoded = torch.flatten(encoded, 1)
        
        mean = self.mean_linear(encoded)
        logvar = self.var_linear(encoded)
        
        eps = torch.randn_like(logvar)
        std = torch.exp(logvar / 2)
        
        z = eps * std + mean
        
        x = self.decoder_projection(z)
        x = torch.reshape(x, (-1, *self.decoder_input_chw))
        
        decoded = self.decoder(x)

        return decoded, mean, logvar

class MixVisionTransformer(nn.Module):
    
    def __init__(self,
                 in_channels=3, num_classes=1000,
                 embed_dims=[32, 64, 160, 256],
                 num_heads=[1, 2, 4, 8],
                 mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        
        super().__init__()
        
        self.num_classes    = num_classes
        self.depths         = depths

        #----------------------------------#
        #   Transformer模块,共有四个部分
        #----------------------------------#
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        #----------------------------------#
        #   block1
        #----------------------------------#
        #-----------------------------------------------#
        #   对输入图像进行分区,并下采样
        #   512, 512, 3 => 128, 128, 32 => 16384, 32
        #-----------------------------------------------#
        self.patch_embed1 = OverlapPatchEmbed(in_channels=in_channels, 
                                              patch_size=7, stride=4, 
                                              embed_dim=embed_dims[0])
        #-----------------------------------------------#
        #   利用transformer模块进行特征提取
        #   16384, 32 => 16384, 32
        #-----------------------------------------------#
        cur = 0
        self.block1 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[0]
                )
                for i in range(depths[0])
            ]
        )
        
        self.norm1 = norm_layer(embed_dims[0])

        #----------------------------------#
        #   block2
        #----------------------------------#
        #-----------------------------------------------#
        #   对输入图像进行分区,并下采样
        #   128, 128, 32 => 64, 64, 64 => 4096, 64
        #-----------------------------------------------#
        self.patch_embed2 = OverlapPatchEmbed(in_channels=embed_dims[0],
                                              patch_size=3, stride=2,
                                              embed_dim=embed_dims[1])
        #-----------------------------------------------#
        #   利用transformer模块进行特征提取
        #   4096, 64 => 4096, 64
        #-----------------------------------------------#
        cur += depths[0]
        self.block2 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[1],
                    num_heads=num_heads[1], mlp_ratio=mlp_ratios[1],
                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                    norm_layer=norm_layer, sr_ratio=sr_ratios[1]
                )
                for i in range(depths[1])
            ]
        )
        self.norm2 = norm_layer(embed_dims[1])

        #----------------------------------#
        #   block3
        #----------------------------------#
        #-----------------------------------------------#
        #   对输入图像进行分区,并下采样
        #   64, 64, 64 => 32, 32, 160 => 1024, 160
        #-----------------------------------------------#
        self.patch_embed3 = OverlapPatchEmbed(in_channels=embed_dims[1],
                                              patch_size=3, stride=2,
                                              embed_dim=embed_dims[2])
        #-----------------------------------------------#
        #   利用transformer模块进行特征提取
        #   1024, 160 => 1024, 160
        #-----------------------------------------------#
        cur += depths[1]
        self.block3 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[2],
                    num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], 
                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                    norm_layer=norm_layer, sr_ratio=sr_ratios[2]
                )
                for i in range(depths[2])
            ]
        )
        self.norm3 = norm_layer(embed_dims[2])

        #----------------------------------#
        #   block4
        #----------------------------------#
        #-----------------------------------------------#
        #   对输入图像进行分区,并下采样
        #   32, 32, 160 => 16, 16, 256 => 256, 256
        #-----------------------------------------------#
        self.patch_embed4 = OverlapPatchEmbed(in_channels=embed_dims[2],
                                              patch_size=3, stride=2,
                                              embed_dim=embed_dims[3])
        #-----------------------------------------------#
        #   利用transformer模块进行特征提取
        #   256, 256 => 256, 256
        #-----------------------------------------------#
        cur += depths[2]
        self.block4 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[3],
                    num_heads=num_heads[3], mlp_ratio=mlp_ratios[3],
                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                    norm_layer=norm_layer, sr_ratio=sr_ratios[3]
                )
                for i in range(depths[3])
            ]
        )
        self.norm4 = norm_layer(embed_dims[3])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
                
    def forward(self, x):
        B = x.shape[0]
        outs = []

        #----------------------------------#
        #   block1
        #----------------------------------#
        x, H, W = self.patch_embed1.forward(x)
        
        for i, blk in enumerate(self.block1):
            x = blk.forward(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        
        outs.append(x)

        #----------------------------------#
        #   block2
        #----------------------------------#
        x, H, W = self.patch_embed2.forward(x)
        for i, blk in enumerate(self.block2):
            x = blk.forward(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        #----------------------------------#
        #   block3
        #----------------------------------#
        x, H, W = self.patch_embed3.forward(x)
        for i, blk in enumerate(self.block3):
            x = blk.forward(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        #----------------------------------#
        #   block4
        #----------------------------------#
        x, H, W = self.patch_embed4.forward(x)
        for i, blk in enumerate(self.block4):
            x = blk.forward(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs


class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x
    
class ConvModule(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        super(ConvModule, self).__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act    = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class SegFormerHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, num_classes=20, in_channels=[32, 64, 160, 256], head_embed_dim=768, dropout_ratio=0.1):
        super(SegFormerHead, self).__init__()
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=head_embed_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=head_embed_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=head_embed_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=head_embed_dim)

        self.linear_fuse = ConvModule(
            c1=(head_embed_dim * 4),
            c2=head_embed_dim,
            k=1,
        )

        self.reconstruct1 = VAE(in_channels=head_embed_dim,
                                input_size=128,
                                latent_dim=128)
        # self.concat1 = ConvModule(c1=embed_dims[0] * 2,
        #                           c2=embed_dims[0],
        #                           k=3,
        #                           s=1,
        #                           p=1)

        self.block_vae_trans = nn.ModuleList(
            [
                Block(
                    dim=head_embed_dim, num_heads=1, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                    drop=0.0, attn_drop=0.1, drop_path=0.1, norm_layer=nn.LayerNorm, sr_ratio=8
                )
                for i in range(2)
            ]
        )
        self.norm1_vae_trans = nn.LayerNorm(head_embed_dim)

        self.concat1 = SKFF(in_channels=head_embed_dim, height=2)

        self.linear_pred    = nn.Conv2d(head_embed_dim, num_classes, kernel_size=1)
        self.dropout        = nn.Dropout2d(dropout_ratio)
    
    def forward(self, inputs):
        c1, c2, c3, c4 = inputs

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape
        
        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        reconstructs = []
        reconstruct1, mean, logvar = self.reconstruct1(_c.detach())
        reconstructs = [_c.detach(), reconstruct1 + _c.detach(), mean, logvar]

        recon = reconstruct1.detach() + _c
        B, C, H, W = recon.shape
        recon = recon.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.block_vae_trans):
            recon = blk.forward(recon, H, W)
        recon = self.norm1_vae_trans(recon)
        recon = recon.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        _c = _c + self.concat1([recon, _c])
        # x = x + self.concat1(torch.concat([reconstruct1.detach(), x], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x, reconstructs

class SegFormer(nn.Module):
    def __init__(self,
                 num_classes,
                 pretrained_path,
                 in_channels,
                 embed_dims,
                 num_heads,
                 mlp_ratios,
                 qkv_bias,
                 depths,
                 sr_ratios,
                 drop_rate,
                 drop_path_rate,
                 head_embed_dim):
        
        super().__init__()
        
        self.backbone = MixVisionTransformer(in_channels=in_channels,
                                             num_classes=num_classes,
                                             embed_dims=embed_dims,
                                             num_heads=num_heads,
                                             mlp_ratios=mlp_ratios,
                                             qkv_bias=qkv_bias,
                                             depths=depths,
                                             sr_ratios=sr_ratios,
                                             drop_rate=drop_rate,
                                             drop_path_rate=drop_path_rate,
                                             norm_layer=partial(nn.LayerNorm, eps=1e-6))
        if pretrained_path:
            print("Load backbone weights")
            state_dict = torch.load(pretrained_path)
            self.backbone.load_state_dict(state_dict=state_dict, strict=False)

        self.decode_head = SegFormerHead(num_classes=num_classes,
                                         in_channels=embed_dims,
                                         head_embed_dim=head_embed_dim)

    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)
        
        x = self.backbone.forward(inputs)
        out, recons = self.decode_head.forward(x)
        
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=True)

        return x, recons, out

class SKFF(nn.Module):

    def __init__(self, in_channels, height=3, reduction=8, bias=False):
        super(SKFF, self).__init__()
        
        self.height = height
        d = max(int(in_channels / reduction), 4)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.LeakyReLU(0.2))

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1,bias=bias))
        
        self.softmax = nn.Softmax(dim=1)

        self.weight_vale = torch.zeros(size=(1, height, 1, 1, 1))
        self.weight_vale[0, :] = 1
        self.weight = torch.nn.Parameter(self.weight_vale, requires_grad=True)


    def forward(self, inp_feats):
        batch_size = inp_feats[0].shape[0]
        n_feats =  inp_feats[0].shape[1]
        

        inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])
        
        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)
        # self.weight = self.weight.view(1, self.height, n_feats, 1, 1)

        attention_vectors = self.weight * attention_vectors

        # stx()
        attention_vectors = self.softmax(attention_vectors)
        
        feats_V = torch.sum(inp_feats * attention_vectors, dim=1)
        
        return feats_V   

if __name__ == '__main__':

    model = SegFormer(num_classes=3,
                    pretrained_path=None,
                    in_channels=3,
                    embed_dims=[32, 64, 160, 256],
                    num_heads=[1, 2, 5, 8],
                    mlp_ratios=[4, 4, 4, 4],
                    qkv_bias=True,
                    depths=[2, 2, 2, 2],
                    sr_ratios=[8, 4, 2, 1],
                    drop_rate=0.0,
                    drop_path_rate=0.1,
                    head_embed_dim=256)
    input_tensor = torch.rand(size=(1, 3, 512, 512))
    _, recons, output_tensor = model(input_tensor)
    print(recons[0].shape, output_tensor.shape)