# ---------------------------------------------------------------
# Ref:
# [1]Code: https://github.com/bubbliiiing/segformer-pytorch
# ---------------------------------------------------------------

import os
import math
from einops import rearrange
from functools import partial

import torch
import torch.nn.functional as F
from torch.nn import Module, Conv2d, Linear, Identity, Sequential, ModuleList
from torch.nn import BatchNorm2d, LayerNorm, GroupNorm
from torch.nn import LeakyReLU, GELU, Dropout
from torch.nn import Embedding
from torch.nn.init import trunc_normal_, constant_

def _init_weights(m):

    if isinstance(m, Linear):
        trunc_normal_(m.weight, std=.02)
        if isinstance(m, Linear) and m.bias is not None:
            constant_(m.bias, 0)

    elif isinstance(m, LayerNorm):
        constant_(m.bias, 0)
        constant_(m.weight, 1.0)
    
    elif isinstance(m, Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        fan_out //= m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()

def drop_path(x,
              drop_prob: float = 0.,
              training: bool = False,
              scale_by_keep: bool = True):

    # Keep path
    if drop_prob == 0. or not training:
        return x
    
    keep_prob = 1 - drop_prob
    # (B, 1, 1, ...)
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    # Dim
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    # Scale
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)

    return x * random_tensor

class DropPath(Module):

    def __init__(self,
                 drop_prob=None,
                 scale_by_keep=True):
        
        super().__init__()
        
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x,
                         self.drop_prob,
                         self.training,
                         self.scale_by_keep)

class ConvModule(Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 groups=1):
        
        super().__init__()
        
        self.conv = Conv2d(in_channels=in_channels,
                           out_channels=out_channels,
                           kernel_size=kernel_size,
                           stride=stride,
                           padding=padding,
                           groups=groups,
                           bias=False)
        self.norm = GroupNorm(num_groups=1,
                              num_channels=out_channels)
        self.act = GELU()

    def forward(self, x):

        return self.act(self.norm(self.conv(x)))

class OverlapPatchEmbed(Module):
    '''
    Overlap Patch Embedding
    '''
    
    def __init__(self,
                 in_channels,
                 patch_size,
                 stride,
                 embed_dim):
    
        super().__init__()
    
        padding = patch_size // 2

        self.proj = Conv2d(in_channels=in_channels,
                           out_channels=embed_dim,
                           kernel_size=patch_size,
                           stride=stride,
                           padding=padding)
        
        self.norm = LayerNorm(normalized_shape=embed_dim)

        self.apply(_init_weights)

    def forward(self, x):

        # Embed Patch
        x = self.proj(x)

        return x

        # Reshape
        # _, _, H, W = x.shape
        # x = rearrange(x, pattern='b c h w -> b (h w) c')
        # x = self.norm(x)

        # return x, H, W

class Attention(Module):
    '''
    '''
    
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 sr_ratio=1):
        
        super().__init__()

        assert dim % num_heads == 0

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = Linear(in_features=dim,
                        out_features=dim,
                        bias=qkv_bias)
        
        # Smaller Key & Value
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr     = Conv2d(in_channels=dim, out_channels=dim,
                                 kernel_size=sr_ratio, stride=sr_ratio)
            self.norm   = LayerNorm(normalized_shape=dim)
        self.kv = Linear(dim, dim * 2, bias=qkv_bias)
        
        self.attn_drop  = Dropout(attn_drop)
        
        self.proj       = Linear(dim, dim)
        self.proj_drop  = Dropout(proj_drop)

        self.apply(_init_weights)

    def forward(self, x, H, W):

        B, N, C = x.shape
        
        q = self.q(x)
        q = rearrange(q,
                      'b n (head c_head) -> b head n c_head',
                      head=self.num_heads,
                      c_head=(C // self.num_heads))

        if self.sr_ratio > 1:
            x_ = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
            x_ = self.sr(x_)
            x_ = rearrange(x_, 'b c h w -> b (h w) c')
            x_ = self.norm(x_)
            kv = self.kv(x_)
            kv = rearrange(kv,
                           'b n (kv head c_head) -> kv b head n c_head',
                           kv=2,
                           head=self.num_heads,
                           c_head=(C // self.num_heads))
        else:
            kv = self.kv(x)
            kv = rearrange(kv,
                           'b n (kv head c_head) -> kv b head n c_head',
                           kv=2,
                           head=self.num_heads,
                           c_head=(C // self.num_heads))

        k = rearrange(kv[0], 'b head n c_head -> b head c_head n')
        v = kv[1]

        # (b, head, n, c_head) @ (b, head, c_head, n)
        attn = (q @ k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # (b, head, n, n) @ (b, head, n, c_head)
        x = attn @ v
        x = rearrange(x, 'b head n c_head -> b n (head c_head)')
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class DepthwiseConv(Module):

    def __init__(self, dim):

        super().__init__()
        
        self.dwconv = Conv2d(in_channels=dim,
                             out_channels=dim,
                             kernel_size=3,
                             stride=1,
                             padding=1,
                             groups=dim,
                             bias=True)

    def forward(self, x, H, W):

        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        x = self.dwconv(x)
        x = rearrange(x, 'b c h w -> b (h w) c')

        return x
    
class MLP(Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=GELU,
                 drop=0.):
        
        super().__init__()
        
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = Linear(in_features=in_features,
                          out_features=hidden_features)
        self.dwconv = DepthwiseConv(dim=hidden_features)
        self.act = act_layer()
        
        self.fc2 = Linear(in_features=hidden_features,
                          out_features=out_features)
        self.drop = Dropout(drop)

        self.apply(_init_weights)

    def forward(self, x, H, W):
        
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        
        x = self.fc2(x)
        x = self.drop(x)
        
        return x

class AttentionBlock(Module):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=GELU,
                 norm_layer=LayerNorm,
                 sr_ratio=1):
        
        super().__init__()
        
        self.norm1 = norm_layer(dim)
        
        self.attn = Attention(dim,
                              num_heads=num_heads,
                              qkv_bias=qkv_bias,
                              qk_scale=qk_scale,
                              attn_drop=attn_drop,
                              proj_drop=drop,
                              sr_ratio=sr_ratio)
        
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(in_features=dim,
                       hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer,
                       drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()

        self.norm3 = norm_layer(dim)

        self.apply(_init_weights)

    # def forward(self, x, H, W):
    def forward(self, x):

        # Reshape
        _, _, H, W = x.shape
        x = rearrange(x, pattern='b c h w -> b (h w) c')
        
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        x = self.norm3(x)
        
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        
        return x

class SegEncoder(Module):

    def __init__(self,
                 in_channels=3,
                 embed_dims=[32, 64, 160, 256], # AttentionBlock dim
                 depths=[3, 4, 6, 3],           # AttentionBlock num
                 drop_path_rate=0.,             # AttentionBlock drop path rate
                 num_heads=[1, 2, 4, 8],        # Attention head number
                 qkv_bias=False,                # Attention Linear bias
                 sr_ratios=[8, 4, 2, 1],        # Attention K, V size ratio
                 qk_scale=None,                 # Attention value scale
                 attn_drop_rate=0.,             # Attention Q, K dropout
                 drop_rate=0.,                  # Attention + MLP last dropout
                 mlp_ratios=[4, 4, 4, 4],       # MLP hidden dim ratio
                 norm_layer=LayerNorm,          # Attention + MLP norm type
                 latent_dim=512,                # Latent dim
                 ):
        
        super().__init__()

        self.blocks = ModuleList()
        
        drop_path_rate_list = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        input_channel = in_channels

        for i in range(len(embed_dims)):

            output_channel = embed_dims[i]

            path_embed = OverlapPatchEmbed(in_channels=input_channel,
                                           embed_dim=output_channel,
                                           patch_size=3, stride=2)
            input_channel = output_channel
            
            layers = [
                AttentionBlock(dim=output_channel,
                               drop_path=drop_path_rate_list[i + d], 
                               num_heads=num_heads[i],
                               qkv_bias=qkv_bias,
                               sr_ratio=sr_ratios[i],
                               qk_scale=qk_scale,
                               attn_drop=attn_drop_rate,
                               mlp_ratio=mlp_ratios[i],
                               drop=drop_rate,
                               norm_layer=norm_layer)
                for d in range(depths[i])]
            
            self.blocks.append(Sequential(path_embed, *layers))
        
        self.encode_latent = Conv2d(in_channels=embed_dims[-1],
                                    out_channels=latent_dim,
                                    stride=3,
                                    kernel_size=1,
                                    padding=1)
        
        self.apply(_init_weights)

    def forward(self, x):

        feature_maps = []

        for block in self.blocks:
            x = block(x)
            feature_maps.append(x)

        latent_feature = self.encode_latent(x)
        feature_maps.append(latent_feature)

        return feature_maps

class Codebook(Module):

    def __init__(self, num_vectors, latent_dim, beta):

        super().__init__()

        self.num_vectors = num_vectors
        self.latent_dim = latent_dim
        self.beta = beta

        self.embedding = Embedding(self.num_vectors, self.latent_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_vectors, 1.0 / self.num_vectors)

    def forward(self, x):

        x = rearrange(x, 'b c h w -> b h w c')
        x_flattened = rearrange(x, 'b h w c -> (b h w) c')

        # (b x h x w, num_vectors)
        d = torch.sum(x_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - \
            2 * (torch.matmul(x_flattened, self.embedding.weight.t()))

        # (b x h x w, 1)
        min_encoding_indices = torch.argmin(d, dim=1)
        # (b , h, w, latent_dim)
        x_q = self.embedding(min_encoding_indices).view(x.shape)

        loss = torch.mean((x_q.detach() - x) ** 2) + self.beta * torch.mean((x_q - x.detach()) ** 2)

        x_q = x + (x_q - x).detach()
        x_q = rearrange(x_q, 'b h w c -> b c h w')

        return x_q, min_encoding_indices, loss

class FuseBlock(Module):

    def __init__(self, in_channels, out_channels, block_num=2) -> None:

        super().__init__()
        
        self.up_block = ConvModule(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
        
        self.seg_block = ConvModule(in_channels=out_channels,
                                    out_channels=out_channels,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)
        
        self.fuse = ConvModule(in_channels=(out_channels * 2),
                               out_channels=out_channels)
    
    def forward(self, x, concat_x):

        x = self.up_block(x)
        x = F.interpolate(x, size=(concat_x.shape[2:]), mode='bilinear')

        concat_x = concat_x + self.seg_block(concat_x)

        x = x + self.fuse(torch.concat([x, concat_x], dim=1))

        return x

class SegDecoder(Module):

    def __init__(self,
                 embed_dims=[32, 64, 160, 256], # AttentionBlock dim
                 depths=[3, 4, 6, 3],           # AttentionBlock num
                 drop_path_rate=0.,             # AttentionBlock drop path rate
                 num_heads=[1, 2, 4, 8],        # Attention head number
                 qkv_bias=False,                # Attention Linear bias
                 sr_ratios=[8, 4, 2, 1],        # Attention K, V size ratio
                 qk_scale=None,                 # Attention value scale
                 attn_drop_rate=0.,             # Attention Q, K dropout
                 drop_rate=0.,                  # Attention + MLP last dropout
                 mlp_ratios=[4, 4, 4, 4],       # MLP hidden dim ratio
                 norm_layer=LayerNorm,          # Attention + MLP norm type
                 num_classes=2,                 # Seg class num
                 latent_dim=512,                # Latent dim num
                 ):
        
        super().__init__()

        embed_dims.reverse()
        sr_ratios.reverse()
        num_heads.reverse()
        depths.reverse()
        mlp_ratios.reverse()

        self.fuse_blocks = ModuleList()
        self.blocks = ModuleList()
        
        drop_path_rate_list = [
            x.item() for x in torch.linspace(drop_path_rate, 0, sum(depths))]

        input_channel = latent_dim

        for i in range(len(embed_dims)):

            output_channel = embed_dims[i]

            self.fuse_blocks.append(
                FuseBlock(in_channels=input_channel, out_channels=output_channel))
            
            input_channel = output_channel

            layers = [
                AttentionBlock(dim=output_channel,
                               drop_path=drop_path_rate_list[i + d], 
                               num_heads=num_heads[i],
                               qkv_bias=qkv_bias,
                               sr_ratio=sr_ratios[i],
                               qk_scale=qk_scale,
                               attn_drop=attn_drop_rate,
                               mlp_ratio=mlp_ratios[i],
                               drop=drop_rate,
                               norm_layer=norm_layer)
                for d in range(depths[i])]
            
            self.blocks.append(Sequential(*layers))
        
        self.pred = Conv2d(in_channels=embed_dims[-1],
                           out_channels=num_classes,
                           stride=1,
                           kernel_size=1,
                           padding=0)
        
        self.apply(_init_weights)

    def forward(self, x, feature_maps):

        feature_maps.reverse()

        for i in range(len(self.blocks)):
            x = self.fuse_blocks[i](x, feature_maps[i + 1])
            x = self.blocks[i](x)

        pred = self.pred(x)

        return pred

class VQSeg(Module):

    def __init__(self,
                 num_classes=2,
                 in_channels=3,
                 embed_dims=[32, 64, 128, 256],
                 num_heads=[1, 2, 4, 8],
                 mlp_ratios=[4, 4, 4, 4],
                 qkv_bias=True,
                 depths=[2, 2, 2, 2],
                 sr_ratios=[8, 4, 2, 1],
                 drop_rate=0.,
                 latent_dim=256,
                 num_vectors=1024,
                 beta=0.25):

        super().__init__()
        
        self.encoder = SegEncoder(in_channels=in_channels,
                                  embed_dims=embed_dims,
                                  depths=depths,
                                  num_heads=num_heads,
                                  qkv_bias=qkv_bias,
                                  sr_ratios=sr_ratios,
                                  drop_rate=drop_rate,
                                  mlp_ratios=mlp_ratios,
                                  latent_dim=latent_dim)
        self.quant_conv = Conv2d(in_channels=latent_dim,
                                 out_channels=latent_dim,
                                 kernel_size=1)
        self.codebook = Codebook(num_vectors=num_vectors,
                                 latent_dim=latent_dim,
                                 beta=beta)
        
        self.post_quant_conv = Conv2d(in_channels=latent_dim,
                                      out_channels=latent_dim,
                                      kernel_size=1)
        
        self.decoder = SegDecoder(embed_dims=embed_dims,
                                  depths=depths,
                                  num_heads=num_heads,
                                  qkv_bias=qkv_bias,
                                  sr_ratios=sr_ratios,
                                  drop_rate=drop_rate,
                                  mlp_ratios=[1, 1, 1, 1],
                                  num_classes=num_classes,
                                  latent_dim=latent_dim)

        self.apply(_init_weights)

    def forward(self, x):

        feature_maps = self.encoder(x)
        x = self.quant_conv(feature_maps[-1])
        x, x_indices, quant_loss = self.codebook(x)

        x = self.post_quant_conv(x)
        x = self.decoder(x, feature_maps)

        x = F.interpolate(x, scale_factor=2.0, mode='bilinear')

        return x, x_indices, quant_loss

    def encode(self, x):
        
        feature_maps = self.encoder(x)
        x = self.quant_conv(feature_maps[-1])
        x, x_indices, quant_loss = self.codebook(x)

        return x, x_indices, quant_loss

    def decode(self, x):

        x = self.post_quant_conv(x)
        x = self.decoder(x)

        return x

    def calculate_lambda(self, perceptual_loss, gan_loss):

        last_layer = self.decoder.model[-1]
        last_layer_weight = last_layer.weight
        perceptual_loss_grads = torch.autograd.grad(perceptual_loss, last_layer_weight, retain_graph=True)[0]
        gan_loss_grads = torch.autograd.grad(gan_loss, last_layer_weight, retain_graph=True)[0]

        λ = torch.norm(perceptual_loss_grads) / (torch.norm(gan_loss_grads) + 1e-4)
        λ = torch.clamp(λ, 0, 1e4).detach()

        return 0.8 * λ

    @staticmethod
    def adopt_weight(disc_factor, i, threshold, value=0.):
        if i < threshold:
            disc_factor = value
        return disc_factor

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path))

if __name__ == '__main__':

    model = VQSeg()
    input_tensor = torch.rand(size=(1, 3, 256, 256))
    x, x_indices, quant_loss = model(input_tensor)
    print(x.shape)

"""
PatchGAN Discriminator (https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py#L538)
"""

class Discriminator(Module):

    def __init__(self, image_channels=3, num_filters_last=64, n_layers=3):

        super().__init__()

        layers = [Conv2d(image_channels, num_filters_last,
                            kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2)]
        num_filters_mult = 1

        for i in range(1, n_layers + 1):
            num_filters_mult_last = num_filters_mult
            num_filters_mult = min(2 ** i, 8)
            layers += [
                Conv2d(num_filters_last * num_filters_mult_last,
                          num_filters_last * num_filters_mult,
                          kernel_size=4,
                          stride=2 if i < n_layers else 1,
                          padding=1,
                          bias=False),
                BatchNorm2d(num_filters_last * num_filters_mult),
                LeakyReLU(0.2, True)
            ]

        layers.append(Conv2d(num_filters_last * num_filters_mult, 1,
                                kernel_size=4, stride=1, padding=1))
        self.model = Sequential(*layers)

    def forward(self, x):
        return self.model(x)
