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
import torch.nn as nn
from torch.nn.init import trunc_normal_, constant_

def _init_weights(m):

    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            constant_(m.bias, 0)

    elif isinstance(m, nn.LayerNorm):
        constant_(m.bias, 0)
        constant_(m.weight, 1.0)
    
    elif isinstance(m, nn.Conv2d):
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

class DropPath(nn.Module):

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

class ConvModule(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 groups=1,
                 bias=False):
        
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels=in_channels,
                           out_channels=out_channels,
                           kernel_size=kernel_size,
                           stride=stride,
                           padding=padding,
                           groups=groups,
                           bias=bias)
        self.norm = nn.GroupNorm(num_groups=1,
                              num_channels=out_channels)
        self.act = nn.GELU()

    def forward(self, x):

        return self.act(self.norm(self.conv(x)))

class OverlapPatchEmbed(nn.Module):
    '''
    Overlap Patch nn.Embedding
    '''
    
    def __init__(self,
                 in_channel,
                 embed_dim,
                 patch_size,
                 stride,
                 ):
    
        super().__init__()
    
        padding = patch_size // 2

        self.proj = nn.Conv2d(in_channels=in_channel,
                           out_channels=embed_dim,
                           kernel_size=patch_size,
                           stride=stride,
                           padding=padding)
        
        self.norm = nn.LayerNorm(normalized_shape=embed_dim)

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

class Attention(nn.Module):
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

        self.q = nn.Linear(in_features=dim,
                        out_features=dim,
                        bias=qkv_bias)
        
        # Smaller Key & Value
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr     = nn.Conv2d(in_channels=dim, out_channels=dim,
                                 kernel_size=sr_ratio, stride=sr_ratio)
            self.norm   = nn.LayerNorm(normalized_shape=dim)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        
        self.attn_drop  = nn.Dropout(attn_drop)
        
        self.proj       = nn.Linear(dim, dim)
        self.proj_drop  = nn.Dropout(proj_drop)

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

class DepthwiseConv(nn.Module):

    def __init__(self, dim):

        super().__init__()
        
        self.dwconv = nn.Conv2d(in_channels=dim,
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
    
class MLP(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        
        super().__init__()
        
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features=in_features,
                          out_features=hidden_features)
        self.dwconv = DepthwiseConv(dim=hidden_features)
        self.act = act_layer()
        
        self.fc2 = nn.Linear(in_features=hidden_features,
                          out_features=out_features)
        self.drop = nn.Dropout(drop)

        self.apply(_init_weights)

    def forward(self, x, H, W):
        
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        
        x = self.fc2(x)
        x = self.drop(x)
        
        return x

class AttentionBlock(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
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

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

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

class SKFF(nn.Module):

    def __init__(self, in_channels, height=3, reduction=8, bias=False):
        
        super().__init__()
        
        self.height = height
        d = max(int(in_channels / reduction), 4)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(in_channels, d, 1, padding=0, bias=bias),
            nn.LeakyReLU(0.2))

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

class SoftMaxPool(nn.Module):

    def __init__(self, dim) -> None:

        super().__init__()

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0,
                                     return_indices=True)
        self.max_unpool = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)

        self.soft_indice = ConvModule(in_channels=dim, out_channels=dim,
                                      kernel_size=3, stride=2, padding=1)
        
        self.conv_down = ConvModule(in_channels=dim, out_channels=dim,
                                    kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):

        # soft max down
        x_max, x_indices = self.max_pool(x)
        
        ones = torch.ones_like(x_indices).to(x)
        ones = self.max_unpool(ones, x_indices)
        x_soft_max = x * ones + x
        x_soft_max = self.soft_indice(x_soft_max)

        # conv down
        x = x_soft_max + self.conv_down(x)

        return x

class SegEncoder(nn.Module):

    def __init__(self,
                 in_channels=3,
                 embed_dims=[32, 64, 160, 256], # AttentionBlock dim
                 depths=[3, 4, 6, 3],           # AttentionBlock num
                 drop_path_rate=0.,             # AttentionBlock drop path rate
                 num_heads=[1, 2, 4, 8],        # Attention head number
                 qkv_bias=False,                # Attention nn.Linear bias
                 sr_ratios=[8, 4, 2, 1],        # Attention K, V size ratio
                 qk_scale=None,                 # Attention value scale
                 attn_drop_rate=0.,             # Attention Q, K dropout
                 drop_rate=0.,                  # Attention + MLP last dropout
                 mlp_ratios=[4, 4, 4, 4],       # MLP hidden dim ratio
                 norm_layer=nn.LayerNorm,          # Attention + MLP norm type
                 latent_dim=512,                # Latent dim
                 ):
        
        super().__init__()

        # down sample 1/4
        self.down1 = ConvModule(in_channels=in_channels,
                                out_channels=embed_dims[0],
                                kernel_size=3, stride=2, padding=1,
                                bias=False)
        self.down2 = ConvModule(in_channels=embed_dims[0],
                                out_channels=embed_dims[0],
                                kernel_size=3, stride=2, padding=1,
                                bias=False)

        # seg blocks
        self.blocks = nn.ModuleList([])
        
        drop_path_rate_list = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        input_channel = embed_dims[0]

        for i in range(len(embed_dims)):

            output_channel = embed_dims[i]

            # get patch embeding
            path_embed = OverlapPatchEmbed(in_channel=input_channel,
                                           embed_dim=output_channel,
                                           patch_size=7, stride=1)
            # update input channel
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

            # down sample 1/2
            pool = SoftMaxPool(dim=output_channel)
            
            self.blocks.append(nn.Sequential(path_embed, *layers, pool))
        
        # feature quant
        self.encode_latent = nn.Conv2d(in_channels=embed_dims[-1],
                                    out_channels=latent_dim,
                                    stride=3,
                                    kernel_size=1,
                                    padding=1)
               
        self.apply(_init_weights)

    def forward(self, x):

        x = self.down2(self.down1(x))

        feature_maps = []

        for block in self.blocks:
            x = block(x)
            feature_maps.append(x)

        latent_feature = self.encode_latent(x)
        feature_maps.append(latent_feature)

        return feature_maps

class Codebook(nn.Module):

    def __init__(self, num_vectors, latent_dim, beta):

        super().__init__()

        self.num_vectors = num_vectors
        self.latent_dim = latent_dim
        self.beta = beta

        self.embedding_key = nn.Embedding(self.num_vectors, self.latent_dim)
        self.embedding_key.weight.data.uniform_(-1.0 / self.num_vectors, 1.0 / self.num_vectors)

        orthogonal_tensor = torch.empty(self.latent_dim, self.latent_dim)
        torch.nn.init.orthogonal_(orthogonal_tensor, gain=1)
        self.embedding_value = nn.Parameter(orthogonal_tensor)

    def forward(self, x):

        x = rearrange(x, 'b c h w -> b h w c')
        x_flattened = rearrange(x, 'b h w c -> (b h w) c')

        # (b x h x w, num_vectors)
        d = torch.sum(x_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding_key.weight ** 2, dim=1) - \
            2 * (torch.matmul(x_flattened, self.embedding_key.weight.t()))

        # (b x h x w, 1)
        min_encoding_indices = torch.argmin(d, dim=1)
        # (b x h x w, latent_dim)
        x_q = self.embedding_key(min_encoding_indices)
        # (b, h, w, latent_dim)
        x_q = (x_q @ self.embedding_value).view(x.shape)

        x_q_grad = x_q
        x_q_grad = rearrange(x_q_grad, 'b h w c -> b c h w')

        loss = torch.mean((x_q.detach() - x) ** 2) + self.beta * torch.mean((x_q - x.detach()) ** 2)

        x_q = x + (x_q - x).detach()
        x_q = rearrange(x_q, 'b h w c -> b c h w')

        return x_q, x_q_grad, min_encoding_indices, loss

class SegDecoder(nn.Module):

    def __init__(self,
                 embed_dims=[],     # Latent dim num
                 linear_dim=256,    # Latent dim num
                 num_classes=2,     # Seg class num
                 ):
        
        super().__init__()

        self.linear_list = nn.ModuleList([])

        for dim in embed_dims:

            linear = nn.Linear(in_features=dim, out_features=linear_dim)
            self.linear_list.append(linear)
        
        self.concat = ConvModule(in_channels=(linear_dim * len(embed_dims)),
                                 out_channels=linear_dim,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
        
        self.seg = ConvModule(in_channels=linear_dim,
                              out_channels=num_classes)
        
        self.point_head = PointHead(num_classes=num_classes,
                                    in_c=linear_dim)
        
        
        self.apply(_init_weights)

    def forward(self, x, image):

        self.output_size = x[0].shape[2:]

        for index in range(len(x)):

            linear = self.linear_list[index]
            fea = x[index]

            B, C, H, W = fea.shape
            fea = rearrange(fea, 'b c h w -> b (h w) c')
            fea = linear(fea)
            fea = rearrange(fea, 'b (h w) c -> b c h w', h=H, w=W)

            x[index] = F.interpolate(fea, size=self.output_size, mode='bilinear')

        x = self.concat(torch.concat(x, dim=1))

        pred = self.seg(x)

        # point rend
        points = self.point_head(image, x, pred)
        
        if not self.training:
            pred = points["fine"]

        return pred, points

def point_sample(input, point_coords, **kwargs):
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output


@torch.no_grad()
def sampling_points(mask, N, k=3, beta=0.75, training=True):
    assert mask.dim() == 4, "Dim must be N(Batch)CHW"
    device = mask.device
    B, _, H, W = mask.shape
    mask, _ = mask.sort(1, descending=True)

    if not training:
        H_step, W_step = 1 / H, 1 / W
        N = min(H * W, N)
        uncertainty_map = -1 * (mask[:, 0] - mask[:, 1])
        _, idx = uncertainty_map.view(B, -1).topk(N, dim=1)

        points = torch.zeros(B, N, 2, dtype=torch.float, device=device)
        points[:, :, 0] = W_step / 2.0 + (idx  % W).to(torch.float) * W_step
        points[:, :, 1] = H_step / 2.0 + (idx // W).to(torch.float) * H_step
        return idx, points

    over_generation = torch.rand(B, k * N, 2, device=device)
    over_generation_map = point_sample(mask, over_generation, align_corners=False)

    uncertainty_map = -1 * (over_generation_map[:, 0] - over_generation_map[:, 1])
    _, idx = uncertainty_map.topk(int(beta * N), -1)

    shift = (k * N) * torch.arange(B, dtype=torch.long, device=device)

    idx += shift[:, None]

    importance = over_generation.view(-1, 2)[idx.view(-1), :].view(B, int(beta * N), 2)
    coverage = torch.rand(B, N - int(beta * N), 2, device=device)
    return torch.cat([importance, coverage], 1).to(device)

# https://blog.csdn.net/weixin_42028608/article/details/105379233
class PointHead(nn.Module):
    def __init__(self,num_classes, in_c=512, k=3, beta=0.75):
        super().__init__()
        self.mlp = nn.Conv1d(in_c+num_classes, num_classes, 1)
        self.k = k
        self.beta = beta

    def forward(self, x, res2, out):
        """
        1. Fine-grained features are interpolated from res2 for DeeplabV3
        2. During training we sample as many points as there are on a stride 16 feature map of the input
        3. To measure prediction uncertainty
           we use the same strategy during training and inference: the difference between the most
           confident and second most confident class probabilities.
        """
        if not self.training:
            return self.inference(x, res2, out)

        points = sampling_points(out, x.shape[-1] // 16, self.k, self.beta)

        coarse = point_sample(out, points, align_corners=False)
        fine = point_sample(res2, points, align_corners=False)

        feature_representation = torch.cat([coarse, fine], dim=1)

        rend = self.mlp(feature_representation)

        return {"rend": rend, "points": points}
    
    @torch.no_grad()
    def inference(self, x, res2, out):
        """
        During inference, subdivision uses N=8096
        (i.e., the number of points in the stride 16 map of a 1024×2048 image)
        """
        num_points = 8096

        while out.shape[-1] != x.shape[-1]:
            out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=True)

            points_idx, points = sampling_points(out, num_points, training=self.training)

            coarse = point_sample(out, points, align_corners=False)
            fine = point_sample(res2, points, align_corners=False)

            feature_representation = torch.cat([coarse, fine], dim=1)

            rend = self.mlp(feature_representation)

            B, C, H, W = out.shape
            points_idx = points_idx.unsqueeze(1).expand(-1, C, -1)
            out = (out.reshape(B, C, -1).scatter_(2, points_idx, rend).view(B, C, H, W))

        return {"fine": out, "rend": rend, "points": points}
    

class VQSeg(nn.Module):

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
                 num_vectors=256,
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
        self.quant_conv = nn.Conv2d(in_channels=latent_dim,
                                 out_channels=latent_dim,
                                 kernel_size=1)
        self.codebook = Codebook(num_vectors=num_vectors,
                                 latent_dim=latent_dim,
                                 beta=beta)
        
        self.post_quant_conv = nn.Conv2d(in_channels=latent_dim,
                                      out_channels=latent_dim,
                                      kernel_size=1)
        
        self.skff = SKFF(in_channels=latent_dim,
                         height=2)
        
        self.decoder = SegDecoder(embed_dims=(embed_dims + [latent_dim]),
                                  linear_dim=256,
                                  num_classes=num_classes)
        
        self.discriminator = Discriminator(image_channels=num_classes)

        
        self.apply(_init_weights)

    def forward(self, x):

        image = x
        output_size = x.shape[2:]

        x = self.encoder(x)
        
        x_quant = self.quant_conv(x[-1])
        x_quant, x_quant_grad, x_indices, quant_loss = self.codebook(x_quant)
        x_quant = self.post_quant_conv(x_quant)
        x[-1] = self.skff([x_quant, x[-1]])

        x, points = self.decoder(x, image)

        # quant feature
        x_quant_grad = self.post_quant_conv(x_quant_grad)
        B, C, H, W = x_quant_grad.shape
        x_quant_grad = rearrange(x_quant_grad, 'b c h w -> b (h w) c')
        x_quant_grad = self.decoder.linear_list[-1](x_quant_grad)
        x_quant_grad = rearrange(x_quant_grad, 'b (h w) c -> b c h w', h=H, w=W)
        x_quant_grad = F.interpolate(x_quant_grad, size=x.shape[2:], mode='bilinear')
        x_quant_grad = self.decoder.seg(x_quant_grad)
        disc = self.discriminator(x_quant_grad)

        x = F.interpolate(x, size=output_size, mode='bilinear')

        return x, x_indices, quant_loss, disc, points


class Discriminator(nn.Module):

    def __init__(self,
                 image_channels=3,
                 num_filters_last=64,
                 n_layers=3):

        super().__init__()

        layers = [
            nn.Conv2d(image_channels, num_filters_last,
                      kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2)]
        num_filters_mult = 1

        for i in range(1, n_layers + 1):
            num_filters_mult_last = num_filters_mult
            num_filters_mult = min(2 ** i, 8)
            layers += [
                nn.Conv2d(num_filters_last * num_filters_mult_last,
                          num_filters_last * num_filters_mult,
                          kernel_size=3,
                          stride=2 if i < n_layers else 1,
                          padding=1,
                          bias=False),
                nn.BatchNorm2d(num_filters_last * num_filters_mult),
                nn.LeakyReLU(0.2, True)
            ]

        layers += [
            nn.Conv2d(num_filters_last * num_filters_mult, 1,
                      kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()]
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':

    model = VQSeg()
    input_tensor = torch.rand(size=(1, 3, 256, 256))
    x, x_indices, quant_loss, _, _ = model(input_tensor)
    print(x.shape)