import math
import time
import numpy as np
from einops import rearrange

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.init import trunc_normal_, constant_


class EdgeDetect(nn.Module):

    def __init__(self) -> None:

        super().__init__()
        
        sobel_x_kernel = np.array([[-1, 0, 1],
                                   [-2, 0, 2],
                                   [-1, 0, 1]],
                                   dtype='float32')
    
        sobel_y_kernel = np.array([[1, 2, 1],
                                   [0, 0, 0],
                                   [-1, -2, -1]],
                                   dtype='float32')
        
        sobel_x_kernel = sobel_x_kernel.reshape((1, 1, 3, 3))
        sobel_y_kernel = sobel_y_kernel.reshape((1, 1, 3, 3))

        self.weight_x = nn.Parameter(torch.from_numpy(sobel_x_kernel),
                                     requires_grad=False)
        self.weight_y = nn.Parameter(torch.from_numpy(sobel_y_kernel),
                                     requires_grad=False)
        

    def forward(self, image):

        edge_x = F.conv2d(image, self.weight_x, stride=1, padding=1).abs()
        edge_y = F.conv2d(image, self.weight_y, stride=1, padding=1).abs()

        return edge_x + edge_y

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
        self.norm = nn.BatchNorm2d(num_features=out_channels)
        self.act = nn.GELU()

        self.apply(_init_weights)

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
        
        self.norm = nn.BatchNorm2d(num_features=embed_dim)

        self.act = nn.GELU()

        self.apply(_init_weights)

    def forward(self, x):

        # Embed Patch
        # [new]
        x = self.act(self.norm(self.proj(x)))

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
        
        self.apply(_init_weights)

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

    def __init__(self, dim, height=2, reduction=8, bias=False):
        
        super().__init__()
        
        self.height = height
        self.dim = dim
        d = max(int(dim / reduction), 4)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(dim, d, 1, padding=0, bias=bias),
            nn.LeakyReLU(0.2))

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, dim, kernel_size=1, stride=1,bias=bias))
        
        self.softmax = nn.Softmax(dim=1)

        self.weight_vale = torch.zeros(size=(1, height, 1, 1, 1))
        self.weight_vale[0, :] = 1
        self.weight = torch.nn.Parameter(self.weight_vale, requires_grad=True)

        self.apply(_init_weights)

    def forward(self, inp_feats):
        batch_size = inp_feats[0].shape[0]
        n_feats =  self.dim

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

class DomainFuse(nn.Module):

    def __init__(self, dim):
        
        super().__init__()

        self.spatial_attn = nn.Sequential(
            ConvModule(in_channels=dim, out_channels=dim,
                       kernel_size=3, stride=1, padding=1),
            ConvModule(in_channels=dim, out_channels=2,
                       kernel_size=1, stride=1, padding=0))
        
        self.softmax = nn.Softmax(dim=1)

        self.weight_vale = torch.zeros(size=(1, 2, 1, 1))
        self.weight_vale[0, :] = 1
        self.weight = torch.nn.Parameter(self.weight_vale, requires_grad=True)

        self.apply(_init_weights)

    def forward(self, inp_feats):

        attn_map = inp_feats[0] * inp_feats[1]
        # attn_map = attn_map.sum(dim=1, keepdim=True)

        attn_map = self.spatial_attn(attn_map)
        attn_map = self.softmax(attn_map)

        return inp_feats[0] * attn_map[:, 0 : 1, :, :] * self.weight[:, 0 : 1, :, :] + \
            inp_feats[1] * attn_map[:, 1 :, :, :] * self.weight[:, 1 :, :, :]
    
class DefaultMaxPool(nn.Module):

    def __init__(self, dim, *args, **kwargs) -> None:

        super().__init__()

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.apply(_init_weights)
    
    def forward(self, x):

        x = self.max_pool(x)

        return x

class SoftMaxPool(nn.Module):

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
                 sr_ratio=1) -> None:

        super().__init__()

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0,
                                     return_indices=True)
        self.max_unpool = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)

        self.soft_indice = ConvModule(in_channels=dim, out_channels=dim,
                                      kernel_size=3, stride=2, padding=1)
        
        self.drop = nn.Dropout()

        self.conv_down = nn.Sequential(
            AttentionBlock(dim=dim,
                           drop_path=drop_path, 
                           num_heads=num_heads,
                           qkv_bias=qkv_bias,
                           sr_ratio=sr_ratio,
                           qk_scale=qk_scale,
                           attn_drop=attn_drop,
                           mlp_ratio=mlp_ratio,
                           drop=drop,
                           norm_layer=norm_layer),
            AttentionBlock(dim=dim,
                           drop_path=drop_path, 
                           num_heads=num_heads,
                           qkv_bias=qkv_bias,
                           sr_ratio=sr_ratio,
                           qk_scale=qk_scale,
                           attn_drop=attn_drop,
                           mlp_ratio=mlp_ratio,
                           drop=drop,
                           norm_layer=norm_layer),
            ConvModule(in_channels=dim, out_channels=dim,
                        kernel_size=3, stride=2, padding=1)
        )
        
        self.apply(_init_weights)
    
    def forward(self, x):

        # soft max down
        x_max, x_indices = self.max_pool(x)
        
        ones = torch.ones_like(x_indices).to(x)
        ones = self.max_unpool(ones, x_indices)
        x_soft_max = x * ones + x
        x_soft_max = self.soft_indice(x_soft_max)

        # conv down
        x = self.drop(x_soft_max) + self.conv_down(x)

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
                 pooling_module=SoftMaxPool,
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
            rate.item() for rate in torch.linspace(0, drop_path_rate, sum(depths))]

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
            pool = pooling_module(dim=output_channel,
                                  drop_path=0., 
                                  num_heads=num_heads[i],
                                  qkv_bias=qkv_bias,
                                  sr_ratio=sr_ratios[i],
                                  qk_scale=qk_scale,
                                  attn_drop=attn_drop_rate,
                                  mlp_ratio=mlp_ratios[i],
                                  drop=drop_rate,
                                  norm_layer=norm_layer)
            
            self.blocks.append(nn.Sequential(path_embed, *layers, pool))
        
        # feature quant
        # self.encode_latent = nn.Conv2d(in_channels=embed_dims[-1],
        #                             out_channels=latent_dim,
        #                             stride=3,
        #                             kernel_size=1,
        #                             padding=1)
               
        self.apply(_init_weights)

    def forward(self, x):

        down = self.down2(self.down1(x))
        x = down

        feature_maps = []

        for block in self.blocks:
            # start = time.time()

            x = block(x)
            feature_maps.append(x)

            # print('block', time.time() - start)

        # latent_feature = self.encode_latent(x)
        # feature_maps.append(latent_feature)

        return down, feature_maps

class Codebook(nn.Module):

    def __init__(self, num_vectors, latent_dim):

        super().__init__()

        vqgan_from = 'train_crack500_viz/tunnel_crack500_vqgan_patch_reconstruct/version_0/checkpoints/step=100000.ckpt'
        vqgan_state_dict = torch.load(vqgan_from)['state_dict']
        embedding_weight = vqgan_state_dict['vqgan.codebook.embedding.weight']

        num_vectors, latent_dim = embedding_weight.shape

        self.num_vectors = num_vectors
        self.latent_dim = latent_dim

        self.embedding = nn.Embedding(self.num_vectors, self.latent_dim)
        # self.embedding.weight.data.uniform_(-1.0 / self.num_vectors, 1.0 / self.num_vectors)

        self.embedding.weight.data.copy_(embedding_weight)
        
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

        x_q_detach = x + (x_q - x).detach()
        x_q_detach = rearrange(x_q_detach, 'b h w c -> b c h w')

        return {'x_q_detach': x_q_detach,
                'x_q': x_q,
                'x_fea': x}

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

class SegDecoder(nn.Module):

    def __init__(self,
                 embed_dims=[],     # Latent dim num
                 linear_dim=256,    # Latent dim num
                 num_classes=2,     # Seg class num
                 is_seg_refine=False,
                 ):
        
        super().__init__()

        self.is_seg_refine = is_seg_refine

        self.linear_list = nn.ModuleList([])
        self.concat_list = nn.ModuleList([])

        embed_dims = [embed_dims[0]] + embed_dims

        for index in range(1, len(embed_dims)):

            linear = nn.Sequential(
                nn.Linear(in_features=embed_dims[index],
                          out_features=embed_dims[index - 1]),
            )
            self.linear_list.append(linear)

            if index == 1:
                concat = nn.Identity()
            else:
                concat = nn.Sequential(
                    ConvModule(in_channels=embed_dims[index - 1] * 2,
                               out_channels=embed_dims[index - 1],
                               kernel_size=1,
                               stride=1,
                               padding=0),
                    CBAM(gate_channels=embed_dims[index - 1]))
        
            self.concat_list.append(concat)

        self.aug = ConvModule(in_channels=embed_dims[0],
                                 out_channels=linear_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        
        self.seg = ConvModule(in_channels=linear_dim,
                              out_channels=num_classes)

        if self.is_seg_refine:

            dim = linear_dim + 32
            # self.edge_detect = EdgeDetect()
            self.scale3 = DWConv(in_channels=dim,
                            out_channels=embed_dims[0],
                            kernel_size=3,
                            stride=1,
                            padding=1)
            self.scale5 = DWConv(in_channels=dim,
                                out_channels=embed_dims[0],
                                kernel_size=5,
                                stride=1,
                                padding=2)
            self.scale7 = DWConv(in_channels=dim,
                                out_channels=embed_dims[0],
                                kernel_size=7,
                                stride=1,
                                padding=3)
            self.edge_aug = ConvModule(in_channels=embed_dims[0] * 3,
                                 out_channels=linear_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
            # self.edge_concat = ConvModule(in_channels=linear_dim * 2,
            #                      out_channels=linear_dim,
            #                      kernel_size=1,
            #                      stride=1,
            #                      padding=0)

        self.apply(_init_weights)

    def forward(self, x, feas):

        for index in range(len(feas) - 1, -1, -1):

            linear = self.linear_list[index]
            fea = feas[index]

            B, C, H, W = fea.shape
            fea = rearrange(fea, 'b c h w -> b (h w) c')
            fea = linear(fea)
            fea = rearrange(fea, 'b (h w) c -> b c h w', h=H, w=W)

            if index > 0:
                concat = self.concat_list[index]
                fea_up = F.interpolate(fea,
                                       size=feas[index - 1].shape[2 : ],
                                       mode='bilinear',
                                       align_corners=True)
                fea_up = torch.concat([fea_up, feas[index - 1]], dim=1)
                feas[index - 1] = feas[index - 1] + concat(fea_up)
            else:
                fea = F.interpolate(fea,
                                    scale_factor=2,
                                    mode='bilinear',
                                    align_corners=True)
        
        fea = self.aug(fea)
        seg = self.seg(fea)

        origin_seg = seg

        if self.is_seg_refine:
            # [new]
            # crack_seg = seg.softmax(dim=1)[:, 1 : 2, :, :]
            # edge_mask = self.edge_detect(crack_seg)

            image_edge = torch.concat([x, fea], dim=1)
            image_edge3 = self.scale3(image_edge)
            image_edge5 = self.scale3(image_edge)
            image_edge7 = self.scale5(image_edge)

            image_edge = self.edge_aug(torch.concat([image_edge3, image_edge5, image_edge7], dim=1))

            # fea = self.edge_concat(torch.concat([fea, image_edge], dim=1)) + fea

            seg = self.seg(image_edge)

        # return seg, image_edge
        return seg, origin_seg


class DWConv(nn.Module):

    def __init__(self,
                 in_channels, out_channels, 
                 kernel_size, stride=3, padding=1):
        
        super().__init__()
        
        self.dconv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
        )
        self.norm1 = nn.GroupNorm(num_groups=in_channels,
                                  num_channels=in_channels)
        self.act1 = nn.GELU()
        self.pconv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
        )
        self.norm2 = nn.GroupNorm(num_groups=1,
                                  num_channels=out_channels)
        self.act2 = nn.GELU()

        self.apply(_init_weights)
 
    def forward(self, x):
        x = self.act1(self.norm1(self.dconv(x)))
        x = self.act2(self.norm2(self.pconv(x)))
        return x


class AVQSeg(nn.Module):

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
                 pooling_module='soft',
                 decode_dim=256,
                 is_feature_quant=True, # vector quant
                 embedding_frozen=False,
                 quant_fuse_module='spatial_fuse',
                 quant_dim=1,
                 num_vectors=1024,
                 latent_dim=256,
                 is_seg_refine=True, # seg refine
                 sample_size=1,
                 is_edge_weight=False,
                 sample_beta=0.25
                 ):

        super().__init__()

        self.is_feature_quant = is_feature_quant
        self.quant_dim = quant_dim
        self.is_seg_refine = is_seg_refine
        self.is_edge_weight = is_edge_weight

        # if use soft max pooling
        if pooling_module == 'soft':
            pooling_module = SoftMaxPool
        else:
            pooling_module = DefaultMaxPool
        
        if quant_fuse_module == 'channel':
            quant_fuse_module = SKFF
        elif quant_fuse_module == 'spatial':
            quant_fuse_module = DomainFuse
        
        self.encoder = SegEncoder(in_channels=in_channels,
                                  embed_dims=embed_dims,
                                  depths=depths,
                                  num_heads=num_heads,
                                  qkv_bias=qkv_bias,
                                  sr_ratios=sr_ratios,
                                  drop_rate=drop_rate,
                                  mlp_ratios=mlp_ratios,
                                  latent_dim=latent_dim,
                                  pooling_module=pooling_module)

        # if quantilize features of last layer
        if is_feature_quant:

            self.quant_conv = nn.Sequential(
                ConvModule(in_channels=embed_dims[quant_dim],
                           out_channels=latent_dim,
                           kernel_size=1,
                           stride=1,
                           padding=0))
            
            self.quant_gap = nn.AdaptiveAvgPool2d(output_size=1)
            # [new]
            self.quant_scale = nn.Sequential(
                nn.Conv2d(in_channels=latent_dim, out_channels=latent_dim // 16, kernel_size=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(in_channels=latent_dim // 16, out_channels=latent_dim, kernel_size=1, bias=False),
                nn.Sigmoid())
            
            self.quant_shift = nn.Sequential(
                nn.Conv2d(in_channels=latent_dim, out_channels=latent_dim // 16, kernel_size=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(in_channels=latent_dim // 16, out_channels=latent_dim, kernel_size=1, bias=False),
                nn.Sigmoid())
            
            self.codebook = Codebook(num_vectors=num_vectors,
                                     latent_dim=latent_dim)
            if embedding_frozen:
                self.codebook.embedding.requires_grad_(False)
        
            self.post_quant_conv = nn.Sequential(
                ConvModule(in_channels=latent_dim,
                           out_channels=embed_dims[quant_dim],
                           kernel_size=1,
                           stride=1,
                           padding=0))
            
            self.quant_fuse = quant_fuse_module(dim=embed_dims[quant_dim])

        self.decoder = SegDecoder(embed_dims=embed_dims,
                                  linear_dim=decode_dim,
                                  num_classes=num_classes,
                                  is_seg_refine=is_seg_refine)
        
        self.apply(_init_weights)


    def forward(self, x):

        x_down, x_feas = self.encoder(x)

        if self.is_feature_quant:

            x_fea = x_feas[self.quant_dim]
            x_q = self.quant_conv(x_fea)
            x_q_gap = self.quant_gap(x_q)
            x_q_scale = self.quant_scale(x_q_gap) + 1e-6
            x_q_shift = self.quant_shift(x_q_gap)
            x_q = x_q_scale * x_q + x_q_shift

            x_q_res = self.codebook(x_q)

            x_q = (x_q_res['x_q_detach'] - x_q_shift) / x_q_scale
            x_q = self.post_quant_conv(x_q)

            # [new]
            x_feas[self.quant_dim] = self.quant_fuse([x_fea, x_q]) + x_fea

        seg, origin_seg = self.decoder(x_down, x_feas)
        # seg, image_edge = self.decoder(x_down, x_feas)

        seg = F.interpolate(seg, size=x.shape[2:],
                            mode='bilinear',
                            align_corners=True)
        
        origin_seg = F.interpolate(origin_seg, size=x.shape[2:],
                                   mode='bilinear',
                                   align_corners=True)

        # image_edge = F.interpolate(image_edge, size=x.shape[2:],
        #                            mode='bilinear',
        #                            align_corners=True)

        return {'seg': seg,
                'x_fea': None if not self.is_feature_quant else x_q_res['x_fea'],
                'x_q': None if not self.is_feature_quant else x_q_res['x_q'],
                'points': None,
                'rend': None,
                'origin_seg': origin_seg if self.is_seg_refine and self.is_edge_weight else None
        }

if __name__ == '__main__':

    from thop import profile, clever_format

    model = AVQSeg(num_classes=2,
                   in_channels=3,
                   embed_dims=[32, 64, 128, 256],
                   num_heads=[1, 2, 4, 8],
                   mlp_ratios=[4, 4, 4, 4],
                   qkv_bias=True,
                   depths=[2, 2, 2, 2],
                   sr_ratios=[8, 4, 2, 1],
                   drop_rate=0.,
                   pooling_module='soft',
                   decode_dim=256,
                   is_feature_quant=True, # vector quant
                   embedding_frozen=False,
                   quant_fuse_module='spatial',
                   quant_dim=3,
                   num_vectors=1024,
                   latent_dim=256,
                   is_seg_refine=True,
                   sample_size=1,
                   is_edge_weight=False,
                   sample_beta=0.75)

    x = torch.randn(size=(1, 3, 512, 512))

    # start = time.time()
    # model(x)
    # print(time.time() - start)

    flops, params = profile(model, inputs=(x,))
    flops, params = clever_format([flops, params], '%.3f')
    print(f'Params: {params} FLOPs: {flops}') # Params: 10.138M FLOPs: 54.398G

    #-----------------------------------
    # test vqgan codebook
    #-----------------------------------

    # vqgan_from = 'train_crack500_viz/tunnel_crack500_vqgan_patch_reconstruct/version_0/checkpoints/step=100000.ckpt'
    # vqgan_state_dict = torch.load(vqgan_from)['state_dict']
    # embedding_weight = vqgan_state_dict['vqgan.codebook.embedding.weight']
    # print(embedding_weight.shape)


    
