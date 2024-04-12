import math
import torch
from torch import nn

class matmul(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x1, x2):
        x = x1@x2
        return x

def count_matmul(m, x, y):
    num_mul = x[0].numel() * x[1].size(-1)
    # m.total_ops += torch.DoubleTensor([int(num_mul)])
    m.total_ops += torch.DoubleTensor([int(0)])

class PixelNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=2, keepdim=True) + 1e-8)

def gelu(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initialy created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def leakyrelu(x):
    return nn.functional.leaky_relu_(x, 0.2)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class CustomAct(nn.Module):
    def __init__(self, act_layer):
        super().__init__()
        # if act_layer == "gelu":
        #     self.act_layer = gelu
        # elif act_layer == "leakyrelu":
        #     self.act_layer = leakyrelu
        self.act_layer = gelu
        
    def forward(self, x):
        return self.act_layer(x)
        
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=gelu, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = CustomAct(act_layer)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mat = matmul()
        self.noise_strength_1 = torch.nn.Parameter(torch.zeros([]))

    def forward(self, x):
        B, N, C = x.shape
        x = x + torch.randn([x.size(0), x.size(1), 1], device=x.device) * self.noise_strength_1
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (self.mat(q, k.transpose(-2, -1))) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = self.mat(attn, v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CustomNorm(nn.Module):
    def __init__(self, norm_layer, dim):
        super().__init__()
        # self.norm_type = norm_layer
        # if norm_layer == "ln":
        #     self.norm = nn.LayerNorm(dim)
        # elif norm_layer == "bn":
        #     self.norm = nn.BatchNorm1d(dim)
        # elif norm_layer == "in":
        #     self.norm = nn.InstanceNorm1d(dim)
        # elif norm_layer == "pn":
        #     self.norm = PixelNorm(dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # if self.norm_type == "bn" or self.norm_type == "in":
        #     x = self.norm(x.permute(0,2,1)).permute(0,2,1)
        #     return x
        # elif self.norm_type == "none":
        #     return x
        # else:
        #     return self.norm(x)
        return self.norm(x)

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=gelu, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = CustomNorm(norm_layer, dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = CustomNorm(norm_layer, dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Discriminator(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, 
                 df_dim=64,
                 d_depth=7,
                 patch_size=4,
                 diff_aug=None,
                 img_size=32,
                 in_chans=3, num_classes=1, 
                 embed_dim=None, depth=7,
                 num_heads=4, mlp_ratio=4., qkv_bias=False, qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm):
        
        super().__init__()

        self.diff_aug = diff_aug

        self.num_classes = num_classes
        self.num_features = embed_dim = self.embed_dim = df_dim
        # num_features for consistency with other models

        depth = d_depth
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, 
            stride=patch_size, 
            padding=0)
        num_patches = (img_size // patch_size)**2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) # (1, 1, h)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        # num_patches = patch的个数；+1是给[cls]的额外的位置

        self.pos_drop = nn.Dropout(p=drop_rate)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  
        # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias, 
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], 
                norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)


        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        if self.diff_aug is not None:
            x = DiffAugment(x, self.diff_aug, True)
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).permute(0,2,1)

        cls_tokens = self.cls_token.expand(B, -1, -1)  
        # stole cls_tokens impl from Phil Wang, thanks

        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:,0]

    def forward(self, x):
        x = self.forward_features(x) # 若干transformer encoder blocks
        x = self.head(x) # 实现的是从[cls]的特征表示到二分类real/fake
        return x


if __name__ == "__main__":

    input_tensor = torch.rand(size=(8, 128, 32, 32))
    dis = Discriminator(None, img_size=32, in_chans=128, num_classes=3)
    output = dis(input_tensor)
    print(output.shape)