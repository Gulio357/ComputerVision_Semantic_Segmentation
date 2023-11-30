from einops import rearrange
from torch import einsum

from utils_2 import *



class OverlapPatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size, stride, padding):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.stride = stride
        self.padding = padding

        self.overlapped_patch = nn.Unfold(kernel_size=self.patch_size, stride=self.stride, padding=self.padding)
        self.embed = nn.Conv2d(self.in_channels * self.patch_size ** 2, self.out_channels, 1)

    def forward(self, patches):
        # patches.shape:  (B, C, H, W)
        H, W = patches.shape[-2:]

        x = patches
        x = self.overlapped_patch(x)
        num_patches = x.shape[-1]

        scalar = sqrt((H * W) / num_patches)
        x = rearrange(x, 'b c (h w) -> b c h w', h = int(H // scalar))

        out = self.embed(x)

        return out



class EfficientSelfAttention(nn.Module):
    def __init__(self, channels, reduction_ratio, num_heads):
        super().__init__()
        assert channels % num_heads == 0, f"channels {channels} should be divided by num_heads {num_heads}."

        self.num_heads = num_heads
        self.channels = channels
        self.reduction_ratio = reduction_ratio

        self.query_projection = nn.Conv2d(channels, channels, 1)
        self.key_projection = nn.Conv2d(channels, channels, 1)
        self.value_projection = nn.Conv2d(channels, channels, 1)

        self.out_projection = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        # x.shape:  (B, C, H, W)

        scale = (self.channels // self.num_heads) ** -0.5
        cur_h, cur_w = x.shape[-2:]

        q = self.query_projection(x)
        k = self.key_projection(x)
        v = self.value_projection(x)

        q = rearrange(q, 'b c h w -> b c (h w)', h = cur_h)
        k = rearrange(k, 'b c h w -> b c (h w)', h = cur_h)
        v = rearrange(v, 'b c h w -> b c (h w)', h = cur_h)

        q = rearrange(q, 'b (h c) n -> (b h) c n', h = self.num_heads)
        k = rearrange(k, 'b (h c) n -> (b h) c n', h = self.num_heads)
        v = rearrange(v, 'b (h c) n -> (b h) c n', h = self.num_heads)

        sim = einsum('b c i, b c j -> b i j', q, k) * scale


        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b c j -> b i c', attn, v)
        out = rearrange(out, '(b h) (x y) c -> b (h c) x y', h = self.num_heads, x = cur_h, y = cur_w)
        out = self.out_projection(out)

        return out
    

class DsConv(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size = 3, padding = 1):
        super(DsConv, self).__init__()
        self.depthwise = nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding, groups=dim_in, bias=True)
        self.pointwise = nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=True)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class MixFFN(nn.Module):
    def __init__(self, channels, expansion_factor):
        super().__init__()
        out_dim = channels * expansion_factor
        self.network = nn.Sequential(
            nn.Conv2d(channels, out_dim, 1),
            DsConv(out_dim, out_dim, kernel_size = 3, padding = 1),
            nn.GELU(),
            nn.Conv2d(out_dim, channels, 1)
        )

    def forward(self, x):
        # x: (B, C, H, W)
        return self.network(x)
    

class MixTransformerEncoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size, stride, padding,
                 n_layers, reduction_ratio, num_heads, expansion_factor):
        super().__init__()

        self.overlapped_patch = OverlapPatchMerging(in_channels, out_channels, patch_size, stride, padding)
        self.attn_layers = nn.ModuleList([])

        for _ in range(n_layers):
          self.attn_layers.append(
              nn.ModuleList
              ([
              EfficientSelfAttention(out_channels, reduction_ratio, num_heads),
              MixFFN(out_channels, expansion_factor)
              ])
              )

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.overlapped_patch(x)

        for (attn, mixff) in self.attn_layers:
          x = attn(x) + x
          x = mixff(x) + x

        return x
    

class MLPDecoder(nn.Module):
    def __init__(self, in_channels, embed_channels, out_dims, num_classes):
        super().__init__()

        self.mlp_layers = nn.ModuleList([])
        for i, in_channel in enumerate(in_channels):
            self.mlp_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, embed_channels, 1),
                    nn.Upsample(scale_factor = 2 ** i)
                )
            )


        # self.output_layer = nn.Sequential(
        #     nn.Conv2d(4 * embed_channels, num_classes, 1)
        # )

        self.output_layer = nn.Sequential(
            nn.Conv2d(4 * embed_channels, embed_channels, 1),
            nn.ReLU(),
            nn.Conv2d(embed_channels, num_classes, 1)
        )

    def forward(self, x):
        x_list = []

        for i, mlp_layer in enumerate(self.mlp_layers):
            x_list.append(mlp_layer(x[i]))

        x = torch.cat(x_list, dim = 1)
        out = self.output_layer(x)

        return out
