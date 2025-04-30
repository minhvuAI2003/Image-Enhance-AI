# Full Restormer Model with Deformable Fusion (Optimized DeformableConv2d)
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------- GDFN ---------------------
class GDFN(nn.Module):
    def __init__(self, channels, expansion_factor=2.66):
        super(GDFN, self).__init__()
        hidden_channels = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, 1, bias=False)
        self.dwconv = nn.Conv2d(hidden_channels * 2, hidden_channels * 2, 3, padding=1, groups=hidden_channels * 2, bias=False)
        self.project_out = nn.Conv2d(hidden_channels, channels, 1, bias=False)

    def h_swish(self, x):
        return x * F.relu6(x + 3, inplace=True) / 6

    def forward(self, x):
        x1, x2 = self.dwconv(self.project_in(x)).chunk(2, dim=1)
        gated = self.h_swish(x1) * x2
        x = self.project_out(gated)
        return x

# --------------------- MDTA ---------------------
class MDTA(nn.Module):
    def __init__(self, channels, num_heads):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.qkv_conv = nn.Conv2d(channels * 3, channels * 3, 3, padding=1, groups=channels * 3, bias=False)
        self.project_out = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv_conv(self.qkv(x)).chunk(3, dim=1)
        q = q.reshape(b, self.num_heads, -1, h*w)
        k = k.reshape(b, self.num_heads, -1, h*w)
        v = v.reshape(b, self.num_heads, -1, h*w)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.temperature
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).reshape(b, -1, h, w)
        out = self.project_out(out)
        return out

# --------------------- Multi-Stream Transformer Block ---------------------
class MultiStreamTransformerBlock(nn.Module):
    def __init__(self, channels, num_heads, expansion_factor=2.66, res_scale=0.1, streams=2):
        super(MultiStreamTransformerBlock, self).__init__()
        self.streams = nn.ModuleList([MDTA(channels, num_heads) for _ in range(streams)])
        self.stream_weights = nn.Parameter(torch.ones(streams))
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = GDFN(channels, expansion_factor)
        self.res_scale = res_scale

    def forward(self, x):
        b, c, h, w = x.shape
        x_flat = x.permute(0, 2, 3, 1)
        x_norm1 = self.norm1(x_flat).permute(0, 3, 1, 2)
        outs = torch.stack([stream(x_norm1) for stream in self.streams], dim=0)
        weights = F.softmax(self.stream_weights, dim=0)
        attn_out = (weights.view(-1, 1, 1, 1, 1) * outs).sum(dim=0)
        x = x + attn_out * self.res_scale
        identity = x
        x_flat = x.permute(0, 2, 3, 1)
        x_norm2 = self.norm2(x_flat).permute(0, 3, 1, 2)
        out = self.ffn(x_norm2)
        x = identity + out * self.res_scale
        return x

class DeformableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=True):
        super(DeformableConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.regular_conv = nn.Conv2d(in_channels * kernel_size * kernel_size, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)

    def forward(self, x):
        offset = self.offset_conv(x)
        ks = self.kernel_size
        b, c, h, w = x.size()

        if self.padding:
            x = F.pad(x, (self.padding, self.padding, self.padding, self.padding))

        p0_x, p0_y = torch.meshgrid(
            torch.arange(h, dtype=torch.float32, device=x.device),
            torch.arange(w, dtype=torch.float32, device=x.device),
            indexing='ij'
        )
        p0 = torch.stack((p0_y, p0_x), dim=-1)  # (h, w, 2)
        p0 = p0.unsqueeze(0).unsqueeze(1)  # (1,1,h,w,2)

        p_n = self._get_pn(ks, h, w, x.device)  # (1,ks*ks,h,w,2)

        offset = offset.view(b, ks * ks, 2, h, w).permute(0,1,3,4,2)
        p = p0 + p_n + offset

        p = p.view(b, ks * ks, h, w, 2)
        p_norm = torch.zeros_like(p)
        p_norm[..., 0] = 2 * p[..., 0] / (w - 1) - 1
        p_norm[..., 1] = 2 * p[..., 1] / (h - 1) - 1

        p_norm = p_norm.permute(0,2,3,1,4).contiguous().view(b, h*ks, w*ks, 2)

        x_sampled = F.grid_sample(x, p_norm, mode='bilinear', padding_mode='zeros', align_corners=True)
        x_sampled = x_sampled.view(b, c, h, w, ks*ks)
        x_sampled = x_sampled.permute(0, 1, 4, 2, 3).contiguous()
        x_sampled = x_sampled.view(b, c * ks * ks, h, w)

        out = self.regular_conv(x_sampled)
        return out

    def _get_pn(self, ks, h, w, device):
        p_n_x, p_n_y = torch.meshgrid(
            torch.linspace(-(ks-1)//2, (ks-1)//2, steps=ks, device=device),
            torch.linspace(-(ks-1)//2, (ks-1)//2, steps=ks, device=device),
            indexing='ij'
        )
        p_n = torch.stack((p_n_y, p_n_x), dim=-1)  # (ks, ks, 2)
        p_n = p_n.view(1, ks*ks, 1, 1, 2).repeat(1, 1, h, w, 1)
        return p_n


# --------------------- Deformable Fusion ---------------------
class DeformableFusion(nn.Module):
    def __init__(self, channels):
        super(DeformableFusion, self).__init__()
        self.encoder_proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.decoder_proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.deform_conv = DeformableConv2d(channels*2, channels, kernel_size=3, padding=1)
        self.output_proj = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, encoder_feat, decoder_feat):
        enc = self.encoder_proj(encoder_feat)
        dec = self.decoder_proj(decoder_feat)
        fused = torch.cat([enc, dec], dim=1)
        out = self.deform_conv(fused)
        out = self.output_proj(out)
        return out

class DeformableFusion_1(nn.Module):
    def __init__(self, channels):
        super(DeformableFusion_1, self).__init__()
        self.encoder_proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.decoder_proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.deform_conv = DeformableConv2d(channels*2, channels, kernel_size=3, padding=1)
        self.output_proj = nn.Conv2d(channels, channels*2, 1, bias=False)

    def forward(self, encoder_feat, decoder_feat):
        enc = self.encoder_proj(encoder_feat)
        dec = self.decoder_proj(decoder_feat)
        fused = torch.cat([enc, dec], dim=1)
        out = self.deform_conv(fused)
        out = self.output_proj(out)
        return out

# --------------------- DownSample / UpSample ---------------------
class DownSample(nn.Module):
    def __init__(self, channels):
        super(DownSample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1, bias=False),
            nn.PixelUnshuffle(2)
        )

    def forward(self, x):
        return self.body(x)

class UpSample(nn.Module):
    def __init__(self, channels):
        super(UpSample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 3, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.body(x)

# --------------------- Full Model ---------------------
class Restormer(nn.Module):
    def __init__(self, num_blocks=[4,6,6,8], num_heads=[1,2,4,8], channels=[48,96,192,384], expansion_factor=2.66, num_refinement=4, res_scale=0.1):
        super(Restormer, self).__init__()
        self.embed = nn.Conv2d(3, channels[0], kernel_size=3, padding=1, bias=False)

        self.encoders = nn.ModuleList([
            nn.Sequential(*[MultiStreamTransformerBlock(c, h, expansion_factor, res_scale) for _ in range(b)])
            for c, h, b in zip(channels, num_heads, num_blocks)
        ])

        self.downs = nn.ModuleList([DownSample(c) for c in channels[:-1]])
        self.ups = nn.ModuleList([UpSample(num_ch) for num_ch in list(reversed(channels))[:-1]])
        self.reduces = nn.ModuleList([
            nn.Conv2d(channels[i], channels[i-1], kernel_size=1, bias=False)
            for i in reversed(range(2, len(channels)))
        ])

        self.cross_scale_attn = nn.ModuleList([
            DeformableFusion(channels[i]) for i in reversed(range(1, len(channels)-1))
        ])
        self.cross_scale_attn_bottleneck = DeformableFusion_1(channels[0])

        self.decoders = nn.ModuleList([
            nn.Sequential(*[MultiStreamTransformerBlock(channels[2], num_heads[2], expansion_factor) for _ in range(num_blocks[2])]),
            nn.Sequential(*[MultiStreamTransformerBlock(channels[1], num_heads[1], expansion_factor) for _ in range(num_blocks[1])]),
            nn.Sequential(*[MultiStreamTransformerBlock(channels[1], num_heads[0], expansion_factor) for _ in range(num_blocks[0])])
        ])

        self.refinement = nn.Sequential(
            *[MultiStreamTransformerBlock(channels[1], num_heads[0], expansion_factor, res_scale) for _ in range(int(num_refinement))]
        )

        self.output = nn.Conv2d(channels[1], 3, 3, padding=1)

    def forward(self, x):
        fea = self.embed(x)
        out_enc1 = self.encoders[0](fea)
        out_enc2 = self.encoders[1](self.downs[0](out_enc1))
        out_enc3 = self.encoders[2](self.downs[1](out_enc2))
        out_enc4 = self.encoders[3](self.downs[2](out_enc3))
        out_dec3 = self.decoders[0](self.cross_scale_attn[0](out_enc3, self.ups[0](out_enc4)))
        out_dec2 = self.decoders[1](self.cross_scale_attn[1](out_enc2, self.ups[1](out_dec3)))
        fd = self.decoders[2](self.cross_scale_attn_bottleneck(out_enc1, self.ups[2](out_dec2)))
        fr = self.refinement(fd)
        out = self.output(fr) + x
        return out
