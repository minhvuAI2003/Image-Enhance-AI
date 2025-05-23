import torch
import torch.nn as nn
import torch.nn.functional as F

# Simple Embedding (No Overlap Patch)


# GDFN with H-Swish
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

# MDTA
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

class TransformerBlock(nn.Module):
         def __init__(self, channels, num_heads, expansion_factor=2.66, res_scale=0.1, streams=2):
             super(TransformerBlock, self).__init__()
             
             self.streams = nn.ModuleList([
                 MDTA(channels, num_heads) for _ in range(streams)
             ])
             
             self.stream_weights = nn.Parameter(torch.ones(streams))  # learnable weights
     
             self.norm1 = nn.LayerNorm(channels)
             self.norm2 = nn.LayerNorm(channels)
             self.ffn = GDFN(channels, expansion_factor)
             self.res_scale = res_scale
 
         def forward(self, x):
             b, c, h, w = x.shape
     
             # LayerNorm trước Attention
             x_flat = x.permute(0, 2, 3, 1)  # (B, H, W, C)
             x_norm1 = self.norm1(x_flat).permute(0, 3, 1, 2)  # (B, C, H, W)
     
             outs = []
             for stream in self.streams:
                 out = stream(x_norm1)
                 outs.append(out)
     
             outs = torch.stack(outs, dim=0)  # (streams, B, C, H, W)
     
             # Normalize stream weights
             weights = F.softmax(self.stream_weights, dim=0)  # (streams,)
     
             # Weighted sum các stream outputs
             attn_out = (weights.view(-1, 1, 1, 1, 1) * outs).sum(dim=0)
     
             # Residual Add sau Attention
             x = x + attn_out * self.res_scale
     
             # LayerNorm trước FFN
             identity = x
             x_flat = x.permute(0, 2, 3, 1)  # (B, H, W, C)
             x_norm2 = self.norm2(x_flat).permute(0, 3, 1, 2)  # (B, C, H, W)
     
             out = self.ffn(x_norm2)
     
             # Residual Add sau FFN
             x = identity + out * self.res_scale
     
             return x


# Cross-Scale Attention Fusion


# DownSample and UpSample
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
        self.body = nn.Sequential(nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

# Full Model
class Restormer(nn.Module):
    def __init__(self, num_blocks=[4, 6, 6, 8], num_heads=[1, 2, 4, 8], channels=[48, 96, 192, 384], num_refinement=4,
                 expansion_factor=2.66):
        super(Restormer, self).__init__()

        self.embed_conv = nn.Conv2d(3, channels[0], kernel_size=3, padding=1, bias=False)

        self.encoders = nn.ModuleList([nn.Sequential(*[TransformerBlock(
            num_ch, num_ah, expansion_factor) for _ in range(num_tb)]) for num_tb, num_ah, num_ch in
                                       zip(num_blocks, num_heads, channels)])
        # the number of down sample or up sample == the number of encoder - 1
        self.downs = nn.ModuleList([DownSample(num_ch) for num_ch in channels[:-1]])
        self.ups = nn.ModuleList([UpSample(num_ch) for num_ch in list(reversed(channels))[:-1]])
        # the number of reduce block == the number of decoder - 1
        self.reduces = nn.ModuleList([nn.Conv2d(channels[i], channels[i - 1], kernel_size=1, bias=False)
                                      for i in reversed(range(2, len(channels)))])
        # the number of decoder == the number of encoder - 1
        self.decoders = nn.ModuleList([nn.Sequential(*[TransformerBlock(channels[2], num_heads[2], expansion_factor)
                                                       for _ in range(num_blocks[2])])])
        self.decoders.append(nn.Sequential(*[TransformerBlock(channels[1], num_heads[1], expansion_factor)
                                             for _ in range(num_blocks[1])]))
        # the channel of last one is not change
        self.decoders.append(nn.Sequential(*[TransformerBlock(channels[1], num_heads[0], expansion_factor)
                                             for _ in range(num_blocks[0])]))

        self.refinement = nn.Sequential(*[TransformerBlock(channels[1], num_heads[0], expansion_factor)
                                          for _ in range(num_refinement)])
        self.output = nn.Conv2d(channels[1], 3, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        fo = self.embed_conv(x)
        out_enc1 = self.encoders[0](fo)
        out_enc2 = self.encoders[1](self.downs[0](out_enc1))
        out_enc3 = self.encoders[2](self.downs[1](out_enc2))
        out_enc4 = self.encoders[3](self.downs[2](out_enc3))

        out_dec3 = self.decoders[0](self.reduces[0](torch.cat([self.ups[0](out_enc4), out_enc3], dim=1)))
        out_dec2 = self.decoders[1](self.reduces[1](torch.cat([self.ups[1](out_dec3), out_enc2], dim=1)))
        fd = self.decoders[2](torch.cat([self.ups[2](out_dec2), out_enc1], dim=1))
        fr = self.refinement(fd)
        out = self.output(fr) + x
        return out
