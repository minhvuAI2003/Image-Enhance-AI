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
    def __init__(self, channels, num_heads, expansion_factor):
        super(TransformerBlock, self).__init__()

        self.norm1 = nn.LayerNorm(channels)
        self.attn = MDTA(channels, num_heads)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = GDFN(channels, expansion_factor)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x + self.attn(self.norm1(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                          .contiguous().reshape(b, c, h, w))
        x = x + self.ffn(self.norm2(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                         .contiguous().reshape(b, c, h, w))
        return x


# Cross-Scale Attention Fusion
class CrossScaleAttentionFusion(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

    def forward(self, q_feat, kv_feat):
        B, C, H, W = q_feat.shape
        q = self.q_proj(self.norm_q(q_feat.flatten(2).transpose(1, 2)))  # [B, HW, C]
        k = self.k_proj(self.norm_kv(kv_feat.flatten(2).transpose(1, 2)))
        v = self.v_proj(self.norm_kv(kv_feat.flatten(2).transpose(1, 2)))

        q = q.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, -1, C)
        out = self.out_proj(out).transpose(1, 2).reshape(B, C, H, W)
        return out
class CrossScaleAttentionFusion_1(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim*2)
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

    def forward(self, q_feat, kv_feat):
        B, C, H, W = q_feat.shape
        q = self.q_proj(self.norm_q(q_feat.flatten(2).transpose(1, 2)))  # [B, HW, C]
        k = self.k_proj(self.norm_kv(kv_feat.flatten(2).transpose(1, 2)))
        v = self.v_proj(self.norm_kv(kv_feat.flatten(2).transpose(1, 2)))

        q = q.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, -1, C)
        out = self.out_proj(out).transpose(1, 2).reshape(B, C, H, W)
        return out

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
    def __init__(self, num_blocks=[4,6,6,8], num_heads=[1,2,4,8], channels=[48,96,192,384], expansion_factor=2.66, num_refinement=4, res_scale=0.1):
        super(Restormer, self).__init__()
        self.embed = nn.Conv2d(3, channels[0], kernel_size=3, padding=1, bias=False)

        self.encoders = nn.ModuleList()
        for i, (num_tb, num_ah, num_ch) in enumerate(zip(num_blocks, num_heads, channels)):
            if i == 0:
        # First encoder level: normal linear path
                self.encoders.append(nn.Sequential(*[TransformerBlock(num_ch, num_ah, expansion_factor) for _ in range(num_tb)]))
            else:
        # Multi-stream: create 2 streams, 2 blocks each
                stream_blocks = nn.ModuleList([
            nn.Sequential(*[TransformerBlock(num_ch, num_ah, expansion_factor) for _ in range(num_tb // 2)]),
            nn.Sequential(*[TransformerBlock(num_ch, num_ah, expansion_factor) for _ in range(num_tb // 2)])
        ])
                self.encoders.append(stream_blocks)


        self.downs = nn.ModuleList([DownSample(c) for c in channels[:-1]])

        self.ups = nn.ModuleList([UpSample(num_ch) for num_ch in list(reversed(channels))[:-1]])

        self.reduces = nn.ModuleList([nn.Conv2d(channels[i], channels[i - 1], kernel_size=1, bias=False)
                                      for i in reversed(range(2, len(channels)))])
        self.cross_scale_attn = nn.ModuleList([
            CrossScaleAttentionFusion(channels[i]) for i in reversed(range(1, len(channels)-1))
        ])
        self.cross_scale_attn_bottleneck = CrossScaleAttentionFusion_1(channels[0])

        self.decoders = nn.ModuleList([nn.Sequential(*[TransformerBlock(channels[2], num_heads[2], expansion_factor)
                                                       for _ in range(num_blocks[2])])])
        self.decoders.append(nn.Sequential(*[TransformerBlock(channels[1], num_heads[1], expansion_factor)
                                             for _ in range(num_blocks[1])]))
        # the channel of last one is not change
        self.decoders.append(nn.Sequential(*[TransformerBlock(channels[1], num_heads[0], expansion_factor)
                                             for _ in range(num_blocks[0])]))

        self.refinement = nn.Sequential(
    *[TransformerBlock(channels[1], num_heads[0], expansion_factor) for _ in range(int(num_refinement))]
)


        self.output = nn.Conv2d(channels[1], 3, 3, padding=1)
        self.fuse_proj = nn.ModuleList([
    nn.Conv2d(ch, ch, kernel_size=1, bias=False) for ch in channels[1:]
])


    def forward(self, x):
        fea = self.embed(x)

        out_enc1 = self.encoders[0](fea)  # Linear first block

        # Second level with two streams
        in2 = self.downs[0](out_enc1)
        s1_out2 = self.encoders[1][0](in2)
        s2_out2 = self.encoders[1][1](in2)

        out_enc2 = self.fuse_proj[0](s1_out2 + s2_out2 + in2) / 3  # Multi-stream + skip
        
        # Third level with two streams
        in3 = self.downs[1](out_enc2)
        s1_out3 = self.encoders[2][0](in3)
        s2_out3 = self.encoders[2][1](in3)
        out_enc3 = self.fuse_proj[1](s1_out3 + s2_out3 + in3) / 3
        
        # Fourth level with two streams
        in4 = self.downs[2](out_enc3)
        s1_out4 = self.encoders[3][0](in4)
        s2_out4 = self.encoders[3][1](in4)
        out_enc4 = self.fuse_proj[2](s1_out4 + s2_out4 + in4) / 3


        out_dec3 = self.decoders[0](self.cross_scale_attn[0](out_enc3,self.ups[0](out_enc4)))
        # print(out_dec3.shape)
        out_dec2 = self.decoders[1](self.cross_scale_attn[1](out_enc2,self.ups[1](out_dec3)))
        # print(out_dec2.shape)
        fd = self.decoders[2](self.cross_scale_attn_bottleneck(out_enc1,self.ups[2](out_dec2)))

        fr = self.refinement(fd)
        out = self.output(fr) + x
        return out
