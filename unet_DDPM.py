import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_norm_channels(channels: int) -> int:
    return min(32, channels)


class SinusoidalTimeEmbedding(nn.Module):
    """
    Classic sinusoidal (Fourier) embedding used by DDPM / NCSN++.
    Accepts normalized timesteps in [0, 1] and rescales by the configured max step.
    """

    def __init__(self, dim: int, max_steps: int = 1000):
        super().__init__()
        self.dim = dim
        self.max_steps = max_steps

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        exponent = -math.log(10000) * torch.arange(half_dim, device=device) / float(half_dim)
        scaled_time = t * self.max_steps
        emb = torch.exp(exponent)[None, :] * scaled_time[:, None]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ResidualBlock(nn.Module):
    """
    DDPM-style residual block with time embedding injection.
    """

    def __init__(self, in_channels, out_channels, time_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_norm_channels(in_channels), in_channels)
        self.norm2 = nn.GroupNorm(_group_norm_channels(out_channels), out_channels)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.dropout = nn.Dropout(dropout)
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(self.act(self.norm1(x)))
        time_term = self.time_proj(t_emb)[:, :, None, None]
        h = h + time_term
        h = self.conv2(self.dropout(self.act(self.norm2(h))))
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """
    Self-attention block operating on HxW feature maps.
    """

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(_group_norm_channels(channels), channels)
        self.q = nn.Conv2d(channels, channels, kernel_size=1)
        self.k = nn.Conv2d(channels, channels, kernel_size=1)
        self.v = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        x_norm = self.norm(x)
        q = self.q(x_norm).reshape(b, c, h * w).permute(0, 2, 1)    # (b, hw, c)
        k = self.k(x_norm).reshape(b, c, h * w)                     # (b, c, hw)
        attn = torch.softmax(torch.bmm(q, k) / math.sqrt(c), dim=-1)
        v = self.v(x_norm).reshape(b, c, h * w).permute(0, 2, 1)    # (b, hw, c)
        h_out = torch.bmm(attn, v).permute(0, 2, 1).reshape(b, c, h, w)
        h_out = self.proj(h_out)
        return x + h_out


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class Unet(nn.Module):
    """
    DDPM / NCSN++ style UNet with attention and multiple residual blocks per resolution.
    """

    def __init__(
        self,
        timesteps,
        time_embedding_dim,
        in_channels=3,
        out_channels=2,
        base_dim=128,
        dim_mults=(1, 2, 2, 2),
        num_res_blocks=2,
        attn_resolutions=(16,),
        dropout=0.0,
        image_size=16,
    ):
        super().__init__()
        assert isinstance(dim_mults, (list, tuple))
        self.image_size = image_size
        self.attn_resolutions = set(attn_resolutions)
        self.num_resolutions = len(dim_mults)
        self.num_res_blocks = num_res_blocks

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_embedding_dim, max_steps=timesteps),
            nn.Linear(time_embedding_dim, time_embedding_dim * 4),
            nn.SiLU(),
            nn.Linear(time_embedding_dim * 4, time_embedding_dim * 4),
        )
        self.time_dim = time_embedding_dim * 4

        # Initial projection
        self.init_conv = nn.Conv2d(in_channels, base_dim, kernel_size=3, padding=1)

        # Downsampling path
        self.downs = nn.ModuleList()
        ch = base_dim
        current_res = image_size
        skip_channels = []
        for level, mult in enumerate(dim_mults):
            out_ch = base_dim * mult
            blocks = nn.ModuleList()
            attns = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResidualBlock(ch, out_ch, self.time_dim, dropout))
                attns.append(AttentionBlock(out_ch) if current_res in self.attn_resolutions else None)
                ch = out_ch
                skip_channels.append(ch)
            downsample = Downsample(ch) if level != len(dim_mults) - 1 else None
            if downsample is not None:
                current_res //= 2
            self.downs.append(nn.ModuleDict({"blocks": blocks, "attns": attns, "downsample": downsample}))

        # Bottleneck
        self.mid_block1 = ResidualBlock(ch, ch, self.time_dim, dropout)
        self.mid_attn = AttentionBlock(ch)
        self.mid_block2 = ResidualBlock(ch, ch, self.time_dim, dropout)

        # Upsampling path
        self.ups = nn.ModuleList()
        current_res = image_size // (2 ** (len(dim_mults) - 1))
        skip_stack = skip_channels.copy()
        for level, mult in reversed(list(enumerate(dim_mults))):
            out_ch = base_dim * mult
            blocks = nn.ModuleList()
            attns = nn.ModuleList()
            for _ in range(num_res_blocks):
                skip_ch = skip_stack.pop()
                blocks.append(ResidualBlock(ch + skip_ch, out_ch, self.time_dim, dropout))
                attns.append(AttentionBlock(out_ch) if current_res in self.attn_resolutions else None)
                ch = out_ch
            upsample = Upsample(ch) if level != 0 else None
            if upsample is not None:
                current_res *= 2
            self.ups.append(nn.ModuleDict({"blocks": blocks, "attns": attns, "upsample": upsample}))

        self.final_norm = nn.GroupNorm(_group_norm_channels(ch), ch)
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv2d(ch, out_channels, kernel_size=3, padding=1)

    def forward(self, x, t=None):
        if t is None:
            raise ValueError("Time tensor t must be provided.")
        t_emb = self.time_embed(t)
        hs = []

        x = self.init_conv(x)
        for stage in self.downs:
            blocks = stage["blocks"]
            attns = stage["attns"]
            for block, attn in zip(blocks, attns):
                x = block(x, t_emb)
                if attn is not None:
                    x = attn(x)
                hs.append(x)
            if stage["downsample"] is not None:
                x = stage["downsample"](x)

        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_emb)

        for stage in self.ups:
            blocks = stage["blocks"]
            attns = stage["attns"]
            for block, attn in zip(blocks, attns):
                skip = hs.pop()
                x = torch.cat([x, skip], dim=1)
                x = block(x, t_emb)
                if attn is not None:
                    x = attn(x)
            if stage["upsample"] is not None:
                x = stage["upsample"](x)

        x = self.final_conv(self.final_act(self.final_norm(x)))
        return x


if __name__ == "__main__":
    model = Unet(timesteps=1000, time_embedding_dim=256, in_channels=3, out_channels=3)
    dummy_x = torch.randn(2, 3, 32, 32)
    dummy_t = torch.rand(2)
    y = model(dummy_x, dummy_t)
    print(y.shape)
