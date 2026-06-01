"""
v2 wrapper models to experiment with alternative UNet shapes/attention
without modifying the original training codepaths.

Key additions vs `model_cifar10_sde_DDPM.py`:
- configurable `attn_resolutions` passed into `Unet`
- intended defaults: dim_mults=(1,2,2,2) and attn_resolutions=(16,8)
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

from unet_DDPM import Unet
from model_cifar10_sde_DDPM import (
    CIFAR10Diffusion_SDE as _CIFAR10Diffusion_SDE_Base,
    CIFAR10_Active_Diffusion_SDE as _CIFAR10_Active_Diffusion_SDE_Base,
)


def _as_tuple_ints(values: Iterable[int]) -> Tuple[int, ...]:
    return tuple(int(v) for v in values)


class CIFAR10Diffusion_SDE_V2(_CIFAR10Diffusion_SDE_Base):
    """
    Same behavior as CIFAR10Diffusion_SDE, but allows configuring attention resolutions.
    """

    def __init__(
        self,
        image_size: int = 32,
        in_channels: int = 3,
        time_embedding_dim: int = 256,
        timesteps: int = 1000,
        base_dim: int = 128,
        dim_mults: Sequence[int] = (1, 2, 2, 2),
        attn_resolutions: Sequence[int] = (16, 8),
        num_res_blocks: int = 2,
        dropout: float = 0.1,
        T: float = 2.0,
        k: float = 1.0,
        Tp: float = 1.0,
    ):
        super().__init__(
            image_size=image_size,
            in_channels=in_channels,
            time_embedding_dim=time_embedding_dim,
            timesteps=timesteps,
            base_dim=base_dim,
            dim_mults=list(dim_mults),
            num_res_blocks=num_res_blocks,
            T=T,
            k=k,
            Tp=Tp,
        )

        # Replace the UNet with a configurable-attention version.
        self.model = Unet(
            timesteps,
            time_embedding_dim,
            in_channels,
            in_channels,
            base_dim,
            list(dim_mults),
            num_res_blocks=num_res_blocks,
            attn_resolutions=_as_tuple_ints(attn_resolutions),
            dropout=dropout,
            image_size=image_size,
        )


class CIFAR10_Active_Diffusion_SDE_V2(_CIFAR10_Active_Diffusion_SDE_Base):
    """
    Same behavior as CIFAR10_Active_Diffusion_SDE, but allows configuring attention resolutions.
    """

    def __init__(
        self,
        image_size: int = 32,
        time_embedding_dim: int = 256,
        timesteps: int = 1000,
        base_dim: int = 128,
        dim_mults: Sequence[int] = (1, 2, 2, 2),
        attn_resolutions: Sequence[int] = (16, 8),
        num_res_blocks: int = 2,
        Tp: float = 1e-3,
        Ta: float = 1.0,
        k: float = 1.0,
        tau: float = 0.1,
        T: float = 2.0,
        dropout: float = 0.1,
    ):
        super().__init__(
            image_size=image_size,
            time_embedding_dim=time_embedding_dim,
            timesteps=timesteps,
            base_dim=base_dim,
            dim_mults=list(dim_mults),
            num_res_blocks=num_res_blocks,
            Tp=Tp,
            Ta=Ta,
            k=k,
            tau=tau,
            T=T,
        )

        # Replace the UNet with a configurable-attention version.
        self.model = Unet(
            timesteps,
            time_embedding_dim,
            6,
            6,
            base_dim,
            list(dim_mults),
            num_res_blocks=num_res_blocks,
            attn_resolutions=_as_tuple_ints(attn_resolutions),
            dropout=dropout,
            image_size=image_size,
        )


