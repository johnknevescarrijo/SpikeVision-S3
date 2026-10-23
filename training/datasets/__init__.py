"""
Módulo de datasets e codificadores de spikes para SpikeVision-S3.
"""

from .cifar10_cat import (
    BinaryCatCIFAR10,
    encode_spikes,
    get_dataloaders,
    CIFAR10_CAT_LABEL,
)

__all__ = [
    "BinaryCatCIFAR10",
    "encode_spikes",
    "get_dataloaders",
    "CIFAR10_CAT_LABEL",
]
