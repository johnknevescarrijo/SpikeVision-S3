"""
Módulo de datasets e codificadores de spikes para SpikeVision-S3.
"""

from .cifar10_cat import (
    BinaryCatCIFAR10,
    encode_spikes,
    get_dataloaders,
    get_cifar10_cat_dataloaders,
    get_cifar10_multiclass_dataloaders,
    CIFAR10_CAT_LABEL,
    CIFAR10_CLASS_NAMES,
)

__all__ = [
    "BinaryCatCIFAR10",
    "encode_spikes",
    "get_dataloaders",
    "get_cifar10_cat_dataloaders",
    "get_cifar10_multiclass_dataloaders",
    "CIFAR10_CAT_LABEL",
    "CIFAR10_CLASS_NAMES",
]
