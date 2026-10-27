"""
SpikeVision-S3: Binary Cat vs Non-Cat CIFAR-10 Dataset Loader and Spike Encoders.

Este módulo prepara o dataset CIFAR-10 balanceado para a tarefa de classificação
binária ("Gato" vs "Não-Gato"), fornecendo também funções de codificação de spikes
(Rate Coding e Direct Coding) para uso em redes neurais pulsadas (SNN).
"""

import os
import math
import logging
from typing import Tuple, Optional, Callable, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import snntorch.spikegen as spikegen

logger = logging.getLogger(__name__)

# Índice da classe 'cat' no CIFAR-10
CIFAR10_CAT_LABEL = 3

# Nomes das 10 classes do CIFAR-10
CIFAR10_CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


class BinaryCatCIFAR10(Dataset):
    """
    Dataset balanceado 'Gato' (1) vs 'Não-Gato' (0) baseado no CIFAR-10.

    - Treino: 5.000 gatos + 5.000 não-gatos (total: 10.000 amostras balanceadas).
    - Teste: 1.000 gatos + 1.000 não-gatos (total: 2.000 amostras balanceadas).
    - As amostras da classe 0 (não-gato) são amostradas de forma estratificada/uniforme
      a partir das outras 9 classes do CIFAR-10 com seed fixa para reprodutibilidade.
    - Valores de pixel mantidos no intervalo [0, 1] para consistência com Rate Coding.
    """

    def __init__(
        self,
        root: str = "./training/data",
        train: bool = True,
        transform: Optional[Callable] = None,
        in_channels: int = 3,
        download: bool = True,
        seed: int = 42,
        smoke_test: bool = False,
        smoke_samples: int = 64,
    ):
        super().__init__()
        self.root = root
        self.train = train
        self.in_channels = in_channels
        self.seed = seed
        self.smoke_test = smoke_test

        # Definição das transformações caso não sejam informadas
        if transform is not None:
            self.transform = transform
        else:
            transform_list = []
            if train:
                transform_list.extend([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                ])
            if in_channels == 1:
                transform_list.append(transforms.Grayscale(num_output_channels=1))
            transform_list.append(transforms.ToTensor())  # Escala para [0.0, 1.0]
            self.transform = transforms.Compose(transform_list)

        # Carregar CIFAR-10 original (com fallback gracioso sintético se offline)
        try:
            self.cifar10 = torchvision.datasets.CIFAR10(
                root=self.root,
                train=self.train,
                download=download,
                transform=None,  # transform aplicado pelo __getitem__
            )
            self.is_synthetic = False
        except Exception as e:
            logger.warning(
                f"Não foi possível carregar/baixar o CIFAR-10 ({e}). "
                "Criando dataset sintético para execução offline/smoke test."
            )
            self.is_synthetic = True
            self._create_synthetic_data()
            return

        # Filtragem e balanceamento dos índices
        self.indices, self.binary_targets = self._balance_dataset()

        # Se smoke test ativado, selecionar um subconjunto balanceado
        if self.smoke_test:
            self._apply_smoke_subset(smoke_samples)

    def _create_synthetic_data(self):
        """Cria dados sintéticos balanceados caso o download do CIFAR-10 falhe."""
        n_samples = 64 if self.smoke_test else (200 if not self.train else 400)
        rng = np.random.RandomState(self.seed)
        # Imagens aleatórias em uint8 [0, 255] no formato PIL/HWC
        self.synthetic_images = rng.randint(0, 256, size=(n_samples, 32, 32, 3), dtype=np.uint8)
        self.binary_targets = np.array([1 if i % 2 == 0 else 0 for i in range(n_samples)], dtype=np.int64)
        self.indices = list(range(n_samples))

    def _balance_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filtra gatos (label 3 -> 1) e amostra igualmente das outras 9 classes (label 0).
        Retorna (índices_selecionados, rótulos_binários).
        """
        targets = np.array(self.cifar10.targets)
        cat_indices = np.where(targets == CIFAR10_CAT_LABEL)[0]
        num_cats = len(cat_indices)

        rng = np.random.RandomState(self.seed)

        # Coletar amostras das outras 9 classes balanceadamente
        other_classes = [c for c in range(10) if c != CIFAR10_CAT_LABEL]
        samples_per_class = num_cats // len(other_classes)
        remainder = num_cats % len(other_classes)

        non_cat_indices = []
        for i, c in enumerate(other_classes):
            c_indices = np.where(targets == c)[0]
            take_count = samples_per_class + (1 if i < remainder else 0)
            chosen = rng.choice(c_indices, size=take_count, replace=False)
            non_cat_indices.extend(chosen)

        non_cat_indices = np.array(non_cat_indices)

        # Unir e embaralhar os índices com a seed fixa
        all_indices = np.concatenate([cat_indices, non_cat_indices])
        all_targets = np.concatenate([
            np.ones(len(cat_indices), dtype=np.int64),
            np.zeros(len(non_cat_indices), dtype=np.int64),
        ])

        shuffle_perm = rng.permutation(len(all_indices))
        return all_indices[shuffle_perm], all_targets[shuffle_perm]

    def _apply_smoke_subset(self, max_samples: int):
        """Reduz o dataset a uma fração balanceada para testes rápidos."""
        cats = [i for i, t in enumerate(self.binary_targets) if t == 1]
        non_cats = [i for i, t in enumerate(self.binary_targets) if t == 0]
        half = max_samples // 2
        selected = cats[:half] + non_cats[:half]
        self.indices = self.indices[selected]
        self.binary_targets = self.binary_targets[selected]

    def __len__(self) -> int:
        return len(self.binary_targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if getattr(self, "is_synthetic", False):
            img_arr = self.synthetic_images[idx]
            from PIL import Image
            img = Image.fromarray(img_arr)
            target = int(self.binary_targets[idx])
        else:
            orig_idx = self.indices[idx]
            img, _ = self.cifar10[orig_idx]
            target = int(self.binary_targets[idx])

        if self.transform is not None:
            img = self.transform(img)

        return img, target


def encode_spikes(
    images: torch.Tensor,
    num_steps: int = 8,
    method: str = "rate",
) -> torch.Tensor:
    """
    Converte um tensor de imagens [B, C, H, W] com valores em [0, 1] em trens de spikes
    ao longo da dimensão temporal [T, B, C, H, W].

    Args:
        images: Tensor de entrada [B, C, H, W], valores em [0, 1].
        num_steps: Número de timesteps (T) da simulação da SNN.
        method: 'rate' para Poisson Rate Coding (snntorch.spikegen.rate)
                'direct' para Direct Coding (repetição da imagem em T passos).

    Returns:
        Tensor de pulsos de shape [T, B, C, H, W].
    """
    if method == "rate":
        # snntorch.spikegen.rate aceita tensores normalizados em [0, 1]
        return spikegen.rate(images, num_steps=num_steps)
    elif method == "direct":
        # Repete a imagem estática para cada passo temporal.
        # A primeira camada convolucional da SNN atua como codificador direto.
        return images.unsqueeze(0).repeat(num_steps, 1, 1, 1, 1)
    else:
        raise ValueError(
            f"Método de codificação '{method}' não suportado. Escolha 'rate' ou 'direct'."
        )


def get_dataloaders(
    data_dir: str = "./training/data",
    batch_size: int = 64,
    num_workers: int = 2,
    in_channels: int = 3,
    download: bool = True,
    seed: int = 42,
    smoke_test: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Cria e retorna os DataLoaders de treino e teste balanceados para a SNN.

    Args:
        data_dir: Diretório onde o CIFAR-10 será baixado/armazenado.
        batch_size: Tamanho do lote.
        num_workers: Número de subprocessos para carregamento.
        in_channels: 3 para RGB ou 1 para escala de cinza.
        download: Se deve baixar o CIFAR-10 caso não exista.
        seed: Seed para amostragem reprodutível.
        smoke_test: Se True, carrega um subconjunto mínimo para testes rápidos.

    Returns:
        Tupla (train_loader, test_loader).
    """
    os.makedirs(data_dir, exist_ok=True)

    train_dataset = BinaryCatCIFAR10(
        root=data_dir,
        train=True,
        in_channels=in_channels,
        download=download,
        seed=seed,
        smoke_test=smoke_test,
        smoke_samples=64,
    )

    test_dataset = BinaryCatCIFAR10(
        root=data_dir,
        train=False,
        in_channels=in_channels,
        download=download,
        seed=seed,
        smoke_test=smoke_test,
        smoke_samples=32,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return train_loader, test_loader


# Alias para retrocompatibilidade
get_cifar10_cat_dataloaders = get_dataloaders


def get_cifar10_multiclass_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 64,
    num_workers: int = 2,
    augment: bool = True,
    download: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Carrega o dataset CIFAR-10 completo com as 10 classes (50.000 treino, 10.000 teste).

    Args:
        data_dir: Diretório onde o CIFAR-10 será baixado/armazenado.
        batch_size: Tamanho do lote.
        num_workers: Número de workers no DataLoader.
        augment: Se True, aplica RandomCrop(32, pad=4, reflect) e RandomHorizontalFlip.
        download: Se deve baixar o CIFAR-10 caso não exista.

    Returns:
        Tupla (train_loader, test_loader).
    """
    os.makedirs(data_dir, exist_ok=True)

    train_transforms_list = []
    if augment:
        train_transforms_list.extend([
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
        ])
    # Mantém valores no intervalo [0.0, 1.0] para compatibilidade com encode_spikes
    train_transforms_list.append(transforms.ToTensor())
    train_transform = transforms.Compose(train_transforms_list)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=download,
        transform=train_transform,
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=download,
        transform=test_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return train_loader, test_loader
