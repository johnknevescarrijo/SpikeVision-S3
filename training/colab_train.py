#!/usr/bin/env python3
"""
================================================================================
SpikeVision-S3: Treinamento CSNN no Google Colab (Script 100% Autocontido)
================================================================================
Instruções de Uso no Google Colab:
1. Abra um novo notebook no Google Colab (https://colab.research.google.com).
2. Vá no menu: 'Ambiente de execução' > 'Alterar tipo de ambiente de execução'
   e selecione 'T4 GPU' (Hardware accelerator: GPU).
3. Copie todo o conteúdo deste arquivo e cole em uma única célula de código.
4. Execute a célula (Shift + Enter).
5. O script irá:
   - Verificar e instalar automaticamente a biblioteca `snntorch`;
   - Baixar e balancear o CIFAR-10 (Gato vs Não-Gato, 10.000 treino, 2.000 teste);
   - Treinar a CSNN (SpikeVisionCSNN) com 15.400 parâmetros (~15.4 KB Int8);
   - Monitorar Acurácia, F1-Score da classe Gato, Firing Rate e Esparsidade;
   - Salvar o melhor checkpoint 'best_spikevision_csnn.pth';
   - Acionar o download automático do modelo treinado para o seu computador!
================================================================================
"""

import sys
import subprocess

# 1. Instalação automática de dependências caso ausentes no ambiente Colab
try:
    import snntorch
except ImportError:
    print("⚡ Instalando snnTorch para o ambiente Colab...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "snntorch"])
    import snntorch

import os
import time
import json
import random
from typing import Tuple, Optional, Callable, Dict, Any, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms

import snntorch as snn
from snntorch import surrogate
import snntorch.spikegen as spikegen
import snntorch.functional as SF
from tqdm import tqdm


# ==============================================================================
# CONFIGURAÇÃO DE HIPERPARÂMETROS (Fácil de editar diretamente)
# ==============================================================================
CONFIG = {
    # Treinamento
    "epochs": 25,
    "batch_size": 128,
    "lr": 1e-3,
    "weight_decay": 1e-5,

    # Arquitetura SNN & Simulação Temporal
    "num_steps": 8,          # Timesteps T da simulação SNN
    "beta": 0.9,             # Fator de decaimento do neurônio Leaky LIF
    "slope": 25.0,           # Inclinação do gradiente substituto FastSigmoid
    "encoding": "direct",    # 'direct' (recomendado, alta acurácia) ou 'rate'
    "in_channels": 3,        # 3 para RGB (ou 1 para escala de cinza)
    "num_classes": 2,        # Binário: 0 (Não-Gato) vs 1 (Gato)

    # Sistema e Reprodutibilidade
    "seed": 42,
    "num_workers": 2,
    "data_dir": "./data",
    "checkpoint_path": "best_spikevision_csnn.pth",
    "history_path": "training_history.json",
}


# ==============================================================================
# REPRODUTIBILIDADE
# ==============================================================================
def set_seed(seed: int = 42):
    """Garante reprodutibilidade em todas as bibliotecas e inicializações."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ==============================================================================
# DATASET: CIFAR-10 BINÁRIO BALANCEADO (GATO vs NÃO-GATO)
# ==============================================================================
CIFAR10_CAT_LABEL = 3


class BinaryCatCIFAR10(Dataset):
    """
    Dataset balanceado 'Gato' (1) vs 'Não-Gato' (0) baseado no CIFAR-10.
    - Treino: 5.000 gatos + 5.000 não-gatos (total: 10.000 amostras balanceadas).
    - Teste: 1.000 gatos + 1.000 não-gatos (total: 2.000 amostras balanceadas).
    - As amostras não-gato são amostradas de maneira uniforme das outras 9 classes.
    """

    def __init__(
        self,
        root: str = "./data",
        train: bool = True,
        transform: Optional[Callable] = None,
        in_channels: int = 3,
        download: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        self.root = root
        self.train = train
        self.in_channels = in_channels
        self.seed = seed

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

        try:
            self.cifar10 = torchvision.datasets.CIFAR10(
                root=self.root,
                train=self.train,
                download=download,
                transform=None,
            )
            self.is_synthetic = False
            self.indices, self.binary_targets = self._balance_dataset()
        except Exception as e:
            print(f"⚠️ Não foi possível baixar CIFAR-10 ({e}). Gerando dados sintéticos para fallback.")
            self.is_synthetic = True
            self._create_synthetic_data()

    def _create_synthetic_data(self):
        """Cria dados sintéticos balanceados caso o download do CIFAR-10 falhe."""
        n_samples = 200 if not self.train else 400
        rng = np.random.RandomState(self.seed)
        self.synthetic_images = rng.randint(0, 256, size=(n_samples, 32, 32, 3), dtype=np.uint8)
        self.binary_targets = np.array([1 if i % 2 == 0 else 0 for i in range(n_samples)], dtype=np.int64)
        self.indices = list(range(n_samples))

    def _balance_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Filtra gatos (label 3 -> 1) e amostra igualmente das outras 9 classes (label 0)."""
        targets = np.array(self.cifar10.targets)
        cat_indices = np.where(targets == CIFAR10_CAT_LABEL)[0]
        num_cats = len(cat_indices)

        rng = np.random.RandomState(self.seed)

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

        all_indices = np.concatenate([cat_indices, non_cat_indices])
        all_targets = np.concatenate([
            np.ones(len(cat_indices), dtype=np.int64),
            np.zeros(len(non_cat_indices), dtype=np.int64),
        ])

        shuffle_perm = rng.permutation(len(all_indices))
        return all_indices[shuffle_perm], all_targets[shuffle_perm]

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
    method: str = "direct",
) -> torch.Tensor:
    """
    Converte um tensor de imagens [B, C, H, W] em trens de spikes ao longo do tempo [T, B, C, H, W].
    """
    if method == "rate":
        return spikegen.rate(images, num_steps=num_steps)
    elif method == "direct":
        return images.unsqueeze(0).repeat(num_steps, 1, 1, 1, 1)
    else:
        raise ValueError(f"Método '{method}' não suportado. Escolha 'rate' ou 'direct'.")


def get_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 2,
    in_channels: int = 3,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Cria os DataLoaders balanceados para treino e teste."""
    os.makedirs(data_dir, exist_ok=True)

    train_dataset = BinaryCatCIFAR10(
        root=data_dir,
        train=True,
        in_channels=in_channels,
        download=True,
        seed=seed,
    )

    test_dataset = BinaryCatCIFAR10(
        root=data_dir,
        train=False,
        in_channels=in_channels,
        download=True,
        seed=seed,
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


# ==============================================================================
# MODELO: SPIKEVISION CONVOLUTIONAL SNN (CSNN) PARA ESP32-S3
# ==============================================================================
class SpikeVisionCSNN(nn.Module):
    """
    Convolutional Spiking Neural Network (CSNN) para classificação binária (Gato vs Não-Gato).
    Arquitetura ultra-compacta (15.400 parâmetros / ~15.4 KB Int8) ideal para o ESP32-S3.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        beta: float = 0.9,
        slope: float = 25.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.beta = beta

        spike_grad = surrogate.fast_sigmoid(slope=slope)

        # Camada 1: 32x32 -> MaxPool -> 16x16
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=False)

        # Camada 2: 16x16 -> MaxPool -> 8x8
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=False)

        # Camada 3: 8x8 -> MaxPool -> 4x4
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool3 = nn.MaxPool2d(2)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=False)

        # Classificador (Head): 32 * 4 * 4 = 512 neurônios -> 2 classes
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 4 * 4, num_classes, bias=False)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=False)

    def init_membrane_potentials(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Inicializa potenciais de membrana U das 4 camadas com zeros."""
        mem1 = torch.zeros((batch_size, 16, 16, 16), device=device)
        mem2 = torch.zeros((batch_size, 32, 8, 8), device=device)
        mem3 = torch.zeros((batch_size, 32, 4, 4), device=device)
        mem4 = torch.zeros((batch_size, self.num_classes), device=device)
        return mem1, mem2, mem3, mem4

    def forward(
        self,
        x: torch.Tensor,
        num_steps: int = 8,
        encoding: str = "direct",
        return_sparsity: bool = True,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]]:
        """Propagação direta temporal na SpikeVisionCSNN."""
        if x.dim() == 4:
            spk_in = encode_spikes(x, num_steps=num_steps, method=encoding)
            t_steps = num_steps
            batch_size = x.size(0)
        elif x.dim() == 5:
            spk_in = x
            t_steps = x.size(0)
            batch_size = x.size(1)
        else:
            raise ValueError(f"Shape de entrada inválido: {x.shape}")

        device = x.device
        mem1, mem2, mem3, mem4 = self.init_membrane_potentials(batch_size, device)

        spk_out_rec = []
        mem_out_rec = []

        spk_count_l1 = 0
        spk_count_l2 = 0
        spk_count_l3 = 0
        spk_count_l4 = 0

        for t in range(t_steps):
            inp_t = spk_in[t]

            # Layer 1
            cur1 = self.pool1(self.bn1(self.conv1(inp_t)))
            spk1, mem1 = self.lif1(cur1, mem1)

            # Layer 2
            cur2 = self.pool2(self.bn2(self.conv2(spk1)))
            spk2, mem2 = self.lif2(cur2, mem2)

            # Layer 3
            cur3 = self.pool3(self.bn3(self.conv3(spk2)))
            spk3, mem3 = self.lif3(cur3, mem3)

            # Head (Linear)
            flat = self.flatten(spk3)
            cur4 = self.fc(flat)
            spk4, mem4 = self.lif4(cur4, mem4)

            spk_out_rec.append(spk4)
            mem_out_rec.append(mem4)

            if return_sparsity:
                spk_count_l1 += spk1.detach().sum().item()
                spk_count_l2 += spk2.detach().sum().item()
                spk_count_l3 += spk3.detach().sum().item()
                spk_count_l4 += spk4.detach().sum().item()

        spk_out = torch.stack(spk_out_rec, dim=0)
        mem_out = torch.stack(mem_out_rec, dim=0)

        if not return_sparsity:
            return spk_out, mem_out

        total_neurons_l1 = t_steps * batch_size * (16 * 16 * 16)
        total_neurons_l2 = t_steps * batch_size * (32 * 8 * 8)
        total_neurons_l3 = t_steps * batch_size * (32 * 4 * 4)
        total_neurons_l4 = t_steps * batch_size * self.num_classes
        total_neurons_all = total_neurons_l1 + total_neurons_l2 + total_neurons_l3 + total_neurons_l4
        total_spikes_all = spk_count_l1 + spk_count_l2 + spk_count_l3 + spk_count_l4

        overall_firing_rate = total_spikes_all / max(total_neurons_all, 1)

        sparsity_dict = {
            "firing_rate_layer1": spk_count_l1 / max(total_neurons_l1, 1),
            "firing_rate_layer2": spk_count_l2 / max(total_neurons_l2, 1),
            "firing_rate_layer3": spk_count_l3 / max(total_neurons_l3, 1),
            "firing_rate_layer4": spk_count_l4 / max(total_neurons_l4, 1),
            "overall_firing_rate": overall_firing_rate,
            "overall_sparsity": 1.0 - overall_firing_rate,
        }

        return spk_out, mem_out, sparsity_dict


def print_model_summary(model: nn.Module):
    """Exibe um sumário legível da arquitetura e ocupação estimada na SRAM."""
    total_params = 0
    print("=" * 72)
    print("                 SpikeVision-S3 CSNN Model Summary")
    print("=" * 72)
    print(f"{'Layer / Parameter':<35} {'Shape':<20} {'Params':<10} {'Int8 (B)':<8}")
    print("-" * 72)
    for name, param in model.named_parameters():
        count = param.numel()
        total_params += count
        shape_str = str(list(param.shape))
        print(f"{name:<35} {shape_str:<20} {count:<10} {count * 1:<8}")
    print("=" * 72)
    print(f"Total de Parâmetros:      {total_params:,}")
    print(f"Tamanho em Float32:       {(total_params * 4) / 1024.0:.2f} KB")
    print(f"Tamanho estimado em Int8: {(total_params * 1) / 1024.0:.2f} KB  <-- [Compatível c/ SRAM 512KB ESP32-S3]")
    print("=" * 72)


# ==============================================================================
# LOOP DE TREINO E AVALIAÇÃO
# ==============================================================================
def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    num_steps: int,
    encoding: str,
) -> Tuple[float, float, float]:
    """Treina uma época do modelo."""
    model.train()
    total_loss = 0.0
    correct = 0
    total_samples = 0
    firing_rates = []

    for images, targets in tqdm(train_loader, desc="Treino", leave=False):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        spk_out, mem_out, sparsity_dict = model(
            images, num_steps=num_steps, encoding=encoding, return_sparsity=True
        )

        loss = criterion(spk_out, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * targets.size(0)

        spike_counts = spk_out.sum(dim=0)
        preds = spike_counts.argmax(dim=1)

        no_spike_mask = (spike_counts.sum(dim=1) == 0)
        if no_spike_mask.any():
            preds[no_spike_mask] = mem_out[-1, no_spike_mask].argmax(dim=1)

        correct += (preds == targets).sum().item()
        total_samples += targets.size(0)
        firing_rates.append(sparsity_dict["overall_firing_rate"])

    avg_loss = total_loss / max(total_samples, 1)
    acc = correct / max(total_samples, 1)
    avg_fr = float(np.mean(firing_rates)) if firing_rates else 0.0
    return avg_loss, acc, avg_fr


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_steps: int,
    encoding: str,
) -> Dict[str, float]:
    """Avalia o modelo calculando Acurácia, F1-Score e Esparsidade."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct = 0

    tp = 0
    fp = 0
    fn = 0
    tn = 0

    overall_fr_list = []
    overall_sp_list = []

    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            targets = targets.to(device)

            spk_out, mem_out, sparsity_dict = model(
                images, num_steps=num_steps, encoding=encoding, return_sparsity=True
            )

            loss = criterion(spk_out, targets)
            total_loss += loss.item() * targets.size(0)

            spike_counts = spk_out.sum(dim=0)
            preds = spike_counts.argmax(dim=1)

            no_spike_mask = (spike_counts.sum(dim=1) == 0)
            if no_spike_mask.any():
                preds[no_spike_mask] = mem_out[-1, no_spike_mask].argmax(dim=1)

            correct += (preds == targets).sum().item()
            total_samples += targets.size(0)

            for p, t in zip(preds.cpu().numpy(), targets.cpu().numpy()):
                if t == 1 and p == 1:
                    tp += 1
                elif t == 0 and p == 1:
                    fp += 1
                elif t == 1 and p == 0:
                    fn += 1
                elif t == 0 and p == 0:
                    tn += 1

            overall_fr_list.append(sparsity_dict["overall_firing_rate"])
            overall_sp_list.append(sparsity_dict["overall_sparsity"])

    avg_loss = total_loss / max(total_samples, 1)
    acc = correct / max(total_samples, 1)

    precision = tp / max(tp + fp, 1e-9)
    recall = tp / max(tp + fn, 1e-9)
    f1 = 2 * (precision * recall) / max(precision + recall, 1e-9)

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "cat_precision": precision,
        "cat_recall": recall,
        "cat_f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "overall_firing_rate": float(np.mean(overall_fr_list)) if overall_fr_list else 0.0,
        "overall_sparsity": float(np.mean(overall_sp_list)) if overall_sp_list else 0.0,
    }


# ==============================================================================
# EXECUÇÃO PRINCIPAL
# ==============================================================================
def main():
    set_seed(CONFIG["seed"])

    # 1. Detecção do Dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 72)
    print("🚀 SpikeVision-S3: Treinamento CSNN no Google Colab")
    print("=" * 72)
    print(f"Dispositivo de execução: {device}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU detectada:           {gpu_name} ({vram_gb:.2f} GB VRAM)")
    else:
        print("⚠️ AVISO: GPU não detectada. O treinamento rodará na CPU (mais lento).")
        print("💡 Dica no Colab: 'Ambiente de execução' > 'Alterar tipo' > 'T4 GPU'.")
    print("=" * 72)

    # 2. Carregar Dados
    print(f"\nCarregando CIFAR-10 balanceado (Batch Size: {CONFIG['batch_size']}, Workers: {CONFIG['num_workers']})...")
    train_loader, test_loader = get_dataloaders(
        data_dir=CONFIG["data_dir"],
        batch_size=CONFIG["batch_size"],
        num_workers=CONFIG["num_workers"],
        in_channels=CONFIG["in_channels"],
        seed=CONFIG["seed"],
    )
    print(f"✓ Amostras de Treino: {len(train_loader.dataset):,} | Amostras de Teste: {len(test_loader.dataset):,}")

    # 3. Inicializar Modelo SNN
    model = SpikeVisionCSNN(
        in_channels=CONFIG["in_channels"],
        num_classes=CONFIG["num_classes"],
        beta=CONFIG["beta"],
        slope=CONFIG["slope"],
    ).to(device)

    print_model_summary(model)

    # 4. Otimizador, Scheduler e Loss
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"], eta_min=1e-5)
    criterion = SF.ce_rate_loss()

    best_acc = 0.0
    best_checkpoint_path = CONFIG["checkpoint_path"]
    history = []

    print("\nIniciando treinamento SNN:")
    print(f"Épocas: {CONFIG['epochs']} | Timesteps (T): {CONFIG['num_steps']} | Codificação: {CONFIG['encoding']} | Beta LIF: {CONFIG['beta']}\n")

    start_total_time = time.time()

    for epoch in range(1, CONFIG["epochs"] + 1):
        epoch_start = time.time()

        train_loss, train_acc, train_fr = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_steps=CONFIG["num_steps"],
            encoding=CONFIG["encoding"],
        )

        test_metrics = evaluate(
            model=model,
            data_loader=test_loader,
            criterion=criterion,
            device=device,
            num_steps=CONFIG["num_steps"],
            encoding=CONFIG["encoding"],
        )

        scheduler.step()
        epoch_time = time.time() - epoch_start

        test_acc = test_metrics["accuracy"]
        test_loss = test_metrics["loss"]
        f1 = test_metrics["cat_f1"]
        precision = test_metrics["cat_precision"]
        recall = test_metrics["cat_recall"]
        overall_fr = test_metrics["overall_firing_rate"]
        sparsity = test_metrics["overall_sparsity"]

        print(
            f"Época [{epoch:02d}/{CONFIG['epochs']:02d}] ({epoch_time:4.1f}s) | "
            f"Treino Loss: {train_loss:.4f} Acc: {train_acc*100:5.2f}% | "
            f"Teste Loss: {test_loss:.4f} Acc: {test_acc*100:5.2f}% | "
            f"Gato F1: {f1:.3f} (P: {precision:.2f}, R: {recall:.2f}) | "
            f"Esparsidade: {sparsity*100:5.1f}%"
        )

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_firing_rate": train_fr,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_cat_f1": f1,
            "test_cat_precision": precision,
            "test_cat_recall": recall,
            "test_sparsity": sparsity,
            "epoch_time_sec": epoch_time,
        }
        history.append(record)

        # Salvar checkpoint se superar a melhor acurácia de teste
        if test_acc >= best_acc:
            best_acc = test_acc
            checkpoint_payload = {
                "epoch": epoch,
                "best_acc": best_acc,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": {
                    "in_channels": CONFIG["in_channels"],
                    "num_classes": CONFIG["num_classes"],
                    "beta": CONFIG["beta"],
                    "slope": CONFIG["slope"],
                    "num_steps": CONFIG["num_steps"],
                    "encoding": CONFIG["encoding"],
                },
                "metrics": test_metrics,
            }
            torch.save(checkpoint_payload, best_checkpoint_path)
            print(f"  --> ⭐ Novo melhor modelo salvo: {best_checkpoint_path} (Acurácia: {best_acc*100:.2f}%)")

    total_duration = time.time() - start_total_time

    # Salvar histórico de treinamento
    with open(CONFIG["history_path"], "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 72)
    print("🏁 Treinamento Concluído com Sucesso!")
    print(f"Tempo Total:             {total_duration:.2f} segundos ({total_duration/60:.1f} min)")
    print(f"Melhor Acurácia Teste:   {best_acc * 100:.2f}%")
    print(f"Checkpoint Salvo:        {best_checkpoint_path}")
    print(f"Histórico Salvo:         {CONFIG['history_path']}")
    print("=" * 72)

    # 5. Download automático se estiver rodando no Google Colab
    if "google.colab" in sys.modules:
        print("\n📥 Ambiente Google Colab detectado! Iniciando download automático do checkpoint...")
        try:
            from google.colab import files
            files.download(best_checkpoint_path)
            print("✓ Download iniciado no navegador! Verifique sua pasta de Downloads.")
        except Exception as e:
            print(f"⚠️ Não foi possível iniciar o download automático: {e}")
            print(f"💡 Você pode baixar o arquivo '{best_checkpoint_path}' clicando com o botão direito na aba Arquivos à esquerda.")
    else:
        print(f"\n💡 Dica: Mova o arquivo '{best_checkpoint_path}' para 'training/checkpoints/' para quantização.")


if __name__ == "__main__" or "google.colab" in sys.modules:
    main()
