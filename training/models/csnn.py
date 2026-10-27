"""
SpikeVision-S3: Convolutional Spiking Neural Network (CSNN) para ESP32-S3.

Arquitetura ultra-leve projetada especificamente para:
1. Caber confortavelmente nos 512 KB de SRAM interna do ESP32-S3.
2. Facilitar quantização Int8 e exportação direta em matrizes C/C++ (Fase 3).
3. Maximizar esparsidade sináptica (SOP vs MAC).
"""

from typing import Tuple, Dict, Any, Optional, Union
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

from training.datasets.cifar10_cat import encode_spikes


class SpikeVisionCSNN(nn.Module):
    """
    Convolutional Spiking Neural Network (CSNN) para classificação CIFAR-10 (10 classes).

    Estrutura:
        - Layer 1: Conv2d(in_channels, 16, 3x3, pad=1, bias=False) -> BatchNorm2d(16) -> MaxPool2d(2) -> Leaky LIF
        - Layer 2: Conv2d(16, 32, 3x3, pad=1, bias=False) -> BatchNorm2d(32) -> MaxPool2d(2) -> Leaky LIF
        - Layer 3: Conv2d(32, 32, 3x3, pad=1, bias=False) -> BatchNorm2d(32) -> MaxPool2d(2) -> Leaky LIF
        - Head: Flatten -> Linear(32 * 4 * 4, num_classes, bias=False) -> Leaky LIF

    Total de Parâmetros: ~19.500 (~19.5 KB em Int8) para 10 classes; ~15.400 (~15.4 KB em Int8) para 2 classes.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        beta: float = 0.9,
        slope: float = 25.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.beta = beta

        # Gradiente substituto para BPTT
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
        """Inicializa os potenciais de membrana U das 4 camadas com zeros."""
        mem1 = torch.zeros((batch_size, 16, 16, 16), device=device)
        mem2 = torch.zeros((batch_size, 32, 8, 8), device=device)
        mem3 = torch.zeros((batch_size, 32, 4, 4), device=device)
        mem4 = torch.zeros((batch_size, self.num_classes), device=device)
        return mem1, mem2, mem3, mem4

    def forward(
        self,
        x: torch.Tensor,
        num_steps: int = 8,
        encoding: str = "rate",
        return_sparsity: bool = True,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]]:
        """
        Propagação direta temporal na SpikeVisionCSNN.

        Args:
            x: Tensor de entrada [B, C, H, W] ou [T, B, C, H, W].
            num_steps: Número de timesteps discretos T (se x for 4D).
            encoding: Método de codificação ('rate' ou 'direct').
            return_sparsity: Se True, calcula e retorna estatísticas de esparsidade.

        Returns:
            spk_out_rec: Tensor [T, B, num_classes] com spikes da saída.
            mem_out_rec: Tensor [T, B, num_classes] com potencial de membrana da saída.
            sparsity_metrics (opcional): Dicionário com firing rates e esparsidade de cada camada.
        """
        # Se entrada for estática [B, C, H, W], codificar em trem de spikes temporal
        if x.dim() == 4:
            spk_in = encode_spikes(x, num_steps=num_steps, method=encoding)
            t_steps = num_steps
            batch_size = x.size(0)
        elif x.dim() == 5:
            spk_in = x
            t_steps = x.size(0)
            batch_size = x.size(1)
        else:
            raise ValueError(f"Entrada com shape inválido: {x.shape}. Esperado 4D ou 5D.")

        device = x.device
        mem1, mem2, mem3, mem4 = self.init_membrane_potentials(batch_size, device)

        spk_out_rec = []
        mem_out_rec = []

        # Contadores para estatísticas de esparsidade
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

        spk_out = torch.stack(spk_out_rec, dim=0)  # [T, B, num_classes]
        mem_out = torch.stack(mem_out_rec, dim=0)  # [T, B, num_classes]

        if not return_sparsity:
            return spk_out, mem_out

        # Métricas de Firing Rate e Esparsidade por neurônio por passo de tempo
        total_neurons_l1 = t_steps * batch_size * (16 * 16 * 16)
        total_neurons_l2 = t_steps * batch_size * (32 * 8 * 8)
        total_neurons_l3 = t_steps * batch_size * (32 * 4 * 4)
        total_neurons_l4 = t_steps * batch_size * self.num_classes
        total_neurons_all = (
            total_neurons_l1 + total_neurons_l2 + total_neurons_l3 + total_neurons_l4
        )
        total_spikes_all = (
            spk_count_l1 + spk_count_l2 + spk_count_l3 + spk_count_l4
        )

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


def count_parameters(model: nn.Module) -> Dict[str, Any]:
    """
    Calcula a contagem de parâmetros do modelo e estimativas de memória.

    Returns:
        Dicionário com estatísticas de parâmetros e footprints Float32/Int8.
    """
    total_params = 0
    trainable_params = 0
    layer_info = []

    for name, param in model.named_parameters():
        count = param.numel()
        total_params += count
        if param.requires_grad:
            trainable_params += count
        layer_info.append({
            "name": name,
            "shape": list(param.shape),
            "params": count,
            "float32_bytes": count * 4,
            "int8_bytes": count * 1,
        })

    float32_kb = (total_params * 4) / 1024.0
    int8_kb = (total_params * 1) / 1024.0

    stats = {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "float32_kb": float32_kb,
        "int8_kb": int8_kb,
        "layers": layer_info,
    }
    return stats


def print_model_summary(model: nn.Module):
    """Exibe um sumário legível da arquitetura e ocupação de memória."""
    stats = count_parameters(model)
    print("=" * 72)
    print("                 SpikeVision-S3 CSNN Model Summary")
    print("=" * 72)
    print(f"{'Layer / Parameter':<35} {'Shape':<20} {'Params':<10} {'Int8 (B)':<8}")
    print("-" * 72)
    for layer in stats["layers"]:
        shape_str = str(layer["shape"])
        print(f"{layer['name']:<35} {shape_str:<20} {layer['params']:<10} {layer['int8_bytes']:<8}")
    print("=" * 72)
    print(f"Total de Parâmetros:      {stats['total_parameters']:,}")
    print(f"Parâmetros Treináveis:    {stats['trainable_parameters']:,}")
    print(f"Tamanho em Float32:       {stats['float32_kb']:.2f} KB")
    print(f"Tamanho estimado em Int8: {stats['int8_kb']:.2f} KB  <-- [Excelente p/ 512 KB SRAM do S3]")
    print("=" * 72)


if __name__ == "__main__":
    model = SpikeVisionCSNN(num_classes=10)
    print_model_summary(model)

    x = torch.rand(4, 3, 32, 32)
    spk_out, mem_out, sparsity = model(x, num_steps=8, encoding="rate", return_sparsity=True)
    print(f"Input shape:      {tuple(x.shape)}")
    print(f"Spike out shape:  {tuple(spk_out.shape)}  (esperado: (8, 4, 10))")
    print(f"Mem out shape:    {tuple(mem_out.shape)}  (esperado: (8, 4, 10))")
    print(f"Overall sparsity: {sparsity['overall_sparsity'] * 100:.2f}%")
