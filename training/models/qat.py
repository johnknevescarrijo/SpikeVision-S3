"""
SpikeVision-S3: Quantization-Aware Training (QAT) com Straight-Through Estimator (STE).

Simula quantização simétrica Int8 [-127, 127] por canal (axis=0) ou por tensor
durante o forward, mantendo o gradiente intacto no backward via STE para que
a rede se adapte à perda de precisão antes da exportação para o ESP32-S3.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def fake_quantize_int8_ste(
    weight: torch.Tensor,
    per_channel: bool = True,
) -> torch.Tensor:
    """
    Aplica quantização simétrica falsa (fake quantization) Int8 [-127, 127]
    com Straight-Through Estimator (STE) no backward.

    Args:
        weight: Tensor de pesos Float32 [C_out, ...].
        per_channel: Se True, calcula uma escala por canal de saída (axis=0).
                     Se False, calcula uma única escala por tensor.

    Returns:
        Tensor de pesos quantizado-dequantizado com gradiente STE.
    """
    if per_channel:
        w_flat = weight.detach().reshape(weight.shape[0], -1)
        w_max = w_flat.abs().max(dim=1).values
        s_w = torch.where(w_max < 1e-8, torch.ones_like(w_max), 127.0 / w_max)
        s_w = s_w.view(-1, *([1] * (weight.dim() - 1)))
    else:
        w_max = weight.detach().abs().max()
        s_w = torch.where(w_max < 1e-8, torch.ones_like(w_max), 127.0 / w_max)

    w_q = torch.clamp(torch.round(weight * s_w), -127.0, 127.0) / s_w
    return weight + (w_q - weight).detach()


class QATConv2d(nn.Conv2d):
    """
    Conv2d com suporte a QAT simétrico Int8 via STE.
    Mantém exatamente os mesmos nomes de parâmetros no state_dict que nn.Conv2d.
    """

    def __init__(
        self,
        *args,
        per_channel: bool = True,
        qat_enabled: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.per_channel = per_channel
        self.qat_enabled = qat_enabled

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        w = fake_quantize_int8_ste(self.weight, per_channel=self.per_channel) if self.qat_enabled else self.weight
        return self._conv_forward(input, w, self.bias)


class QATLinear(nn.Linear):
    """
    Linear com suporte a QAT simétrico Int8 via STE.
    Mantém exatamente os mesmos nomes de parâmetros no state_dict que nn.Linear.
    """

    def __init__(
        self,
        *args,
        per_channel: bool = True,
        qat_enabled: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.per_channel = per_channel
        self.qat_enabled = qat_enabled

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        w = fake_quantize_int8_ste(self.weight, per_channel=self.per_channel) if self.qat_enabled else self.weight
        return F.linear(input, w, self.bias)
