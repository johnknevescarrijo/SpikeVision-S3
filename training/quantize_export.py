"""
SpikeVision-S3: Quantização Int8 e Exportação para C/C++ (Fase 3).

Este script realiza:
1. Carregamento do checkpoint treinado (Float32).
2. Fusão analítica exata de Conv2d + BatchNorm2d para as camadas 1, 2 e 3.
3. Quantização simétrica Int8 por camada para pesos (int8_t) e Int32 para bias e limiares (V_th).
4. Simulação inteira bit-accurate em Python reproduzindo a mesma aritmética de ponto fixo
   que executará no ESP32-S3.
5. Validação comparativa nos 2.000 exemplos de teste do CIFAR-10 (Float32 vs Int8).
6. Exportação automática de:
   - firmware/include/weights.h: Cabeçalho C/C++ com matrizes alinhadas e constantes.
   - firmware/test_samples/test_samples.h: Amostras de teste reais com ground-truth e predições esperadas.
   - training/checkpoints/quantization_report.json: Relatório detalhado de quantização e memória.
"""

import os
import sys

# Adiciona o diretório raiz ao sys.path para execução autônoma
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import argparse
import logging
from typing import Tuple, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from training.models.csnn import SpikeVisionCSNN
from training.datasets.cifar10_cat import (
    get_dataloaders,
    get_cifar10_cat_dataloaders,
    get_cifar10_multiclass_dataloaders,
    CIFAR10_CLASS_NAMES,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)

# Constantes de Ponto Fixo para o decaimento Leaky LIF (beta = 0.9)
# 58982 / 65536 = 0.899993896484375 (~0.9000 com erro < 0.001%)
BETA_NUM = 58982
BETA_SHIFT = 16


def fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Realiza a fusão analítica dos parâmetros de Conv2d e BatchNorm2d em modo eval().
    
    Y_conv = W * X
    Y_bn   = gamma * (Y_conv - mean) / sqrt(var + eps) + beta
           = (gamma / sqrt(var + eps)) * W * X + (beta - gamma * mean / sqrt(var + eps))
    
    Returns:
        w_fused: Pesos fundidos [C_out, C_in, K_h, K_w]
        b_fused: Biases fundidos [C_out]
    """
    gamma = bn.weight
    beta = bn.bias
    mean = bn.running_mean
    var = bn.running_var
    eps = bn.eps

    scale = gamma / torch.sqrt(var + eps)
    w_fused = conv.weight * scale.view(-1, 1, 1, 1)
    b_fused = beta - mean * scale

    return w_fused, b_fused


def quantize_layer(
    w_fused: torch.Tensor,
    b_fused: torch.Tensor,
    s_in: float,
    v_th: float = 1.0,
    per_channel: bool = True,
) -> Dict[str, Any]:
    """
    Quantiza os parâmetros de uma camada para Int8 (pesos) e Int32 (bias e threshold).

    Args:
        w_fused: Pesos fundidos Float32 [C_out, ...].
        b_fused: Biases fundidos Float32 [C_out].
        s_in: Escala de entrada (127.0 para camada de pixels, 1.0 para spikes binários).
        v_th: Limiar de disparo Float32 (1.0).
        per_channel: Se True, quantiza simetricamente por canal de saída (axis=0).
                     Se False, quantiza com uma única escala por tensor.

    Returns:
        Dicionário com tensores quantizados e parâmetros de escala.
    """
    if per_channel:
        w_flat = w_fused.reshape(w_fused.shape[0], -1)
        w_max_ch = torch.max(torch.abs(w_flat), dim=1).values
        s_w_ch = torch.where(w_max_ch < 1e-8, torch.ones_like(w_max_ch), 127.0 / w_max_ch)
        s_w_view = s_w_ch.view(-1, *([1] * (w_fused.dim() - 1)))

        w_q = torch.clamp(torch.round(w_fused * s_w_view), -127, 127).to(torch.int8)
        s_total_ch = s_w_ch * s_in
        b_q = torch.round(b_fused * s_total_ch).to(torch.int32)
        vth_q = torch.round(torch.full_like(s_total_ch, v_th) * s_total_ch).to(torch.int32)

        return {
            "per_channel": True,
            "w_fused": w_fused,
            "b_fused": b_fused,
            "w_q": w_q,
            "b_q": b_q,
            "w_max": w_max_ch,
            "s_w": s_w_ch,
            "s_in": s_in,
            "s_total": s_total_ch,
            "vth_q": vth_q,
            "vth_float": v_th,
        }

    w_max = float(torch.max(torch.abs(w_fused)).item())
    if w_max < 1e-8:
        s_w = 1.0
    else:
        s_w = 127.0 / w_max

    s_total = s_w * s_in

    # Pesos em int8 [-127, 127]
    w_q = torch.clamp(torch.round(w_fused * s_w), -127, 127).to(torch.int8)

    # Bias e Threshold na escala acumulada inteira (int32)
    b_q = torch.round(b_fused * s_total).to(torch.int32)
    vth_q = int(round(v_th * s_total))

    return {
        "per_channel": False,
        "w_fused": w_fused,
        "b_fused": b_fused,
        "w_q": w_q,
        "b_q": b_q,
        "w_max": w_max,
        "s_w": s_w,
        "s_in": s_in,
        "s_total": s_total,
        "vth_q": vth_q,
        "vth_float": v_th,
    }


class BitAccurateInt8CSNN:
    """
    Simulador Bit-Accurate da CSNN com aritmética puramente inteira (Int8/Int32).
    
    Implementa a mesma lógica matemática e estruturas de dados que o código C++
    executará no microcontrolador ESP32-S3:
    - Entradas de imagem quantizadas em [0, 127].
    - Camadas 2, 3 e 4 operando exclusivamente com Spikes binários {0, 1} e SOP (Sum of Products).
    - Decaimento de membrana Leaky LIF em ponto fixo 16-bit: (mem * BETA_NUM) >> BETA_SHIFT.
    - Reset por subtração: mem = mem - reset * V_th.
    """

    @staticmethod
    def _prepare_vth(vth_q: Any, is_conv: bool):
        """Prepara vth_q escalar ou vetor por canal para broadcasting em Int32."""
        if isinstance(vth_q, (int, float, np.integer)):
            return int(vth_q)
        t = torch.as_tensor(vth_q, dtype=torch.int32)
        if t.ndim == 0:
            return int(t.item())
        if is_conv and t.ndim == 1:
            return t.view(1, -1, 1, 1)
        if is_conv and t.ndim == 3:
            return t.unsqueeze(0)
        if not is_conv and t.ndim == 1:
            return t.view(1, -1)
        return t

    @staticmethod
    def _build_sop_kernel_mask(h: int, w: int) -> torch.Tensor:
        """Mapa [1, 1, H, W] com contagem de posições válidas do kernel 3x3 (padding=1) por pixel."""
        ky = torch.full((h,), 3, dtype=torch.int32)
        ky[0] = 2
        ky[-1] = 2
        kx = torch.full((w,), 3, dtype=torch.int32)
        kx[0] = 2
        kx[-1] = 2
        return (ky.view(h, 1) * kx.view(1, w)).view(1, 1, h, w)

    def __init__(
        self,
        quant_params: Dict[str, Dict[str, Any]],
        num_steps: int = 8,
        early_exit_enabled: bool = False,
        early_exit_min_t: int = 2,
        early_exit_margin: int = 0,
    ):
        self.p1 = quant_params["layer1"]
        self.p2 = quant_params["layer2"]
        self.p3 = quant_params["layer3"]
        self.p4 = quant_params["layer4"]
        self.num_steps = num_steps
        self.num_classes = self.p4["w_q"].shape[0]
        self.early_exit_enabled = bool(early_exit_enabled)
        self.early_exit_min_t = max(1, int(early_exit_min_t))
        self.early_exit_margin = int(early_exit_margin)

        self._k_weight_16 = self._build_sop_kernel_mask(16, 16)
        self._k_weight_8 = self._build_sop_kernel_mask(8, 8)
        self.last_timesteps_executed = None
        self.last_sample_sops = None

        # Tensores para convolução acelerada via PyTorch com aritmética exata inteira
        self.w1_t = self.p1["w_q"].float()
        self.b1_t = self.p1["b_q"].float()
        self.vth1 = self._prepare_vth(self.p1["vth_q"], is_conv=True)

        self.w2_t = self.p2["w_q"].float()
        self.b2_t = self.p2["b_q"].float()
        self.vth2 = self._prepare_vth(self.p2["vth_q"], is_conv=True)

        self.w3_t = self.p3["w_q"].float()
        self.b3_t = self.p3["b_q"].float()
        self.vth3 = self._prepare_vth(self.p3["vth_q"], is_conv=True)

        self.w4_t = self.p4["w_q"].float()
        self.b4_t = self.p4["b_q"].float()
        self.vth4 = self._prepare_vth(self.p4["vth_q"], is_conv=False)

    def forward_batch(
        self, imgs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Executa a simulação inteira em lote.
        
        Args:
            imgs: Tensor Float32 [B, 3, 32, 32] com valores em [0.0, 1.0].
            
        Returns:
            preds: Tensor [B] com rótulos preditos [0 .. num_classes-1].
            spike_counts: Tensor [B, num_classes] com contagens totais de spikes na saída.
            final_mem: Tensor [B, num_classes] com potencial de membrana final da camada 4.
            sparsity_metrics: Estatísticas de esparsidade de disparo.
        """
        B = imgs.size(0)
        # Quantiza pixels de entrada para [0, 127]
        x_q = torch.round(imgs * 127.0).float()

        # Membranas dos neurônios inicializadas em zero
        mem1 = torch.zeros((B, 16, 16, 16), dtype=torch.int32)
        mem2 = torch.zeros((B, 32, 8, 8), dtype=torch.int32)
        mem3 = torch.zeros((B, 32, 4, 4), dtype=torch.int32)
        mem4 = torch.zeros((B, self.num_classes), dtype=torch.int32)

        # Na Camada 1 com Direct Coding, a imagem estática pode ser pré-convoluída
        # uma única vez (economia computacional idêntica ao microcontrolador)
        c1 = F.conv2d(x_q, self.w1_t, self.b1_t, padding=1)
        p1 = F.max_pool2d(c1, 2).to(torch.int32)

        spk_counts = torch.zeros((B, self.num_classes), dtype=torch.int32)
        final_mem_out = torch.zeros((B, self.num_classes), dtype=torch.int32)
        active = torch.ones(B, dtype=torch.bool)
        timesteps_executed = torch.full((B,), self.num_steps, dtype=torch.int32)
        sample_sops = torch.zeros(B, dtype=torch.int64)

        total_spks_l1 = 0
        total_spks_l2 = 0
        total_spks_l3 = 0
        total_spks_l4 = 0

        for t in range(self.num_steps):
            if not active.any():
                break

            active_i = active.to(torch.int32)
            active_i4d = active_i.view(B, 1, 1, 1)
            active_l = active.to(torch.int64)

            # --- Camada 1 ---
            r1 = (mem1 >= self.vth1).to(torch.int32)
            mem1 = ((mem1 * BETA_NUM) >> BETA_SHIFT) + p1 - r1 * self.vth1
            s1 = (mem1 >= self.vth1).to(torch.int32)
            total_spks_l1 += int((s1 * active_i4d).sum().item())

            # --- Camada 2 ---
            c2 = F.conv2d(s1.float(), self.w2_t, self.b2_t, padding=1)
            p2 = F.max_pool2d(c2, 2).to(torch.int32)
            r2 = (mem2 >= self.vth2).to(torch.int32)
            mem2 = ((mem2 * BETA_NUM) >> BETA_SHIFT) + p2 - r2 * self.vth2
            s2 = (mem2 >= self.vth2).to(torch.int32)
            total_spks_l2 += int((s2 * active_i4d).sum().item())
            sample_sops += (s1 * self._k_weight_16).sum(dim=(1, 2, 3)).to(torch.int64) * 32 * active_l

            # --- Camada 3 ---
            c3 = F.conv2d(s2.float(), self.w3_t, self.b3_t, padding=1)
            p3 = F.max_pool2d(c3, 2).to(torch.int32)
            r3 = (mem3 >= self.vth3).to(torch.int32)
            mem3 = ((mem3 * BETA_NUM) >> BETA_SHIFT) + p3 - r3 * self.vth3
            s3 = (mem3 >= self.vth3).to(torch.int32)
            total_spks_l3 += int((s3 * active_i4d).sum().item())
            sample_sops += (s2 * self._k_weight_8).sum(dim=(1, 2, 3)).to(torch.int64) * 32 * active_l

            # --- Camada 4 (Head Linear) ---
            flat = s3.flatten(start_dim=1).float()
            cur4 = F.linear(flat, self.w4_t, self.b4_t).to(torch.int32)
            r4 = (mem4 >= self.vth4).to(torch.int32)
            mem4 = ((mem4 * BETA_NUM) >> BETA_SHIFT) + cur4 - r4 * self.vth4
            s4 = (mem4 >= self.vth4).to(torch.int32)
            total_spks_l4 += int((s4 * active_i.view(B, 1)).sum().item())
            sample_sops += s3.sum(dim=(1, 2, 3)).to(torch.int64) * self.num_classes * active_l

            spk_counts += s4 * active_i.view(B, 1)

            # --- Critério de Early-Exit Temporal ---
            timesteps_done = t + 1
            remaining = self.num_steps - timesteps_done
            if self.early_exit_enabled and remaining > 0 and timesteps_done >= self.early_exit_min_t:
                if self.num_classes > 1:
                    top2 = torch.topk(spk_counts, k=2, dim=1).values
                    gap = top2[:, 0] - top2[:, 1]
                else:
                    gap = spk_counts[:, 0]
                if self.early_exit_margin > 0:
                    cond = gap >= self.early_exit_margin
                else:
                    cond = gap > remaining
                exit_now = active & cond
                if exit_now.any():
                    timesteps_executed[exit_now] = timesteps_done
                    final_mem_out[exit_now] = mem4[exit_now]
                    active[exit_now] = False

        if active.any():
            final_mem_out[active] = mem4[active]

        self.last_timesteps_executed = timesteps_executed
        self.last_sample_sops = sample_sops

        # Regra de decisão multi-classe idêntica à do C++ (maior contagem de spikes;
        # desempate pelo maior potencial final de membrana mem4)
        preds = torch.zeros(B, dtype=torch.long)
        for b in range(B):
            best_c = 0
            max_spk = spk_counts[b, 0].item()
            best_mem = final_mem_out[b, 0].item()
            for c in range(1, self.num_classes):
                s = spk_counts[b, c].item()
                m = final_mem_out[b, c].item()
                if s > max_spk or (s == max_spk and m > best_mem):
                    max_spk = s
                    best_mem = m
                    best_c = c
            preds[b] = best_c

        # Estatísticas de esparsidade e execução adaptativa
        total_t = int(timesteps_executed.sum().item())
        n_l1 = total_t * (16 * 16 * 16)
        n_l2 = total_t * (32 * 8 * 8)
        n_l3 = total_t * (32 * 4 * 4)
        n_l4 = total_t * self.num_classes
        total_neurons = n_l1 + n_l2 + n_l3 + n_l4
        total_spks = total_spks_l1 + total_spks_l2 + total_spks_l3 + total_spks_l4
        avg_t = float(timesteps_executed.float().mean().item())

        sparsity_metrics = {
            "firing_rate_l1": total_spks_l1 / max(n_l1, 1),
            "firing_rate_l2": total_spks_l2 / max(n_l2, 1),
            "firing_rate_l3": total_spks_l3 / max(n_l3, 1),
            "firing_rate_l4": total_spks_l4 / max(n_l4, 1),
            "overall_firing_rate": total_spks / max(total_neurons, 1),
            "overall_sparsity": 1.0 - (total_spks / max(total_neurons, 1)),
            "avg_timesteps": avg_t,
            "timestep_reduction_percent": 100.0 * (1.0 - avg_t / max(self.num_steps, 1)),
            "avg_sops": float(sample_sops.float().mean().item()),
        }

        return preds, spk_counts, final_mem_out, sparsity_metrics

    def forward_single(
        self, img: torch.Tensor
    ) -> Tuple[int, List[int], List[int]]:
        """Executa a simulação para uma única amostra de imagem [3, 32, 32]."""
        preds, spk_counts, final_mem, _ = self.forward_batch(img.unsqueeze(0))
        return (
            int(preds[0].item()),
            spk_counts[0].cpu().tolist(),
            final_mem[0].cpu().tolist(),
        )


def evaluate_comparison(
    float_model: nn.Module,
    int8_sim: BitAccurateInt8CSNN,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_steps: int = 8,
) -> Tuple[Dict[str, Any], Dict[str, Any], float]:
    """
    Avalia e compara amostra por amostra os modelos Float32 e Int8 em todo o dataset de teste.
    """
    float_model.eval()

    all_targets = []
    all_f_preds = []
    all_i_preds = []

    f_spk_rec = []
    f_mem_rec = []

    total_sparsity_f = []
    total_sparsity_i = []

    logger.info("Executando avaliação comparativa Float32 vs Int8...")
    with torch.no_grad():
        for batch_idx, (imgs, targets) in enumerate(test_loader):
            imgs = imgs.to(device)
            targets = targets.to(device)

            # --- Float32 Model ---
            spk_out, mem_out, s_dict_f = float_model(
                imgs, num_steps=num_steps, encoding="direct", return_sparsity=True
            )
            f_counts = spk_out.sum(dim=0)
            f_preds = f_counts.argmax(dim=1)
            no_spk = (f_counts.sum(dim=1) == 0)
            if no_spk.any():
                f_preds[no_spk] = mem_out[-1, no_spk].argmax(dim=1)

            # --- Int8 Simulator ---
            i_preds, _, _, s_dict_i = int8_sim.forward_batch(imgs.cpu())
            i_preds = i_preds.to(device)

            all_targets.append(targets.cpu())
            all_f_preds.append(f_preds.cpu())
            all_i_preds.append(i_preds.cpu())

            total_sparsity_f.append(s_dict_f["overall_sparsity"])
            total_sparsity_i.append(s_dict_i["overall_sparsity"])

    all_targets = torch.cat(all_targets).numpy()
    all_f_preds = torch.cat(all_f_preds).numpy()
    all_i_preds = torch.cat(all_i_preds).numpy()

    num_classes = getattr(float_model, "num_classes", int8_sim.num_classes)

    def calc_metrics(preds, targets):
        total = len(targets)
        acc = float(np.sum(preds == targets) / max(total, 1))

        per_class_f1 = {}
        per_class_accuracy = {}
        precisions = []
        recalls = []
        f1s = []

        class_names = CIFAR10_CLASS_NAMES if num_classes == 10 else [f"class_{i}" for i in range(num_classes)]

        for c in range(num_classes):
            c_name = class_names[c] if c < len(class_names) else f"class_{c}"
            tp = int(np.sum((preds == c) & (targets == c)))
            fp = int(np.sum((preds == c) & (targets != c)))
            fn = int(np.sum((preds != c) & (targets == c)))
            c_total = int(np.sum(targets == c))

            prec = float(tp / max(tp + fp, 1e-9)) if (tp + fp) > 0 else 0.0
            rec = float(tp / max(tp + fn, 1e-9)) if (tp + fn) > 0 else 0.0
            f1 = float(2.0 * prec * rec / max(prec + rec, 1e-9)) if (prec + rec) > 0 else 0.0
            class_acc = float(tp / max(c_total, 1)) if c_total > 0 else 0.0

            per_class_f1[c_name] = f1
            per_class_accuracy[c_name] = class_acc
            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)

        macro_precision = float(np.mean(precisions)) if precisions else 0.0
        macro_recall = float(np.mean(recalls)) if recalls else 0.0
        macro_f1 = float(np.mean(f1s)) if f1s else 0.0

        res = {
            "accuracy": acc,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "per_class_f1": per_class_f1,
            "per_class_accuracy": per_class_accuracy,
            "total_samples": total,
        }

        # Retrocompatibilidade com métricas escalares
        cat_idx = 3 if num_classes == 10 else 1
        if cat_idx < num_classes:
            res["precision"] = precisions[cat_idx]
            res["recall"] = recalls[cat_idx]
            res["f1_score"] = f1s[cat_idx]
            res["tp"] = int(np.sum((preds == cat_idx) & (targets == cat_idx)))
            res["fp"] = int(np.sum((preds == cat_idx) & (targets != cat_idx)))
            res["fn"] = int(np.sum((preds != cat_idx) & (targets == cat_idx)))
            res["tn"] = int(np.sum((preds != cat_idx) & (targets != cat_idx)))
        else:
            res["precision"] = macro_precision
            res["recall"] = macro_recall
            res["f1_score"] = macro_f1
            res["tp"] = int(np.sum(preds == targets))
            res["fp"] = 0
            res["fn"] = int(np.sum(preds != targets))
            res["tn"] = 0

        return res

    metrics_f = calc_metrics(all_f_preds, all_targets)
    metrics_f["mean_sparsity"] = float(np.mean(total_sparsity_f))

    metrics_i = calc_metrics(all_i_preds, all_targets)
    metrics_i["mean_sparsity"] = float(np.mean(total_sparsity_i))

    agreement = float(np.mean(all_f_preds == all_i_preds))
    agreement_count = int(np.sum(all_f_preds == all_i_preds))

    return metrics_f, metrics_i, agreement


def export_weights_h(
    quant_params: Dict[str, Dict[str, Any]],
    output_path: str,
    config: Dict[str, Any],
):
    """
    Exporta os pesos quantizados e configurações em um cabeçalho C/C++ (firmware/include/weights.h).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    p1 = quant_params["layer1"]
    p2 = quant_params["layer2"]
    p3 = quant_params["layer3"]
    p4 = quant_params["layer4"]

    w1 = p1["w_q"].numpy().astype(np.int8)  # [16, 3, 3, 3]
    b1 = p1["b_q"].numpy().astype(np.int32) # [16]

    w2 = p2["w_q"].numpy().astype(np.int8)  # [32, 16, 3, 3]
    b2 = p2["b_q"].numpy().astype(np.int32) # [32]

    w3 = p3["w_q"].numpy().astype(np.int8)  # [32, 32, 3, 3]
    b3 = p3["b_q"].numpy().astype(np.int32) # [32]

    w4 = p4["w_q"].numpy().astype(np.int8)  # [num_classes, 512]
    b4 = p4["b_q"].numpy().astype(np.int32) # [num_classes]
    num_classes = config.get("num_classes", w4.shape[0])

    def format_array_int8(arr: np.ndarray, indent: int = 4, items_per_line: int = 16) -> str:
        flat = arr.flatten()
        lines = []
        prefix = " " * indent
        for i in range(0, len(flat), items_per_line):
            chunk = flat[i:i + items_per_line]
            lines.append(prefix + ", ".join(f"{x:4d}" for x in chunk) + ",")
        # Remove trailing comma on last line
        if lines:
            lines[-1] = lines[-1].rstrip(",")
        return "\n".join(lines)

    def format_array_int32(arr: np.ndarray, indent: int = 4, items_per_line: int = 8) -> str:
        flat = arr.flatten()
        lines = []
        prefix = " " * indent
        for i in range(0, len(flat), items_per_line):
            chunk = flat[i:i + items_per_line]
            lines.append(prefix + ", ".join(f"{x:10d}" for x in chunk) + ",")
        if lines:
            lines[-1] = lines[-1].rstrip(",")
        return "\n".join(lines)

    per_channel = bool(p1.get("per_channel", False))
    total_weight_bytes = w1.nbytes + w2.nbytes + w3.nbytes + w4.nbytes
    total_bias_bytes = b1.nbytes + b2.nbytes + b3.nbytes + b4.nbytes
    total_bytes = total_weight_bytes + total_bias_bytes

    if per_channel:
        vth1_arr = np.asarray(p1["vth_q"], dtype=np.int32)
        vth2_arr = np.asarray(p2["vth_q"], dtype=np.int32)
        vth3_arr = np.asarray(p3["vth_q"], dtype=np.int32)
        vth4_arr = np.asarray(p4["vth_q"], dtype=np.int32)
        vth_block = f"""/* Fatores de Escala da Entrada e Limiares de Disparo (V_th) Per-Channel em Int32 */
#define SPIKEVISION_INPUT_SCALE        127
#define SPIKEVISION_PER_CHANNEL_QUANT  1

SPIKEVISION_ALIGN16 static const int32_t VTH_L1_CH[16] = {{
{format_array_int32(vth1_arr)}
}};

SPIKEVISION_ALIGN16 static const int32_t VTH_L2_CH[32] = {{
{format_array_int32(vth2_arr)}
}};

SPIKEVISION_ALIGN16 static const int32_t VTH_L3_CH[32] = {{
{format_array_int32(vth3_arr)}
}};

SPIKEVISION_ALIGN16 static const int32_t VTH_FC_CH[{num_classes}] = {{
{format_array_int32(vth4_arr)}
}};"""
    else:
        vth_block = f"""{vth_block}"""

    content = f"""/**
 * SpikeVision-S3: Pesos Quantizados Int8 e Constantes da CSNN para ESP32-S3.
 *
 * ARQUIVO GERADO AUTOMATICAMENTE POR training/quantize_export.py. NÃO EDITE MANUALMENTE.
 *
 * Topologia:
 *  - Camada 1: Conv2d(3, 16, 3x3) + BN (Fundidos) -> MaxPool2d(2) -> Leaky LIF (16x16x16)
 *  - Camada 2: Conv2d(16, 32, 3x3) + BN (Fundidos) -> MaxPool2d(2) -> Leaky LIF (32x8x8)
 *  - Camada 3: Conv2d(32, 32, 3x3) + BN (Fundidos) -> MaxPool2d(2) -> Leaky LIF (32x4x4)
 *  - Camada 4: Linear(512, {num_classes}) -> Leaky LIF ({num_classes})
 *
 * Memória Flash/ROM ocupada:
 *  - Pesos Int8:   {total_weight_bytes:6d} bytes (~{total_weight_bytes / 1024.0:.2f} KB)
 *  - Biases Int32: {total_bias_bytes:6d} bytes (~{total_bias_bytes / 1024.0:.2f} KB)
 *  - Total Parâmetros: {total_bytes:6d} bytes (~{total_bytes / 1024.0:.2f} KB)
 */

#pragma once

#include <stdint.h>
#include <stddef.h>

#if defined(__cplusplus) && (__cplusplus >= 201103L)
  #define SPIKEVISION_ALIGN16 alignas(16)
#elif defined(__GNUC__) || defined(__clang__)
  #define SPIKEVISION_ALIGN16 __attribute__((aligned(16)))
#elif defined(_MSC_VER)
  #define SPIKEVISION_ALIGN16 __declspec(align(16))
#else
  #define SPIKEVISION_ALIGN16
#endif

#ifdef __cplusplus
extern "C" {{
#endif

/* =========================================================================
 * Constantes Globais da Rede SNN
 * ========================================================================= */
#define SPIKEVISION_TIMESTEPS          {config.get('num_steps', 8)}
#define SPIKEVISION_IN_CHANNELS        {config.get('in_channels', 3)}
#define SPIKEVISION_INPUT_HEIGHT       32
#define SPIKEVISION_INPUT_WIDTH        32
#define SPIKEVISION_NUM_CLASSES        {num_classes}

/* Constantes de Ponto Fixo do Decaimento Leaky LIF (beta = 0.900) */
#define SPIKEVISION_BETA_NUM           {BETA_NUM}
#define SPIKEVISION_BETA_SHIFT         {BETA_SHIFT}

{vth_block}

/* Dimensões das Camadas */
#define CONV1_IN_CHANNELS              3
#define CONV1_OUT_CHANNELS             16
#define CONV1_KERNEL_SIZE              3
#define CONV1_PADDING                  1
#define POOL1_OUT_H                    16
#define POOL1_OUT_W                    16

#define CONV2_IN_CHANNELS              16
#define CONV2_OUT_CHANNELS             32
#define CONV2_KERNEL_SIZE              3
#define CONV2_PADDING                  1
#define POOL2_OUT_H                    8
#define POOL2_OUT_W                    8

#define CONV3_IN_CHANNELS              32
#define CONV3_OUT_CHANNELS             32
#define CONV3_KERNEL_SIZE              3
#define CONV3_PADDING                  1
#define POOL3_OUT_H                    4
#define POOL3_OUT_W                    4

#define FC_IN_FEATURES                 512
#define FC_OUT_FEATURES                {num_classes}

/* =========================================================================
 * Pesos e Biases Quantizados (Int8 / Int32)
 * Alinhamento em 16 bytes para compatibilidade ótima com instruções SIMD/PIE
 * ========================================================================= */

/* Camada 1: Pesos [16][3][3][3] = 432 bytes */
SPIKEVISION_ALIGN16 static const int8_t conv1_weights[CONV1_OUT_CHANNELS * CONV1_IN_CHANNELS * CONV1_KERNEL_SIZE * CONV1_KERNEL_SIZE] = {{
{format_array_int8(w1)}
}};

/* Camada 1: Biases [16] = 64 bytes */
SPIKEVISION_ALIGN16 static const int32_t conv1_bias[CONV1_OUT_CHANNELS] = {{
{format_array_int32(b1)}
}};

/* Camada 2: Pesos [32][16][3][3] = 4.608 bytes */
SPIKEVISION_ALIGN16 static const int8_t conv2_weights[CONV2_OUT_CHANNELS * CONV2_IN_CHANNELS * CONV2_KERNEL_SIZE * CONV2_KERNEL_SIZE] = {{
{format_array_int8(w2)}
}};

/* Camada 2: Biases [32] = 128 bytes */
SPIKEVISION_ALIGN16 static const int32_t conv2_bias[CONV2_OUT_CHANNELS] = {{
{format_array_int32(b2)}
}};

/* Camada 3: Pesos [32][32][3][3] = 9.216 bytes */
SPIKEVISION_ALIGN16 static const int8_t conv3_weights[CONV3_OUT_CHANNELS * CONV3_IN_CHANNELS * CONV3_KERNEL_SIZE * CONV3_KERNEL_SIZE] = {{
{format_array_int8(w3)}
}};

/* Camada 3: Biases [32] = 128 bytes */
SPIKEVISION_ALIGN16 static const int32_t conv3_bias[CONV3_OUT_CHANNELS] = {{
{format_array_int32(b3)}
}};

/* Camada 4: Pesos [{num_classes}][512] = {w4.nbytes} bytes */
SPIKEVISION_ALIGN16 static const int8_t fc_weights[FC_OUT_FEATURES * FC_IN_FEATURES] = {{
{format_array_int8(w4)}
}};

/* Camada 4: Biases [{num_classes}] = {b4.nbytes} bytes */
SPIKEVISION_ALIGN16 static const int32_t fc_bias[FC_OUT_FEATURES] = {{
{format_array_int32(b4)}
}};

#ifdef __cplusplus
}}
#endif
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info(f"Cabeçalho de pesos C/C++ salvo em: {output_path}")


def export_test_samples_h(
    test_loader: torch.utils.data.DataLoader,
    int8_sim: BitAccurateInt8CSNN,
    output_path: str,
    samples_per_class: int = 1,
):
    """
    Exporta amostras reais de teste cobrindo todas as classes para validação bit-a-bit no firmware.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    num_classes = int8_sim.num_classes
    class_names = CIFAR10_CLASS_NAMES if num_classes == 10 else ["non-cat", "cat"]

    samples_by_class: Dict[int, List[Tuple[torch.Tensor, int]]] = {c: [] for c in range(num_classes)}
    needed = num_classes * samples_per_class

    for imgs, targets in test_loader:
        for i in range(imgs.size(0)):
            target = int(targets[i].item())
            if target in samples_by_class and len(samples_by_class[target]) < samples_per_class:
                samples_by_class[target].append((imgs[i], target))
        if sum(len(v) for v in samples_by_class.values()) >= needed:
            break

    all_samples = []
    for c in range(num_classes):
        all_samples.extend(samples_by_class[c])

    total_samples = len(all_samples)
    sample_declarations = []
    test_cases_entries = []

    for idx, (img, target) in enumerate(all_samples):
        # Quantiza a imagem em uint8_t [0, 127] no formato CHW
        img_q = torch.round(img * 127.0).clamp(0, 127).numpy().astype(np.uint8)
        pred, spk_counts, final_mem = int8_sim.forward_single(img)

        sample_name = f"test_sample_{idx}"
        c_name = class_names[target] if target < len(class_names) else f"class_{target}"
        pred_name = class_names[pred] if pred < len(class_names) else f"class_{pred}"

        # Formata o array C
        flat = img_q.flatten()
        lines = []
        for j in range(0, len(flat), 16):
            chunk = flat[j:j + 16]
            lines.append("    " + ", ".join(f"{x:3d}" for x in chunk) + ",")
        if lines:
            lines[-1] = lines[-1].rstrip(",")
        array_body = "\n".join(lines)

        spk_str = ", ".join(str(s) for s in spk_counts)
        mem_str = ", ".join(str(m) for m in final_mem)

        decl = f"""/* Amostra {idx}: {c_name} (Label {target}) | Pred Int8: {pred_name} (Label {pred}) | Spikes: [{spk_str}] */
static const uint8_t {sample_name}[3 * 32 * 32] = {{
{array_body}
}};"""
        sample_declarations.append(decl)

        entry = f"""    {{
        "{c_name}",
        {sample_name},
        {target},
        {pred},
        {{{spk_str}}},
        {{{mem_str}}}
    }}"""
        test_cases_entries.append(entry)

    test_cases_body = ",\n".join(test_cases_entries)
    sample_decls_body = "\n\n".join(sample_declarations)

    header_content = f"""/**
 * SpikeVision-S3: Amostras Reais de Teste para Validação Bit-a-Bit no ESP32-S3.
 *
 * ARQUIVO GERADO AUTOMATICAMENTE POR training/quantize_export.py.
 * Contém {total_samples} amostras ({samples_per_class} por classe para {num_classes} classes)
 * extraídas do conjunto de teste do CIFAR-10, junto com os resultados exatos esperados
 * do simulador de inferência Int8 de ponto fixo.
 */

#pragma once

#include <stdint.h>
#include <stddef.h>
#include "weights.h"

#ifdef __cplusplus
extern "C" {{
#endif

#define TEST_SAMPLE_COUNT       {total_samples}
#define TEST_SAMPLE_CHANNELS    3
#define TEST_SAMPLE_HEIGHT      32
#define TEST_SAMPLE_WIDTH       32
#define TEST_SAMPLE_SIZE        (TEST_SAMPLE_CHANNELS * TEST_SAMPLE_HEIGHT * TEST_SAMPLE_WIDTH)

typedef struct {{
    const char* description;
    const uint8_t* image_data;                                  /* Tensor plano CHW com valores em [0, 127] */
    int ground_truth;                                           /* Label ground-truth */
    int expected_prediction;                                    /* Predicao esperada do motor Int8 */
    int32_t expected_spikes[SPIKEVISION_NUM_CLASSES];           /* Contagem de spikes por classe */
    int32_t expected_mem[SPIKEVISION_NUM_CLASSES];              /* Potencial final de membrana no timestep 8 */
}} SNNTestCase;

/* Arrays de dados de imagem das amostras */
{sample_decls_body}

/* Tabela de Casos de Teste para iteracao de teste automatizado */
static const SNNTestCase TEST_SUITE[TEST_SAMPLE_COUNT] = {{
{test_cases_body}
}};

#ifdef __cplusplus
}}
#endif
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header_content)

    logger.info(f"Cabeçalho de amostras de teste salvo em: {output_path}")


def analyze_memory_footprint(quant_params: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calcula com exatidão a alocação de memória (Flash e SRAM) no microcontrolador ESP32-S3.
    """
    p1 = quant_params["layer1"]
    p2 = quant_params["layer2"]
    p3 = quant_params["layer3"]
    p4 = quant_params["layer4"]

    w_bytes = (
        p1["w_q"].numel() * 1 +
        p2["w_q"].numel() * 1 +
        p3["w_q"].numel() * 1 +
        p4["w_q"].numel() * 1
    )
    b_bytes = (
        p1["b_q"].numel() * 4 +
        p2["b_q"].numel() * 4 +
        p3["b_q"].numel() * 4 +
        p4["b_q"].numel() * 4
    )
    flash_bytes = w_bytes + b_bytes

    num_classes = p4["b_q"].numel()

    # Estados dos neurônios (Membranas em int32_t)
    mem_l1_bytes = (16 * 16 * 16) * 4 # 16 KB
    mem_l2_bytes = (32 * 8 * 8) * 4   # 8 KB
    mem_l3_bytes = (32 * 4 * 4) * 4   # 2 KB
    mem_l4_bytes = num_classes * 4
    total_mem_bytes = mem_l1_bytes + mem_l2_bytes + mem_l3_bytes + mem_l4_bytes

    # Buffers de ativação/spikes intermediários (em uint8_t)
    spk_l1_bytes = (16 * 16 * 16) * 1 # 4 KB
    spk_l2_bytes = (32 * 8 * 8) * 1   # 2 KB
    spk_l3_bytes = (32 * 4 * 4) * 1   # 512 bytes
    spk_l4_bytes = num_classes * 1
    total_spk_bytes = spk_l1_bytes + spk_l2_bytes + spk_l3_bytes + spk_l4_bytes

    # Buffer de convolução estática da camada 1 (p1 pré-calculado)
    p1_buf_bytes = (16 * 16 * 16) * 4 # 16 KB

    total_sram_bytes = total_mem_bytes + total_spk_bytes + p1_buf_bytes

    return {
        "flash_weights_bytes": w_bytes,
        "flash_bias_bytes": b_bytes,
        "total_flash_model_bytes": flash_bytes,
        "total_flash_model_kb": flash_bytes / 1024.0,
        "sram_membrane_potentials_bytes": total_mem_bytes,
        "sram_spike_buffers_bytes": total_spk_bytes,
        "sram_layer1_preconv_buffer_bytes": p1_buf_bytes,
        "total_peak_sram_bytes": total_sram_bytes,
        "total_peak_sram_kb": total_sram_bytes / 1024.0,
        "esp32s3_sram_total_kb": 512.0,
        "esp32s3_sram_utilization_percent": (total_sram_bytes / (512.0 * 1024.0)) * 100.0,
    }


def validate_checkpoint_encoding(
    config: Dict[str, Any],
    allow_legacy: bool = False,
    force_mismatch: bool = False,
) -> str:
    """
    Valida o campo `encoding` do checkpoint contra o encoding 'direct' do firmware.

    - Se ausente (legado): emite aviso e assume 'direct' (ou suprime o aviso se allow_legacy=True).
    - Se presente e != 'direct': aborta com SystemExit(1), salvo se force_mismatch=True.
    """
    ckpt_encoding = config.get("encoding")
    if ckpt_encoding is None:
        if not allow_legacy:
            logger.warning(
                "Checkpoint legado sem campo 'encoding' em config; assumindo encoding='direct'. "
                "(Passe --allow-legacy-checkpoint para suprimir este aviso.)"
            )
        config["encoding"] = "direct"
        return "direct"

    if ckpt_encoding != "direct":
        if force_mismatch:
            logger.warning(
                f"Checkpoint treinado com encoding='{ckpt_encoding}', diferente de 'direct'. "
                "Prosseguindo por --force-encoding-mismatch (risco de queda severa de acurácia)."
            )
            return ckpt_encoding
        logger.error(
            f"Incompatibilidade de encoding: checkpoint usa encoding='{ckpt_encoding}', "
            "mas o simulador Int8 e o firmware ESP32-S3 esperam encoding='direct'. "
            "Re-treine com --encoding direct ou passe --force-encoding-mismatch."
        )
        sys.exit(1)

    return ckpt_encoding


def main():
    parser = argparse.ArgumentParser(
        description="SpikeVision-S3: Quantização Int8 e Exportação C/C++"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="training/checkpoints/best_spikevision_csnn.pth",
        help="Caminho para o checkpoint PyTorch treinado",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./training/data",
        help="Diretório onde o CIFAR-10 está armazenado",
    )
    parser.add_argument(
        "--output-weights",
        type=str,
        default="firmware/include/weights.h",
        help="Caminho para o arquivo C/C++ de pesos exportados",
    )
    parser.add_argument(
        "--output-samples",
        type=str,
        default="firmware/test_samples/test_samples.h",
        help="Caminho para o arquivo C/C++ de amostras de teste",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="training/checkpoints/quantization_report.json",
        help="Caminho para o relatório de quantização em JSON",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size para avaliação no conjunto de teste",
    )
    parser.add_argument(
        "--test-samples-per-class",
        type=int,
        default=5,
        help="Número de amostras reais por classe exportadas para validação em firmware",
    )
    parser.add_argument(
        "--allow-legacy-checkpoint",
        action="store_true",
        help="Permite checkpoints legados sem o campo 'encoding' (assume 'direct' sem aviso)",
    )
    parser.add_argument(
        "--quant-mode",
        type=str,
        default="per-channel",
        choices=["per-channel", "per-tensor"],
        help="Modo de quantização simétrica Int8 ('per-channel' axis=0 padrão ou 'per-tensor')",
    )
    parser.add_argument(
        "--force-encoding-mismatch",
        action="store_true",
        help="Força exportação mesmo se o checkpoint não tiver sido treinado com encoding='direct'",
    )
    parser.add_argument(
        "--early-exit",
        action="store_true",
        help="Executa avaliação comparativa de Early-Exit Temporal (full-T vs adaptativo)",
    )
    parser.add_argument(
        "--early-exit-min-t",
        type=int,
        default=2,
        help="Número mínimo de timesteps antes de avaliar early-exit (padrão: 2)",
    )
    parser.add_argument(
        "--early-exit-margin",
        type=int,
        default=0,
        help="Margem de spikes (top1 - top2) para early-exit; 0 ativa modo lossless matemático",
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("🧠 SpikeVision-S3: Fase 3 - Quantização Int8 e Exportação C/C++")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Dispositivo de execução: {device}")

    # 1. Carregar Checkpoint
    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint não encontrado em: {args.checkpoint}")
        sys.exit(1)

    logger.info(f"Carregando checkpoint de: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    config = ckpt.get("config", {})
    validate_checkpoint_encoding(
        config,
        allow_legacy=args.allow_legacy_checkpoint,
        force_mismatch=args.force_encoding_mismatch,
    )
    trained_metrics = ckpt.get("metrics", {})
    epoch = ckpt.get("epoch", 0)

    print(f"\n📋 Metadados do Treinamento:")
    print(f"  • Época: {epoch}")
    print(f"  • Acurácia Treinada: {ckpt.get('best_acc', 0.0) * 100:.2f}%")
    print(f"  • Configuração: {config}")
    print(f"  • F1-Score: {trained_metrics.get('cat_f1', 0.0) * 100:.2f}%")
    print(f"  • Esparsidade: {trained_metrics.get('overall_sparsity', 0.0) * 100:.2f}%\n")

    num_classes = config.get(
        "num_classes",
        10 if ("fc.weight" in ckpt.get("model_state_dict", {}) and ckpt["model_state_dict"]["fc.weight"].shape[0] == 10) else 2
    )
    config["num_classes"] = num_classes

    # 2. Inicializar Modelo PyTorch Float32
    model = SpikeVisionCSNN(
        in_channels=config.get("in_channels", 3),
        num_classes=num_classes,
        beta=config.get("beta", 0.9),
        slope=config.get("slope", 25.0),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 3. Fusão Conv2d + BatchNorm2d
    logger.info("Executando fusão analítica Conv2d + BatchNorm2d (Camadas 1, 2 e 3)...")
    w1_f, b1_f = fuse_conv_bn(model.conv1, model.bn1)
    w2_f, b2_f = fuse_conv_bn(model.conv2, model.bn2)
    w3_f, b3_f = fuse_conv_bn(model.conv3, model.bn3)
    w4_f = model.fc.weight.data.clone()
    b4_f = torch.zeros(model.num_classes, device=device)

    # 4. Quantização Int8 e Cálculo de Escalas
    per_channel = args.quant_mode == "per-channel"
    logger.info(
        f"Calculando escalas simétricas ({args.quant_mode}) e quantizando pesos em Int8 / biases em Int32..."
    )
    q1 = quantize_layer(w1_f.cpu(), b1_f.cpu(), s_in=127.0, v_th=1.0, per_channel=per_channel)
    q2 = quantize_layer(w2_f.cpu(), b2_f.cpu(), s_in=1.0, v_th=1.0, per_channel=per_channel)
    q3 = quantize_layer(w3_f.cpu(), b3_f.cpu(), s_in=1.0, v_th=1.0, per_channel=per_channel)
    q4 = quantize_layer(w4_f.cpu(), b4_f.cpu(), s_in=1.0, v_th=1.0, per_channel=per_channel)

    quant_params = {
        "layer1": q1,
        "layer2": q2,
        "layer3": q3,
        "layer4": q4,
    }

    print(f"\n📊 Parâmetros de Quantização por Camada ({args.quant_mode}):")
    for name, q in quant_params.items():
        print(f"  [{name.upper()}]:")
        if q.get("per_channel", False):
            w_max_t = q["w_max"]
            s_w_t = q["s_w"]
            s_tot_t = q["s_total"]
            vth_t = q["vth_q"]
            print(f"    - W_max:     [{w_max_t.min().item():.6f}, {w_max_t.max().item():.6f}]")
            print(f"    - S_w:       [{s_w_t.min().item():.4f}, {s_w_t.max().item():.4f}]")
            print(f"    - S_in:      {q['s_in']:.1f}")
            print(f"    - S_total:   [{s_tot_t.min().item():.4f}, {s_tot_t.max().item():.4f}]")
            print(f"    - V_th Int:  [{vth_t.min().item()}, {vth_t.max().item()}] (Float: {q['vth_float']:.1f})")
        else:
            print(f"    - W_max:     {q['w_max']:.6f}")
            print(f"    - S_w:       {q['s_w']:.4f}")
            print(f"    - S_in:      {q['s_in']:.1f}")
            print(f"    - S_total:   {q['s_total']:.4f}")
            print(f"    - V_th Int:  {q['vth_q']} (Float: {q['vth_float']:.1f})")
        print(f"    - W_q shape: {list(q['w_q'].shape)} ({q['w_q'].numel()} bytes)")
        print(f"    - B_q shape: {list(q['b_q'].shape)} ({q['b_q'].numel() * 4} bytes)")

    # 5. Instanciar Simulador Bit-Accurate Int8
    int8_sim = BitAccurateInt8CSNN(quant_params, num_steps=config.get("num_steps", 8))

    # 6. Carregar Dataset de Teste (Multi-classe ou Binário)
    if num_classes == 10:
        logger.info("Carregando amostras do conjunto de teste completo do CIFAR-10 (10 classes)...")
        _, test_loader = get_cifar10_multiclass_dataloaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=2,
            augment=False,
            download=False,
        )
    else:
        logger.info("Carregando 2.000 amostras do conjunto de teste binário do CIFAR-10...")
        _, test_loader = get_cifar10_cat_dataloaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=2,
            in_channels=config.get("in_channels", 3),
            download=False,
            seed=42,
            smoke_test=False,
        )

    # 7. Avaliação e Comparação Float32 vs Int8
    metrics_f, metrics_i, agreement = evaluate_comparison(
        float_model=model,
        int8_sim=int8_sim,
        test_loader=test_loader,
        device=device,
        num_steps=config.get("num_steps", 8),
    )

    print("\n" + "=" * 70)
    print(f"📈 RESULTADOS DA VALIDAÇÃO ({len(test_loader.dataset):,} Amostras do Conjunto de Teste - {num_classes} Classes)")
    print("=" * 70)
    print(f"{'Métrica':<25} | {'Float32 Original':<20} | {'Int8 Quantizado':<20} | {'Diferença':<12}")
    print("-" * 70)
    acc_diff = (metrics_i["accuracy"] - metrics_f["accuracy"]) * 100.0
    f1_diff = (metrics_i["f1_score"] - metrics_f["f1_score"]) * 100.0
    rec_diff = (metrics_i["recall"] - metrics_f["recall"]) * 100.0
    prec_diff = (metrics_i["precision"] - metrics_f["precision"]) * 100.0
    sp_diff = (metrics_i["mean_sparsity"] - metrics_f["mean_sparsity"]) * 100.0

    f1_label = "F1-Score (Macro)" if num_classes == 10 else "F1-Score (Gatos)"
    prec_label = "Precisão (Macro)" if num_classes == 10 else "Precisão (Gatos)"
    rec_label = "Recall (Macro)" if num_classes == 10 else "Recall (Gatos)"

    print(f"{'Acurácia Total':<25} | {metrics_f['accuracy']*100:6.2f}%              | {metrics_i['accuracy']*100:6.2f}%              | {acc_diff:+6.2f}%")
    print(f"{f1_label:<25} | {metrics_f['f1_score']*100:6.2f}%              | {metrics_i['f1_score']*100:6.2f}%              | {f1_diff:+6.2f}%")
    print(f"{prec_label:<25} | {metrics_f['precision']*100:6.2f}%              | {metrics_i['precision']*100:6.2f}%              | {prec_diff:+6.2f}%")
    print(f"{rec_label:<25} | {metrics_f['recall']*100:6.2f}%              | {metrics_i['recall']*100:6.2f}%              | {rec_diff:+6.2f}%")
    print(f"{'Esparsidade Média':<25} | {metrics_f['mean_sparsity']*100:6.2f}%              | {metrics_i['mean_sparsity']*100:6.2f}%              | {sp_diff:+6.2f}%")
    print(f"{'Concordância (Agreement)':<25} | {'100.00%':<20} | {agreement*100:6.2f}%              | -")
    print("-" * 70)

    # 8. Análise de Memória no ESP32-S3
    mem_analysis = analyze_memory_footprint(quant_params)
    print("💾 ANÁLISE DE MEMÓRIA NO ESP32-S3:")
    print(f"  • Memória Flash (Pesos + Biases): {mem_analysis['total_flash_model_kb']:.2f} KB ({mem_analysis['total_flash_model_bytes']} bytes)")
    print(f"  • SRAM Pico Requerida (Membranas + Buffers): {mem_analysis['total_peak_sram_kb']:.2f} KB ({mem_analysis['total_peak_sram_bytes']} bytes)")
    print(f"  • Ocupação da SRAM Interna (512 KB): {mem_analysis['esp32s3_sram_utilization_percent']:.2f}% (Folga de {512.0 - mem_analysis['total_peak_sram_kb']:.1f} KB)\n")

    # 9. Exportar Cabeçalhos C/C++
    export_weights_h(quant_params, args.output_weights, config)
    samples_per_cls = 1 if num_classes == 10 else args.test_samples_per_class
    export_test_samples_h(test_loader, int8_sim, args.output_samples, samples_per_class=samples_per_cls)

    # 9.5 Avaliação opcional de Early-Exit Temporal
    early_exit_summary = None
    if args.early_exit:
        num_steps = config.get("num_steps", 8)
        ee_sim = BitAccurateInt8CSNN(
            quant_params,
            num_steps=num_steps,
            early_exit_enabled=True,
            early_exit_min_t=args.early_exit_min_t,
            early_exit_margin=args.early_exit_margin,
        )
        full_preds, ee_preds, ee_targets = [], [], []
        full_sops_total = 0
        ee_sops_total = 0
        ee_t_total = 0
        total_samples = 0

        for imgs, targets in test_loader:
            imgs_cpu = imgs.cpu()
            p_full, _, _, _ = int8_sim.forward_batch(imgs_cpu)
            p_ee, _, _, _ = ee_sim.forward_batch(imgs_cpu)
            full_preds.append(p_full)
            ee_preds.append(p_ee)
            ee_targets.append(targets.cpu())
            full_sops_total += int(int8_sim.last_sample_sops.sum().item())
            ee_sops_total += int(ee_sim.last_sample_sops.sum().item())
            ee_t_total += int(ee_sim.last_timesteps_executed.sum().item())
            total_samples += imgs_cpu.size(0)

        full_preds_np = torch.cat(full_preds).numpy()
        ee_preds_np = torch.cat(ee_preds).numpy()
        ee_targets_np = torch.cat(ee_targets).numpy()

        avg_t = ee_t_total / max(total_samples, 1)
        t_red_pct = 100.0 * (1.0 - avg_t / max(num_steps, 1))
        sop_red_pct = 100.0 * (1.0 - ee_sops_total / max(full_sops_total, 1))
        ee_agreement = float(np.mean(full_preds_np == ee_preds_np))
        ee_accuracy = float(np.mean(ee_preds_np == ee_targets_np))

        early_exit_summary = {
            "enabled": True,
            "min_timesteps": max(1, args.early_exit_min_t),
            "margin": args.early_exit_margin,
            "mode": "lossless" if args.early_exit_margin <= 0 else f"margin>={args.early_exit_margin}",
            "avg_timesteps": avg_t,
            "max_timesteps": num_steps,
            "timestep_reduction_percent": t_red_pct,
            "full_t_avg_sops": full_sops_total / max(total_samples, 1),
            "early_exit_avg_sops": ee_sops_total / max(total_samples, 1),
            "sop_reduction_percent": sop_red_pct,
            "agreement_with_full_t": ee_agreement,
            "accuracy": ee_accuracy,
        }

        print("⚡ AVALIAÇÃO EARLY-EXIT TEMPORAL (Full-T vs Adaptativo):")
        print(f"  • Modo:                    {early_exit_summary['mode']} (min_t={early_exit_summary['min_timesteps']})")
        print(f"  • t_exit médio:            {avg_t:.2f} / {num_steps} (Economia de {t_red_pct:.2f}% timesteps)")
        print(f"  • SOPs médias:             {early_exit_summary['early_exit_avg_sops']:.0f} vs {early_exit_summary['full_t_avg_sops']:.0f} (Economia de {sop_red_pct:.2f}%)")
        print(f"  • Concordância vs Full-T:  {ee_agreement*100:.2f}%")
        print(f"  • Acurácia Early-Exit:     {ee_accuracy*100:.2f}%\n")

    # 10. Salvar Relatório em JSON
    report = {
        "model_architecture": "SpikeVisionCSNN",
        "training_checkpoint": args.checkpoint,
        "epoch": epoch,
        "config": config,
        "quantization_format": {
            "quant_mode": args.quant_mode,
            "weights": f"int8_t (symmetric {args.quant_mode})",
            "biases": "int32_t (scaled by s_total)",
            "thresholds": "int32_t (scaled by s_total)",
            "input_scale": 127,
            "fixed_point_beta": {
                "beta_num": BETA_NUM,
                "beta_shift": BETA_SHIFT,
                "effective_beta": BETA_NUM / (1 << BETA_SHIFT),
            },
        },
        "layers": {
            k: {
                "per_channel": bool(q.get("per_channel", False)),
                "w_max": q["w_max"].tolist() if isinstance(q["w_max"], torch.Tensor) else q["w_max"],
                "scale_weight": q["s_w"].tolist() if isinstance(q["s_w"], torch.Tensor) else q["s_w"],
                "scale_input": q["s_in"],
                "scale_total": q["s_total"].tolist() if isinstance(q["s_total"], torch.Tensor) else q["s_total"],
                "vth_quantized": q["vth_q"].tolist() if isinstance(q["vth_q"], torch.Tensor) else q["vth_q"],
                "weight_shape": list(q["w_q"].shape),
                "bias_shape": list(q["b_q"].shape),
            }
            for k, q in quant_params.items()
        },
        "metrics_float32": metrics_f,
        "metrics_int8": metrics_i,
        "agreement_rate": agreement,
        "memory_analysis": mem_analysis,
    }
    if early_exit_summary is not None:
        report["early_exit"] = early_exit_summary

    os.makedirs(os.path.dirname(args.output_report), exist_ok=True)
    with open(args.output_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Relatório de quantização salvo em: {args.output_report}")

    print("=" * 70)
    print("✅ FASE 3 CONCLUÍDA COM SUCESSO!")
    print("=" * 70)


if __name__ == "__main__":
    main()
