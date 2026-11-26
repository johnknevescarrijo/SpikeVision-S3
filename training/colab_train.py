#!/usr/bin/env python3
"""
================================================================================
SpikeVision-S3: Treinamento CSNN no Google Colab
================================================================================
Instruções de Uso no Google Colab:
1. Abra um novo notebook no Google Colab (https://colab.research.google.com).
2. Vá no menu: 'Ambiente de execução' > 'Alterar tipo de ambiente de execução'
   e selecione 'T4 GPU' (Hardware accelerator: GPU).
3. Clone o repositório e execute este script:
   !git clone https://github.com/johnknevescarrijo/SpikeVision-S3.git
   %cd SpikeVision-S3
   !python3 training/colab_train.py
4. O script irá:
   - Verificar e instalar automaticamente `snntorch` caso ausente;
   - Reutilizar os módulos canônicos em `training/` (sem código duplicado);
   - Treinar a CSNN (`SpikeVisionCSNN`) no CIFAR-10 (10 classes ou binário);
   - Salvar o melhor checkpoint 'best_spikevision_csnn.pth' e o histórico JSON;
   - Acionar o download automático do modelo no Google Colab!
================================================================================
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _ensure_repo_root_in_path() -> Path:
    """Garante que o diretório raiz do repositório esteja no sys.path."""
    candidates = []
    if "__file__" in globals():
        candidates.append(Path(__file__).resolve().parent.parent)
    cwd = Path.cwd().resolve()
    candidates.extend([cwd, cwd / "SpikeVision-S3"])

    for candidate in candidates:
        if (candidate / "training" / "models" / "csnn.py").is_file():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            return candidate
    return cwd


REPO_ROOT = _ensure_repo_root_in_path()

# Instalação automática de dependências caso ausentes no ambiente Colab
try:
    import snntorch  # noqa: F401
except ImportError:
    print("⚡ Instalando snnTorch para o ambiente Colab...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "snntorch"])
    import snntorch  # noqa: F401

import torch
import torch.optim as optim

from training.datasets.cifar10_cat import (  # noqa: E402
    CIFAR10_CAT_LABEL,
    CIFAR10_CLASS_NAMES,
    BinaryCatCIFAR10,
    encode_spikes,
    get_cifar10_multiclass_dataloaders,
    get_dataloaders as get_binary_dataloaders,
)
from training.models.csnn import SpikeVisionCSNN, count_parameters, print_model_summary  # noqa: E402
from training.models.qat import QATConv2d, QATLinear, fake_quantize_int8_ste  # noqa: E402
from training.quantize_export import (  # noqa: E402
    BitAccurateInt8CSNN,
    export_weights_h,
    fuse_conv_bn,
)
from training.train import (  # noqa: E402
    build_loss_fn,
    compute_multiclass_metrics,
    compute_temporal_loss,
    evaluate,
    set_seed,
    train_one_epoch,
)

__all__ = [
    "CONFIG",
    "REPO_ROOT",
    "CIFAR10_CAT_LABEL",
    "CIFAR10_CLASS_NAMES",
    "BinaryCatCIFAR10",
    "encode_spikes",
    "get_cifar10_multiclass_dataloaders",
    "get_binary_dataloaders",
    "SpikeVisionCSNN",
    "count_parameters",
    "print_model_summary",
    "QATConv2d",
    "QATLinear",
    "fake_quantize_int8_ste",
    "BitAccurateInt8CSNN",
    "export_weights_h",
    "fuse_conv_bn",
    "build_loss_fn",
    "compute_multiclass_metrics",
    "compute_temporal_loss",
    "evaluate",
    "set_seed",
    "train_one_epoch",
    "main",
]


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
    "loss_type": "ce_rate",  # 'ce_rate', 'ce_mem' ou 'mse_count'
    "sparsity_weight": 0.0,  # Peso da regularização de esparsidade (firing rate)
    "target_rate": None,     # Taxa alvo opcional (None = L1 em direção a 0)
    "qat": False,            # Quantization-Aware Training (STE Int8)
    "qat_per_channel": True, # True = per-channel (axis=0), False = per-tensor
    "in_channels": 3,        # 3 para RGB (ou 1 para escala de cinza)
    "num_classes": 10,       # 10 classes completas do CIFAR-10 (ou 2 para Gato vs Não-Gato)

    # Sistema e Reprodutibilidade
    "seed": 42,
    "num_workers": 2,
    "data_dir": "./data",
    "checkpoint_path": "best_spikevision_csnn.pth",
    "history_path": "training_history.json",
}


def main():
    set_seed(CONFIG["seed"])

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

    # Carregar Dados usando os loaders canônicos
    print(
        f"\nCarregando CIFAR-10 ({CONFIG['num_classes']} classes, "
        f"Batch Size: {CONFIG['batch_size']}, Workers: {CONFIG['num_workers']})..."
    )
    if CONFIG["num_classes"] == 10:
        train_loader, test_loader = get_cifar10_multiclass_dataloaders(
            data_dir=CONFIG["data_dir"],
            batch_size=CONFIG["batch_size"],
            num_workers=CONFIG["num_workers"],
            augment=True,
        )
    else:
        train_loader, test_loader = get_binary_dataloaders(
            data_dir=CONFIG["data_dir"],
            batch_size=CONFIG["batch_size"],
            num_workers=CONFIG["num_workers"],
            in_channels=CONFIG["in_channels"],
            seed=CONFIG["seed"],
        )
    print(f"✓ Amostras de Treino: {len(train_loader.dataset):,} | Amostras de Teste: {len(test_loader.dataset):,}")

    # Inicializar Modelo SNN canônico
    model = SpikeVisionCSNN(
        in_channels=CONFIG["in_channels"],
        num_classes=CONFIG["num_classes"],
        beta=CONFIG["beta"],
        slope=CONFIG["slope"],
        qat=CONFIG["qat"],
        qat_per_channel=CONFIG["qat_per_channel"],
    ).to(device)

    print_model_summary(model)

    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"], eta_min=1e-5)
    criterion = build_loss_fn(CONFIG["loss_type"])

    best_acc = 0.0
    best_checkpoint_path = CONFIG["checkpoint_path"]
    history = []

    print("\nIniciando treinamento SNN:")
    print(
        f"Épocas: {CONFIG['epochs']} | Classes: {CONFIG['num_classes']} | Timesteps (T): {CONFIG['num_steps']} | "
        f"Codificação: {CONFIG['encoding']} | Loss: {CONFIG['loss_type']} | "
        f"Sparsity Weight: {CONFIG['sparsity_weight']} | Beta LIF: {CONFIG['beta']}\n"
    )

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
            loss_type=CONFIG["loss_type"],
            sparsity_weight=CONFIG["sparsity_weight"],
            target_rate=CONFIG["target_rate"],
        )

        test_metrics = evaluate(
            model=model,
            data_loader=test_loader,
            criterion=criterion,
            device=device,
            num_steps=CONFIG["num_steps"],
            encoding=CONFIG["encoding"],
            loss_type=CONFIG["loss_type"],
            num_classes=CONFIG["num_classes"],
        )

        scheduler.step()
        epoch_time = time.time() - epoch_start

        test_acc = test_metrics["accuracy"]
        test_loss = test_metrics["loss"]
        macro_f1 = test_metrics.get("macro_f1", 0.0)
        macro_p = test_metrics.get("macro_precision", 0.0)
        macro_r = test_metrics.get("macro_recall", 0.0)
        overall_fr = test_metrics["overall_firing_rate"]
        sparsity = test_metrics["overall_sparsity"]
        train_sparsity = 1.0 - train_fr

        print(
            f"Época [{epoch:02d}/{CONFIG['epochs']:02d}] ({epoch_time:4.1f}s) | "
            f"Treino Loss: {train_loss:.4f} Acc: {train_acc*100:5.2f}% Esp: {train_sparsity*100:5.1f}% | "
            f"Teste Loss: {test_loss:.4f} Acc: {test_acc*100:5.2f}% | "
            f"Macro-F1: {macro_f1:.3f} (P: {macro_p:.2f}, R: {macro_r:.2f}) | "
            f"Esparsidade: {sparsity*100:5.1f}%"
        )

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_firing_rate": train_fr,
            "train_sparsity": train_sparsity,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_macro_f1": macro_f1,
            "test_macro_precision": macro_p,
            "test_macro_recall": macro_r,
            "test_per_class_f1": test_metrics.get("per_class_f1", {}),
            "test_per_class_accuracy": test_metrics.get("per_class_accuracy", {}),
            "test_firing_rate": overall_fr,
            "test_sparsity": sparsity,
            "epoch_time_sec": epoch_time,
        }
        history.append(record)

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
                    "loss_type": CONFIG["loss_type"],
                    "sparsity_weight": CONFIG["sparsity_weight"],
                    "target_rate": CONFIG["target_rate"],
                    "qat": CONFIG["qat"],
                    "qat_per_channel": CONFIG["qat_per_channel"],
                },
                "metrics": test_metrics,
            }
            torch.save(checkpoint_payload, best_checkpoint_path)
            print(f"  --> ⭐ Novo melhor modelo salvo: {best_checkpoint_path} (Acurácia: {best_acc*100:.2f}%)")

    total_duration = time.time() - start_total_time

    with open(CONFIG["history_path"], "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 72)
    print("🏁 Treinamento Concluído com Sucesso!")
    print(f"Tempo Total:             {total_duration:.2f} segundos ({total_duration/60:.1f} min)")
    print(f"Melhor Acurácia Teste:   {best_acc * 100:.2f}%")
    print(f"Checkpoint Salvo:        {best_checkpoint_path}")
    print(f"Histórico Salvo:         {CONFIG['history_path']}")
    print("=" * 72)

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


if __name__ == "__main__":
    main()
