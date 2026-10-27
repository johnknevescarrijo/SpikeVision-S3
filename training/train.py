#!/usr/bin/env python3
"""
SpikeVision-S3: Script de Treinamento da SNN (CSNN) para o ESP32-S3.

Treina a rede neural convolucional pulsada (CSNN) com snnTorch e PyTorch
para classificação binária (Gato vs Não-Gato) no dataset CIFAR-10.

Recursos:
- Rate Coding e Direct Coding
- Monitoramento de Acurácia, F1-Score, Firing Rate e Esparsidade
- Estimativa de operações sinápticas (SOPs vs MACs)
- Checkpointing automático com metadados para quantização futura (Fase 3)
- Modo --smoke-test para validação rápida de integridade do pipeline
"""

import os
import sys
import time
import json
import random
import argparse
from typing import Dict, Any, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Ajuste de path para importação modular
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import snntorch.functional as SF
from training.datasets.cifar10_cat import (
    get_dataloaders,
    get_cifar10_cat_dataloaders,
    get_cifar10_multiclass_dataloaders,
    CIFAR10_CLASS_NAMES,
)
from training.models.csnn import SpikeVisionCSNN, count_parameters, print_model_summary


def compute_multiclass_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 10,
    class_names: Any = None,
) -> Dict[str, Any]:
    """
    Calcula acurácia top-1, métricas macro (precisão, recall, f1)
    e métricas por classe para classificação multi-classe.
    """
    if class_names is None:
        if num_classes == 10:
            class_names = CIFAR10_CLASS_NAMES
        else:
            class_names = [f"class_{i}" for i in range(num_classes)]

    if isinstance(preds, torch.Tensor):
        preds_np = preds.detach().cpu().numpy()
    else:
        preds_np = np.asarray(preds)

    if isinstance(targets, torch.Tensor):
        targets_np = targets.detach().cpu().numpy()
    else:
        targets_np = np.asarray(targets)

    total_samples = len(targets_np)
    correct = int((preds_np == targets_np).sum())
    accuracy = correct / max(total_samples, 1)

    per_class_f1 = {}
    per_class_accuracy = {}
    precisions = []
    recalls = []
    f1s = []

    for c in range(num_classes):
        c_name = class_names[c] if c < len(class_names) else f"class_{c}"
        tp = int(((preds_np == c) & (targets_np == c)).sum())
        fp = int(((preds_np == c) & (targets_np != c)).sum())
        fn = int(((preds_np != c) & (targets_np == c)).sum())
        class_total = int((targets_np == c).sum())

        prec = tp / max(tp + fp, 1e-9) if (tp + fp) > 0 else 0.0
        rec = tp / max(tp + fn, 1e-9) if (tp + fn) > 0 else 0.0
        f1 = 2.0 * (prec * rec) / max(prec + rec, 1e-9) if (prec + rec) > 0 else 0.0
        class_acc = tp / max(class_total, 1) if class_total > 0 else 0.0

        per_class_f1[c_name] = float(f1)
        per_class_accuracy[c_name] = float(class_acc)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    macro_precision = float(np.mean(precisions)) if precisions else 0.0
    macro_recall = float(np.mean(recalls)) if recalls else 0.0
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0

    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "per_class_accuracy": per_class_accuracy,
    }

    # Compatibilidade retroativa para binário / chave gato
    cat_idx = 3 if num_classes == 10 else 1
    if cat_idx < num_classes:
        metrics["cat_precision"] = float(precisions[cat_idx])
        metrics["cat_recall"] = float(recalls[cat_idx])
        metrics["cat_f1"] = float(f1s[cat_idx])

    return metrics


def set_seed(seed: int = 42):
    """Garante reprodutibilidade em todas as bibliotecas."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    num_steps: int,
    encoding: str,
    progress_bar: bool = True,
) -> Tuple[float, float, float]:
    """
    Executa uma época de treinamento da SNN.

    Returns:
        (avg_loss, train_accuracy, avg_firing_rate)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total_samples = 0
    firing_rates = []

    loader = tqdm(train_loader, desc="Treino", leave=False) if progress_bar else train_loader

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward temporal na SNN
        spk_out, mem_out, sparsity_dict = model(
            images, num_steps=num_steps, encoding=encoding, return_sparsity=True
        )

        # Loss calculada sobre a taxa/contagem de spikes emitidos
        loss = criterion(spk_out, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * targets.size(0)

        # Predição: classe com mais spikes acumulados no tempo
        spike_counts = spk_out.sum(dim=0)  # [B, num_classes]
        preds = spike_counts.argmax(dim=1)

        # Desempate com potencial de membrana final caso nenhum neurônio dispare
        no_spike_mask = (spike_counts.sum(dim=1) == 0)
        if no_spike_mask.any():
            preds[no_spike_mask] = mem_out[-1, no_spike_mask].argmax(dim=1)

        correct += (preds == targets).sum().item()
        total_samples += targets.size(0)
        firing_rates.append(sparsity_dict["overall_firing_rate"])

    avg_loss = total_loss / max(total_samples, 1)
    acc = correct / max(total_samples, 1)
    avg_firing_rate = float(np.mean(firing_rates)) if firing_rates else 0.0

    return avg_loss, acc, avg_firing_rate


def evaluate(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_steps: int,
    encoding: str,
    progress_bar: bool = False,
    num_classes: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Avalia a SNN em um conjunto de validação ou teste.

    Calcula Acurácia, Loss, Macro-Precision, Macro-Recall, Macro-F1,
    além da taxa de disparo médio e esparsidade sináptica.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0

    if num_classes is None:
        num_classes = getattr(model, "num_classes", 10)

    all_preds = []
    all_targets = []

    layer_firing_rates = {
        "firing_rate_layer1": [],
        "firing_rate_layer2": [],
        "firing_rate_layer3": [],
        "firing_rate_layer4": [],
        "overall_firing_rate": [],
        "overall_sparsity": [],
    }

    loader = tqdm(data_loader, desc="Avaliação", leave=False) if progress_bar else data_loader

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            spk_out, mem_out, sparsity_dict = model(
                images, num_steps=num_steps, encoding=encoding, return_sparsity=True
            )

            loss = criterion(spk_out, targets)
            total_loss += loss.item() * targets.size(0)

            # Predição por contagem temporal de spikes
            spike_counts = spk_out.sum(dim=0)
            preds = spike_counts.argmax(dim=1)

            # Desempate com potencial de membrana residual
            no_spike_mask = (spike_counts.sum(dim=1) == 0)
            if no_spike_mask.any():
                preds[no_spike_mask] = mem_out[-1, no_spike_mask].argmax(dim=1)

            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
            total_samples += targets.size(0)

            for k in layer_firing_rates:
                if k in sparsity_dict:
                    layer_firing_rates[k].append(sparsity_dict[k])

    avg_loss = total_loss / max(total_samples, 1)

    if all_preds:
        cat_preds = torch.cat(all_preds, dim=0)
        cat_targets = torch.cat(all_targets, dim=0)
        metrics = compute_multiclass_metrics(cat_preds, cat_targets, num_classes=num_classes)
    else:
        metrics = {
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "per_class_f1": {},
            "per_class_accuracy": {},
        }

    metrics["loss"] = avg_loss
    metrics["total_samples"] = total_samples

    for k, v in layer_firing_rates.items():
        metrics[k] = float(np.mean(v)) if v else 0.0

    return metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="SpikeVision-S3: Treinamento de CSNN para CIFAR-10 (ESP32-S3)"
    )
    parser.add_argument("--epochs", type=int, default=10, help="Número de épocas de treinamento")
    parser.add_argument("--batch-size", type=int, default=64, help="Tamanho do lote")
    parser.add_argument("--lr", type=float, default=1e-3, help="Taxa de aprendizado inicial")
    parser.add_argument("--num-steps", type=int, default=8, help="Número de timesteps T da simulação SNN")
    parser.add_argument("--beta", type=float, default=0.9, help="Fator de decaimento do neurônio Leaky LIF")
    parser.add_argument(
        "--encoding",
        type=str,
        default="rate",
        choices=["rate", "direct"],
        help="Método de codificação de entrada ('rate' ou 'direct')",
    )
    parser.add_argument("--in-channels", type=int, default=3, choices=[1, 3], help="Canais de entrada: 3 (RGB) ou 1 (Cinza)")
    parser.add_argument("--num-classes", type=int, default=10, help="Número de classes (10 para CIFAR-10 completo, 2 para binário)")
    parser.add_argument("--data-dir", type=str, default="./training/data", help="Diretório de dados CIFAR-10")
    parser.add_argument("--save-dir", type=str, default="./training/checkpoints", help="Diretório para checkpoints")
    parser.add_argument("--seed", type=int, default=42, help="Seed para reprodutibilidade")
    parser.add_argument("--num-workers", type=int, default=2, help="Workers do DataLoader")
    parser.add_argument("--no-download", action="store_true", help="Não baixar dataset CIFAR-10 se ausente")
    parser.add_argument("--smoke-test", action="store_true", help="Executar teste rápido de sanidade do pipeline")

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # Configuração do modo Smoke Test (validação rápida sem travar)
    if args.smoke_test:
        print("\n" + "=" * 60)
        print("🚀 Executando no modo SMOKE TEST (Validação Rápida)")
        print("=" * 60)
        args.epochs = 1
        args.batch_size = 16
        args.num_steps = 4
        args.num_workers = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo de execução: {device}")
    if torch.cuda.is_available():
        print(f"GPU detectada: {torch.cuda.get_device_name(0)}")

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)

    # 1. Carregar Dados
    if args.num_classes == 10:
        print(f"\nCarregando dataset CIFAR-10 completo (10 classes, Batch: {args.batch_size})...")
        train_loader, test_loader = get_cifar10_multiclass_dataloaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            augment=True,
            download=not args.no_download,
        )
        if args.smoke_test:
            from torch.utils.data import Subset
            train_loader = torch.utils.data.DataLoader(
                Subset(train_loader.dataset, list(range(min(len(train_loader.dataset), 64)))),
                batch_size=args.batch_size,
                shuffle=True,
            )
            test_loader = torch.utils.data.DataLoader(
                Subset(test_loader.dataset, list(range(min(len(test_loader.dataset), 32)))),
                batch_size=args.batch_size,
                shuffle=False,
            )
    else:
        print(f"\nCarregando dataset CIFAR-10 binário balanceado (Canais: {args.in_channels}, Batch: {args.batch_size})...")
        train_loader, test_loader = get_dataloaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            in_channels=args.in_channels,
            download=not args.no_download,
            seed=args.seed,
            smoke_test=args.smoke_test,
        )
    print(f"Amostras de Treino: {len(train_loader.dataset):,} | Amostras de Teste: {len(test_loader.dataset):,}")

    # 2. Inicializar Modelo
    model = SpikeVisionCSNN(
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        beta=args.beta,
        slope=25.0,
    ).to(device)

    # Sumário da Arquitetura
    print_model_summary(model)

    # 3. Otimizador e Função de Perda SNN
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    criterion = SF.ce_rate_loss()

    best_acc = 0.0
    best_checkpoint_path = os.path.join(args.save_dir, "best_spikevision_csnn.pth")
    history = []

    print("\nIniciando treinamento SNN:")
    print(f"Épocas: {args.epochs} | Classes: {args.num_classes} | Timesteps (T): {args.num_steps} | Codificação: {args.encoding} | Beta LIF: {args.beta}\n")

    start_total_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc, train_firing_rate = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_steps=args.num_steps,
            encoding=args.encoding,
            progress_bar=not args.smoke_test,
        )

        test_metrics = evaluate(
            model=model,
            data_loader=test_loader,
            criterion=criterion,
            device=device,
            num_steps=args.num_steps,
            encoding=args.encoding,
            progress_bar=False,
            num_classes=args.num_classes,
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

        print(
            f"Época [{epoch:02d}/{args.epochs:02d}] ({epoch_time:.1f}s) | "
            f"Treino Loss: {train_loss:.4f} Acc: {train_acc*100:5.2f}% | "
            f"Teste Loss: {test_loss:.4f} Acc: {test_acc*100:5.2f}% | "
            f"Macro-F1: {macro_f1:.3f} (P: {macro_p:.2f}, R: {macro_r:.2f}) | "
            f"Esparsidade: {sparsity*100:5.1f}% (Firing: {overall_fr*100:4.1f}%)"
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_firing_rate": train_firing_rate,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_macro_f1": macro_f1,
            "test_macro_precision": macro_p,
            "test_macro_recall": macro_r,
            "test_per_class_f1": test_metrics.get("per_class_f1", {}),
            "test_per_class_accuracy": test_metrics.get("per_class_accuracy", {}),
            "test_sparsity": sparsity,
            "epoch_time_sec": epoch_time,
        }
        history.append(epoch_record)

        # Salvar melhor checkpoint
        if test_acc >= best_acc:
            best_acc = test_acc
            checkpoint_payload = {
                "epoch": epoch,
                "best_acc": best_acc,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": {
                    "in_channels": args.in_channels,
                    "num_classes": args.num_classes,
                    "beta": args.beta,
                    "slope": 25.0,
                    "num_steps": args.num_steps,
                    "encoding": args.encoding,
                },
                "metrics": test_metrics,
            }
            torch.save(checkpoint_payload, best_checkpoint_path)
            print(f"  --> ⭐ Melhor modelo salvo em: {best_checkpoint_path} (Acurácia: {best_acc*100:.2f}%)")

    total_duration = time.time() - start_total_time

    # Salvar histórico de treinamento em JSON
    history_file = os.path.join(args.save_dir, "training_history.json")
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 60)
    print("Treinamento Concluído com Sucesso! 🏁")
    print(f"Tempo Total:             {total_duration:.2f} segundos")
    print(f"Melhor Acurácia Teste:   {best_acc * 100:.2f}%")
    print(f"Checkpoint Salvo:        {best_checkpoint_path}")
    print(f"Histórico Salvo:         {history_file}")
    print("=" * 60)

    # Validar integridade do checkpoint recém-salvo
    if os.path.exists(best_checkpoint_path):
        chk = torch.load(best_checkpoint_path, map_location="cpu")
        print(f"✓ Checkpoint verificado: Época {chk['epoch']} com {len(chk['model_state_dict'])} tensores de pesos.")


if __name__ == "__main__":
    main()
