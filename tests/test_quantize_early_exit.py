import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.models.csnn import SpikeVisionCSNN  # noqa: E402
from training.quantize_export import (  # noqa: E402
    BitAccurateInt8CSNN,
    fuse_conv_bn,
    quantize_layer,
    validate_checkpoint_encoding,
)


def _quantize_model(model: SpikeVisionCSNN, per_channel: bool) -> dict:
    model.eval()
    w1, b1 = fuse_conv_bn(model.conv1, model.bn1)
    w2, b2 = fuse_conv_bn(model.conv2, model.bn2)
    w3, b3 = fuse_conv_bn(model.conv3, model.bn3)
    w4 = model.fc.weight.detach()
    b4 = torch.zeros(w4.shape[0], dtype=torch.float32)

    return {
        "layer1": quantize_layer(w1.detach(), b1.detach(), s_in=127.0, v_th=1.0, per_channel=per_channel),
        "layer2": quantize_layer(w2.detach(), b2.detach(), s_in=1.0, v_th=1.0, per_channel=per_channel),
        "layer3": quantize_layer(w3.detach(), b3.detach(), s_in=1.0, v_th=1.0, per_channel=per_channel),
        "layer4": quantize_layer(w4, b4, s_in=1.0, v_th=1.0, per_channel=per_channel),
    }


class TestQuantizeAndEarlyExit(unittest.TestCase):
    def test_fuse_conv_bn_equivalence(self):
        torch.manual_seed(42)
        conv = nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False)
        bn = nn.BatchNorm2d(8)
        # Populate non-trivial running stats
        bn.train()
        _ = bn(torch.randn(4, 8, 16, 16) * 2.5 + 1.0)
        bn.eval()
        conv.eval()

        x = torch.randn(2, 3, 16, 16)
        with torch.no_grad():
            y_ref = bn(conv(x))
            w_fused, b_fused = fuse_conv_bn(conv, bn)
            y_fused = nn.functional.conv2d(x, w_fused, b_fused, padding=1)

        self.assertTrue(torch.allclose(y_ref, y_fused, atol=1e-5))

    def test_validate_checkpoint_encoding(self):
        cfg_legacy = {}
        self.assertEqual(validate_checkpoint_encoding(cfg_legacy, allow_legacy=False), "direct")
        self.assertEqual(cfg_legacy["encoding"], "direct")

        cfg_legacy2 = {}
        self.assertEqual(validate_checkpoint_encoding(cfg_legacy2, allow_legacy=True), "direct")

        self.assertEqual(validate_checkpoint_encoding({"encoding": "direct"}), "direct")

        with self.assertRaises(SystemExit) as ctx:
            validate_checkpoint_encoding({"encoding": "rate"}, force_mismatch=False)
        self.assertEqual(ctx.exception.code, 1)

        self.assertEqual(
            validate_checkpoint_encoding({"encoding": "rate"}, force_mismatch=True),
            "rate",
        )

    def test_bit_accurate_sim_per_tensor_and_per_channel(self):
        torch.manual_seed(7)
        model = SpikeVisionCSNN(in_channels=3, num_classes=10)
        imgs = torch.rand(4, 3, 32, 32)

        for per_channel in (False, True):
            q_params = _quantize_model(model, per_channel=per_channel)
            sim = BitAccurateInt8CSNN(q_params, num_steps=8)
            preds, spk_counts, final_mem, stats = sim.forward_batch(imgs)
            self.assertEqual(preds.shape, (4,))
            self.assertEqual(spk_counts.shape, (4, 10))
            self.assertEqual(final_mem.shape, (4, 10))
            self.assertTrue(((preds >= 0) & (preds < 10)).all())
            self.assertIn("overall_sparsity", stats)

    def test_early_exit_lossless_and_margin(self):
        # Construct deterministic quant_params where class 0 fires at every timestep
        # and classes 1..9 never fire, guaranteeing early exit at min_t.
        q_params = {
            "layer1": {
                "w_q": torch.zeros((16, 3, 3, 3), dtype=torch.int8),
                "b_q": torch.zeros((16,), dtype=torch.int32),
                "vth_q": 1000,
            },
            "layer2": {
                "w_q": torch.zeros((32, 16, 3, 3), dtype=torch.int8),
                "b_q": torch.zeros((32,), dtype=torch.int32),
                "vth_q": 1000,
            },
            "layer3": {
                "w_q": torch.zeros((32, 32, 3, 3), dtype=torch.int8),
                "b_q": torch.zeros((32,), dtype=torch.int32),
                "vth_q": 1000,
            },
            "layer4": {
                "w_q": torch.zeros((10, 512), dtype=torch.int8),
                "b_q": torch.tensor([2000] + [0] * 9, dtype=torch.int32),
                "vth_q": 1000,
            },
        }
        img = torch.zeros((1, 3, 32, 32), dtype=torch.float32)

        sim_full = BitAccurateInt8CSNN(q_params, num_steps=8, early_exit_enabled=False)
        preds_full, _, _, _ = sim_full.forward_batch(img)

        # Lossless (margin=0, min_t=2): at t=5, top1=5, top2=0, remaining=3 -> 5-0 > 3 -> exits at t=5
        sim_lossless = BitAccurateInt8CSNN(
            q_params, num_steps=8, early_exit_enabled=True, early_exit_min_t=2, early_exit_margin=0
        )
        preds_lossless, _, _, _ = sim_lossless.forward_batch(img)
        self.assertEqual(int(preds_lossless[0].item()), int(preds_full[0].item()))
        self.assertEqual(sim_lossless.last_timesteps_executed, 5)

        # Margin=2, min_t=2: at t=2, top1=2, top2=0 -> 2-0 >= 2 -> exits at t=2
        sim_margin = BitAccurateInt8CSNN(
            q_params, num_steps=8, early_exit_enabled=True, early_exit_min_t=2, early_exit_margin=2
        )
        preds_margin, _, _, _ = sim_margin.forward_batch(img)
        self.assertEqual(int(preds_margin[0].item()), int(preds_full[0].item()))
        self.assertEqual(sim_margin.last_timesteps_executed, 2)


if __name__ == "__main__":
    unittest.main()
