import inspect
import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.models.csnn import SpikeVisionCSNN  # noqa: E402
from training.quantize_export import fuse_conv_bn  # noqa: E402
from training.train import build_loss_fn, compute_temporal_loss  # noqa: E402


class TestTrainingAndQAT(unittest.TestCase):
    def test_default_encoding_is_direct(self):
        sig = inspect.signature(SpikeVisionCSNN.forward)
        self.assertEqual(sig.parameters["encoding"].default, "direct")

        torch.manual_seed(0)
        model = SpikeVisionCSNN(in_channels=3, num_classes=10)
        x = torch.rand(2, 3, 32, 32)
        spk_default, mem_default, sp_default = model(x, num_steps=4, return_sparsity=True)
        spk_direct, mem_direct, _ = model(x, num_steps=4, encoding="direct", return_sparsity=True)
        self.assertEqual(spk_default.shape, (4, 2, 10))
        self.assertEqual(mem_default.shape, (4, 2, 10))
        self.assertTrue(torch.equal(spk_default, spk_direct))
        self.assertTrue(torch.allclose(mem_default, mem_direct))
        self.assertIn("overall_firing_rate", sp_default)

    def test_compute_firing_rate_loss_l1_and_target_rate(self):
        torch.manual_seed(1)
        model = SpikeVisionCSNN(in_channels=3, num_classes=10)
        x = torch.rand(2, 3, 32, 32)
        model(x, num_steps=4, encoding="direct")

        loss_l1 = model.compute_firing_rate_loss(target_rate=None)
        self.assertEqual(loss_l1.ndim, 0)
        self.assertGreaterEqual(float(loss_l1.item()), 0.0)

        loss_target = model.compute_firing_rate_loss(target_rate=0.2)
        self.assertEqual(loss_target.ndim, 0)
        self.assertGreaterEqual(float(loss_target.item()), 0.0)

    def test_losses_2_and_10_classes(self):
        for num_classes in (2, 10):
            model = SpikeVisionCSNN(in_channels=3, num_classes=num_classes)
            x = torch.rand(3, 3, 32, 32)
            targets = torch.randint(0, num_classes, (3,))
            spk_out, mem_out, _ = model(x, num_steps=4, encoding="direct")
            for loss_type in ("ce_rate", "ce_mem", "mse_count"):
                criterion = build_loss_fn(loss_type)
                loss = compute_temporal_loss(criterion, loss_type, spk_out, mem_out, targets)
                self.assertGreaterEqual(loss.numel(), 1)
                self.assertTrue(torch.isfinite(loss).all())

    def test_qat_ste_gradients_and_fuse_conv_bn_compatibility(self):
        torch.manual_seed(2)
        model = SpikeVisionCSNN(in_channels=3, num_classes=10, qat=True, qat_per_channel=True)
        model.train()
        x = torch.rand(2, 3, 32, 32)
        targets = torch.tensor([0, 3], dtype=torch.long)

        spk_out, mem_out, _ = model(x, num_steps=4, encoding="direct")
        criterion = build_loss_fn("ce_rate")
        loss = compute_temporal_loss(criterion, "ce_rate", spk_out, mem_out, targets)
        loss.backward()

        self.assertIsNotNone(model.conv1.weight.grad)
        self.assertIsNotNone(model.fc.weight.grad)
        self.assertTrue(torch.isfinite(model.conv1.weight.grad).all())
        self.assertTrue(torch.isfinite(model.fc.weight.grad).all())

        # State dict compatibility with non-QAT model and fuse_conv_bn
        plain = SpikeVisionCSNN(in_channels=3, num_classes=10, qat=False)
        plain.load_state_dict(model.state_dict())
        model.eval()
        w_fused, b_fused = fuse_conv_bn(model.conv1, model.bn1)
        self.assertEqual(w_fused.shape, model.conv1.weight.shape)
        self.assertEqual(b_fused.shape, (16,))


if __name__ == "__main__":
    unittest.main()
