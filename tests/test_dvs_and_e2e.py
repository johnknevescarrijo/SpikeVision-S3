import json
import subprocess
import sys
import threading
import unittest
import urllib.request
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from dashboard.server import DashboardServer  # noqa: E402
from serial_client import SpikeVisionClient  # noqa: E402
from training.datasets.dvs_encoder import PseudoDVSEncoder, generate_saccade_sequence  # noqa: E402

HOST_BIN = REPO_ROOT / "firmware" / "build_host" / "spikevision_s3_host"


class TestPseudoDVSAndE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        subprocess.run(
            ["make", "-C", str(REPO_ROOT / "firmware"), "host"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert HOST_BIN.is_file(), f"Host binary missing: {HOST_BIN}"

    def test_pseudo_dvs_signed_delta_and_on_off_luma(self):
        base = np.full((3, 32, 32), 20, dtype=np.int8)
        moved = base.copy()
        moved[:, :8, :8] = 80
        moved[:, 8:16, :8] = -20

        for mode in ("signed_delta", "on_off_luma"):
            enc = PseudoDVSEncoder(mode=mode, threshold=10, binary_events=True, min_active_events=16)
            ev0, st0 = enc.process(base)
            self.assertEqual(ev0.shape, (3, 32, 32))
            self.assertEqual(int(np.count_nonzero(ev0)), 0)
            self.assertTrue(st0.is_static_frame)
            self.assertEqual(st0.sparsity, 1.0)

            # Identical frame produces zero events
            ev1, st1 = enc.process(base)
            self.assertEqual(int(np.count_nonzero(ev1)), 0)
            self.assertTrue(st1.is_static_frame)

            # Changed frame produces ON and OFF events
            ev2, st2 = enc.process(moved)
            self.assertGreater(st2.on_events, 0)
            self.assertGreater(st2.off_events, 0)
            self.assertFalse(st2.is_static_frame)
            if mode == "on_off_luma":
                self.assertTrue(np.all(ev2[2] == 0))

            enc.reset()
            ev3, st3 = enc.process(moved)
            self.assertEqual(int(np.count_nonzero(ev3)), 0)
            self.assertTrue(st3.is_static_frame)

    def test_generate_saccade_sequence(self):
        img = np.arange(3 * 32 * 32, dtype=np.int32).reshape(3, 32, 32) % 127
        img = img.astype(np.int8)
        seq = generate_saccade_sequence(img, num_frames=6, max_shift=2, seed=123)
        self.assertEqual(seq.shape, (6, 3, 32, 32))
        self.assertEqual(seq.dtype, np.int8)
        self.assertTrue(np.array_equal(seq[0], img))

    def test_firmware_host_test_suite(self):
        proc = subprocess.run(
            ["make", "-C", str(REPO_ROOT / "firmware"), "test"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertIn("SUCESSO ABSOLUTO", proc.stdout)

    def test_serial_client_e2e_and_dashboard_http(self):
        chw = np.zeros((3, 32, 32), dtype=np.int8)
        chw[0, :16, :16] = 100
        rgb = np.full((32, 32, 3), 64, dtype=np.uint8)

        with SpikeVisionClient(host_bin=str(HOST_BIN)) as client:
            info = client.ping()
            self.assertGreater(info["num_classes"], 0)
            self.assertGreater(info["timesteps"], 0)
            self.assertGreater(info["sram_bytes"], 0)

            res_chw = client.infer_chw32(chw)
            self.assertIn("pred", res_chw)
            self.assertEqual(len(res_chw["spike_counts"]), info["num_classes"])

            res_ee = client.infer_chw32(chw, early_exit="lossless")
            self.assertEqual(res_ee["pred"], res_chw["pred"])

            res_rgb = client.infer_rgb888(rgb, width=32, height=32, early_exit="aggressive")
            self.assertIn("pred", res_rgb)

            ack = client.reset_dvs()
            self.assertIn("type", ack)

            res_dvs = client.infer_chw32(chw, use_dvs=True)
            self.assertIn("pred", res_dvs)

            # Dashboard HTTP endpoints
            server = DashboardServer(("127.0.0.1", 0), client)
            port = server.server_address[1]
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base_url = f"http://127.0.0.1:{port}"
                with urllib.request.urlopen(f"{base_url}/") as resp:
                    self.assertEqual(resp.status, 200)
                    self.assertIn(b"SpikeVision", resp.read())

                with urllib.request.urlopen(f"{base_url}/api/info") as resp:
                    self.assertEqual(resp.status, 200)
                    api_info = json.loads(resp.read().decode("utf-8"))
                    self.assertEqual(len(api_info["class_names"]), info["num_classes"])

                req_infer = urllib.request.Request(
                    f"{base_url}/api/infer",
                    data=json.dumps(
                        {
                            "mode": "chw32",
                            "data": chw.flatten().tolist(),
                            "early_exit": "lossless",
                            "use_dvs": False,
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req_infer) as resp:
                    self.assertEqual(resp.status, 200)
                    infer_res = json.loads(resp.read().decode("utf-8"))
                    self.assertIn("pred", infer_res)

                req_reset = urllib.request.Request(
                    f"{base_url}/api/reset-dvs",
                    data=b"{}",
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req_reset) as resp:
                    self.assertEqual(resp.status, 200)
                    reset_res = json.loads(resp.read().decode("utf-8"))
                    self.assertIn("type", reset_res)
            finally:
                server.shutdown()
                server.server_close()


if __name__ == "__main__":
    unittest.main()
