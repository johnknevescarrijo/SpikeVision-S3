#!/usr/bin/env python3
"""
SpikeVision-S3: Cliente Python para o Protocolo Serial Framed (USB-CDC / Host Pipe).

Framing binário:
  MAGIC(2) = b'SV' | CMD(1) | FLAGS(1) | LEN(2 uint16 LE) | PAYLOAD(LEN) | CRC16_CCITT(2 uint16 LE)

CRC-16/CCITT-FALSE (poly=0x1021, init=0xFFFF) sobre CMD | FLAGS | LEN | PAYLOAD.
Respostas do dispositivo em JSON-line ('\n').
"""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
import threading
from typing import Any, Iterable

MAGIC = b"SV"

CMD_PING = 0x01
CMD_INFER_CHW32 = 0x02
CMD_INFER_RGB888 = 0x03
CMD_RESET_DVS = 0x04

FLAG_EE_LOSSLESS = 0x01
FLAG_EE_AGGRESSIVE = 0x02
FLAG_USE_DVS = 0x04


def crc16_ccitt(data: bytes) -> int:
    """Calcula CRC-16/CCITT-FALSE (poly=0x1021, init=0xFFFF)."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def build_frame(cmd: int, flags: int = 0, payload: bytes = b"") -> bytes:
    """Empacota um frame binário completo com MAGIC, header, payload e CRC16."""
    header = struct.pack("<BBH", cmd & 0xFF, flags & 0xFF, len(payload))
    body = header + payload
    crc = crc16_ccitt(body)
    return MAGIC + body + struct.pack("<H", crc)


def _encode_early_exit_flags(early_exit: str | None, use_dvs: bool) -> int:
    flags = 0
    if early_exit == "lossless":
        flags |= FLAG_EE_LOSSLESS
    elif early_exit == "aggressive":
        flags |= FLAG_EE_AGGRESSIVE
    elif early_exit not in (None, "", "none", "off"):
        raise ValueError(f"Modo early_exit inválido: {early_exit!r}")
    if use_dvs:
        flags |= FLAG_USE_DVS
    return flags


def _to_int8_bytes(arr_i8: Any, expected_len: int) -> bytes:
    if hasattr(arr_i8, "astype") and hasattr(arr_i8, "tobytes"):
        # Suporte opcional a numpy.ndarray sem importar numpy
        flat = arr_i8.reshape(-1)
        if len(flat) != expected_len:
            raise ValueError(f"Esperado {expected_len} elementos, recebido {len(flat)}")
        return flat.astype("int8").tobytes()

    if isinstance(arr_i8, (bytes, bytearray, memoryview)):
        raw = bytes(arr_i8)
        if len(raw) != expected_len:
            raise ValueError(f"Esperado {expected_len} bytes, recebido {len(raw)}")
        return raw

    vals = list(arr_i8)
    if len(vals) != expected_len:
        raise ValueError(f"Esperado {expected_len} elementos, recebido {len(vals)}")
    return bytes(int(v) & 0xFF for v in vals)


def _to_uint8_bytes(arr_u8: Any, expected_len: int) -> bytes:
    if hasattr(arr_u8, "astype") and hasattr(arr_u8, "tobytes"):
        flat = arr_u8.reshape(-1)
        if len(flat) != expected_len:
            raise ValueError(f"Esperado {expected_len} bytes, recebido {len(flat)}")
        return flat.astype("uint8").tobytes()

    if isinstance(arr_u8, (bytes, bytearray, memoryview)):
        raw = bytes(arr_u8)
        if len(raw) != expected_len:
            raise ValueError(f"Esperado {expected_len} bytes, recebido {len(raw)}")
        return raw

    vals = list(arr_u8)
    if len(vals) != expected_len:
        raise ValueError(f"Esperado {expected_len} bytes, recebido {len(vals)}")
    return bytes(max(0, min(255, int(v))) for v in vals)


class SpikeVisionClient:
    """Cliente síncrono thread-safe para inferência remota no SpikeVision-S3."""

    def __init__(
        self,
        port: str | None = None,
        baud: int = 115200,
        host_bin: str | None = None,
        timeout: float = 5.0,
    ) -> None:
        if not port and not host_bin:
            raise ValueError("Informe 'port' (serial) ou 'host_bin' (executável host --serial).")

        self._lock = threading.Lock()
        self._proc: subprocess.Popen[bytes] | None = None
        self._ser: Any = None

        if host_bin:
            self._proc = subprocess.Popen(
                [host_bin, "--serial"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        else:
            try:
                import serial  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "pyserial não instalado. Instale com 'pip install pyserial' ou use --host-bin."
                ) from exc
            self._ser = serial.Serial(port=port, baudrate=baud, timeout=timeout)

    def __enter__(self) -> "SpikeVisionClient":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def _write_and_readline(self, frame: bytes) -> dict[str, Any]:
        with self._lock:
            if self._proc is not None:
                if self._proc.poll() is not None:
                    stderr_out = self._proc.stderr.read().decode("utf-8", errors="replace") if self._proc.stderr else ""
                    raise RuntimeError(f"Processo host encerrou inesperadamente (rc={self._proc.returncode}): {stderr_out}")
                assert self._proc.stdin is not None and self._proc.stdout is not None
                self._proc.stdin.write(frame)
                self._proc.stdin.flush()
                line = self._proc.stdout.readline()
            elif self._ser is not None:
                self._ser.write(frame)
                self._ser.flush()
                line = self._ser.readline()
            else:
                raise RuntimeError("Cliente já fechado.")

        if not line:
            raise RuntimeError("Timeout ou EOF aguardando resposta do dispositivo.")

        text = line.decode("utf-8", errors="replace").strip()
        resp = json.loads(text)
        if resp.get("type") == "error":
            raise RuntimeError(f"Erro do protocolo: {resp.get('code', 'unknown')}")
        return resp

    def ping(self) -> dict[str, Any]:
        return self._write_and_readline(build_frame(CMD_PING))

    def infer_chw32(
        self,
        arr_i8: Iterable[int] | bytes | Any,
        early_exit: str | None = None,
        use_dvs: bool = False,
    ) -> dict[str, Any]:
        payload = _to_int8_bytes(arr_i8, 3 * 32 * 32)
        flags = _encode_early_exit_flags(early_exit, use_dvs)
        return self._write_and_readline(build_frame(CMD_INFER_CHW32, flags, payload))

    def infer_rgb888(
        self,
        rgb_u8: Iterable[int] | bytes | Any,
        width: int = 32,
        height: int = 32,
        early_exit: str | None = None,
        use_dvs: bool = False,
    ) -> dict[str, Any]:
        if width <= 0 or height <= 0 or width > 64 or height > 64:
            raise ValueError(f"Dimensões inválidas ({width}x{height}); máximo suportado é 64x64.")
        if hasattr(rgb_u8, "shape") and len(rgb_u8.shape) == 3:
            height, width = int(rgb_u8.shape[0]), int(rgb_u8.shape[1])
        raw = _to_uint8_bytes(rgb_u8, width * height * 3)
        payload = struct.pack("<HH", width, height) + raw
        flags = _encode_early_exit_flags(early_exit, use_dvs)
        return self._write_and_readline(build_frame(CMD_INFER_RGB888, flags, payload))

    def reset_dvs(self) -> dict[str, Any]:
        return self._write_and_readline(build_frame(CMD_RESET_DVS))

    def close(self) -> None:
        with self._lock:
            if self._proc is not None:
                try:
                    if self._proc.stdin:
                        self._proc.stdin.close()
                except OSError:
                    pass
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    self._proc.wait()
                self._proc = None
            if self._ser is not None:
                try:
                    self._ser.close()
                finally:
                    self._ser = None


def run_self_test(client: SpikeVisionClient) -> int:
    print("[1/5] Testando CMD_PING...")
    pong = client.ping()
    assert pong.get("type") == "pong", f"Resposta inesperada: {pong}"
    assert pong.get("num_classes", 0) > 0
    assert pong.get("timesteps", 0) > 0
    assert pong.get("sram_bytes", 0) > 0
    print(f"  OK: {pong}")

    print("[2/5] Testando CMD_INFER_CHW32 (normal e early-exit lossless)...")
    sample_a = bytes(((i * 37) % 127) for i in range(3072))
    res_norm = client.infer_chw32(sample_a, early_exit=None, use_dvs=False)
    assert res_norm.get("type") == "infer"
    assert len(res_norm["spike_counts"]) == pong["num_classes"]
    assert res_norm["timesteps_executed"] == pong["timesteps"]

    res_ee = client.infer_chw32(sample_a, early_exit="lossless", use_dvs=False)
    assert res_ee["pred"] == res_norm["pred"], "Early-exit lossless divergiu da predição normal!"
    print(f"  OK: pred={res_norm['pred']}, sops={res_norm['total_sops']}, lat={res_norm['latency_us']}us")

    print("[3/5] Testando CMD_INFER_CHW32 com Pseudo-DVS (use_dvs=True)...")
    res_dvs1 = client.infer_chw32(sample_a, use_dvs=True)
    assert res_dvs1.get("dvs_static") is True, f"1º frame DVS deveria ser estático: {res_dvs1}"
    assert res_dvs1.get("dvs_sparsity") == 100.0

    sample_b = bytearray(sample_a)
    for i in range(512):
        sample_b[i] = min(127, sample_b[i] + 40)
    res_dvs2 = client.infer_chw32(bytes(sample_b), use_dvs=True)
    assert res_dvs2.get("dvs_static") is False, f"2º frame com delta deveria disparar eventos: {res_dvs2}"
    assert res_dvs2.get("dvs_on", 0) + res_dvs2.get("dvs_off", 0) > 0
    print(f"  OK: dvs_on={res_dvs2['dvs_on']}, dvs_off={res_dvs2['dvs_off']}, dvs_sparsity={res_dvs2['dvs_sparsity']}%")

    print("[4/5] Testando CMD_INFER_RGB888 (32x32 e 64x64)...")
    rgb32 = bytes(((i * 13) % 256) for i in range(32 * 32 * 3))
    res_rgb32 = client.infer_rgb888(rgb32, width=32, height=32)
    assert res_rgb32.get("type") == "infer"

    rgb64 = bytes(((i * 7) % 256) for i in range(64 * 64 * 3))
    res_rgb64 = client.infer_rgb888(rgb64, width=64, height=64)
    assert res_rgb64.get("type") == "infer"
    print(f"  OK: rgb32 pred={res_rgb32['pred']}, rgb64 pred={res_rgb64['pred']}")

    print("[5/5] Testando CMD_RESET_DVS...")
    ack = client.reset_dvs()
    assert ack == {"type": "ack", "cmd": "reset_dvs"}, f"ACK inesperado: {ack}"
    res_after_reset = client.infer_chw32(sample_a, use_dvs=True)
    assert res_after_reset.get("dvs_static") is True, "Após reset_dvs, 1º frame deve ser estático"
    print("  OK: reset_dvs validado.")

    print("SELF-TEST 100% PASS!")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="SpikeVision-S3 Serial Client")
    parser.add_argument("--port", help="Porta serial (ex: /dev/ttyACM0)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate (padrão: 115200)")
    parser.add_argument("--host-bin", help="Caminho do binário host (executado com --serial)")
    parser.add_argument("--self-test", action="store_true", help="Executa bateria E2E de validação do protocolo")
    args = parser.parse_args()

    if not args.port and not args.host_bin:
        parser.error("Informe --port ou --host-bin.")

    with SpikeVisionClient(port=args.port, baud=args.baud, host_bin=args.host_bin) as client:
        if args.self_test:
            return run_self_test(client)
        print(json.dumps(client.ping(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
