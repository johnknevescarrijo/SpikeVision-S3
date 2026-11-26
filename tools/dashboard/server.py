#!/usr/bin/env python3
"""
SpikeVision-S3: Servidor HTTP Local do Dashboard Web (100% stdlib).

Endpoints:
  GET  /              -> tools/dashboard/index.html
  GET  /api/info      -> ping + metadados de classes
  POST /api/infer     -> inferência CHW32 ou RGB888 (com Early-Exit e Pseudo-DVS)
  POST /api/reset-dvs -> reinicia estado de referência do encoder Pseudo-DVS
"""

from __future__ import annotations

import argparse
import json
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

TOOLS_DIR = Path(__file__).resolve().parent.parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from serial_client import SpikeVisionClient  # noqa: E402

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

INDEX_HTML_PATH = Path(__file__).resolve().parent / "index.html"


def get_class_names(num_classes: int) -> list[str]:
    if num_classes == 10:
        return list(CIFAR10_CLASSES)
    if num_classes == 2:
        return ["non-cat", "cat"]
    return [f"class_{i}" for i in range(num_classes)]


class DashboardServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        client: SpikeVisionClient,
    ) -> None:
        self.client = client
        self.index_html = INDEX_HTML_PATH.read_bytes()
        super().__init__(server_address, DashboardHandler)


class DashboardHandler(BaseHTTPRequestHandler):
    server: DashboardServer

    def log_message(self, format: str, *args: Any) -> None:
        # Mantém saída limpa; erros são logados explicitamente quando necessário
        return

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 2 * 1024 * 1024:
            raise ValueError("Content-Length inválido.")
        raw = self.rfile.read(length)
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Corpo JSON deve ser um objeto.")
        return data

    def do_GET(self) -> None:
        if self.path == "/" or self.path == "/index.html":
            body = self.server.index_html
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/api/info":
            try:
                info = self.server.client.ping()
                info["class_names"] = get_class_names(int(info.get("num_classes", 10)))
                self._send_json(HTTPStatus.OK, info)
            except Exception as exc:
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def do_POST(self) -> None:
        if self.path == "/api/infer":
            try:
                req = self._read_json_body()
                mode = req.get("mode", "rgb888")
                early_exit = req.get("early_exit")
                use_dvs = bool(req.get("use_dvs", False))
                data = req.get("data")
                if not isinstance(data, list):
                    raise ValueError("Campo 'data' deve ser uma lista de inteiros.")

                if mode == "chw32":
                    res = self.server.client.infer_chw32(
                        data, early_exit=early_exit, use_dvs=use_dvs
                    )
                elif mode == "rgb888":
                    width = int(req.get("width", 32))
                    height = int(req.get("height", 32))
                    res = self.server.client.infer_rgb888(
                        data, width=width, height=height, early_exit=early_exit, use_dvs=use_dvs
                    )
                else:
                    raise ValueError(f"Modo inválido: {mode!r}")

                self._send_json(HTTPStatus.OK, res)
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except Exception as exc:
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
            return

        if self.path == "/api/reset-dvs":
            try:
                ack = self.server.client.reset_dvs()
                self._send_json(HTTPStatus.OK, ack)
            except Exception as exc:
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})


def main() -> int:
    parser = argparse.ArgumentParser(description="SpikeVision-S3 Web Dashboard Server")
    parser.add_argument("--port", help="Porta serial do ESP32-S3 (ex: /dev/ttyACM0)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate serial")
    parser.add_argument("--host-bin", help="Caminho do binário host (executado com --serial)")
    parser.add_argument("--bind", default="127.0.0.1", help="Endereço HTTP bind (padrão: 127.0.0.1)")
    parser.add_argument("--http-port", type=int, default=8080, help="Porta HTTP (padrão: 8080)")
    args = parser.parse_args()

    if not args.port and not args.host_bin:
        parser.error("Informe --port ou --host-bin.")

    with SpikeVisionClient(port=args.port, baud=args.baud, host_bin=args.host_bin) as client:
        server = DashboardServer((args.bind, args.http_port), client)
        host, port = server.server_address
        print(f"SpikeVision-S3 Dashboard rodando em http://{host}:{port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
