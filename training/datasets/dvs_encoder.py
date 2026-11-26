"""
Contraparte Python bit-a-bit idêntica ao encoder Pseudo-DVS em C++ (firmware/main/dvs_encoder.cpp).
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


DVS_FRAME_SHAPE = (3, 32, 32)
DVS_PLANE_SIZE = 32 * 32
DVS_TOTAL_BYTES = 3 * 32 * 32


@dataclass
class DVSFrameStats:
    total_sites: int
    on_events: int
    off_events: int
    sparsity: float
    is_static_frame: bool


def _compute_luma_y(curr_chw32: np.ndarray) -> np.ndarray:
    """Calcula Y = (77*R + 150*G + 29*B) >> 8 em aritmética inteira exata."""
    r = np.maximum(curr_chw32[0].astype(np.int32), 0)
    g = np.maximum(curr_chw32[1].astype(np.int32), 0)
    b = np.maximum(curr_chw32[2].astype(np.int32), 0)
    return ((77 * r + 150 * g + 29 * b) >> 8).astype(np.uint8)


class PseudoDVSEncoder:
    """
    Encoder Pseudo-DVS por diferença temporal com atualização condicional de referência.
    100% bit-a-bit idêntico a dvs_encoder_process() em C++.
    """

    def __init__(
        self,
        mode: str = "signed_delta",
        threshold: int = 8,
        event_amplitude: int = 127,
        binary_events: bool = True,
        min_active_events: int = 16,
    ) -> None:
        normalized_mode = mode.lower()
        if normalized_mode not in ("signed_delta", "on_off_luma"):
            raise ValueError(f"Modo inválido: {mode!r}. Esperado 'signed_delta' ou 'on_off_luma'.")

        self.mode = normalized_mode
        self.threshold = int(threshold)
        self.event_amplitude = int(np.int8(event_amplitude))
        self.binary_events = bool(binary_events)
        self.min_active_events = int(min_active_events)

        self._ref_frame = np.zeros(DVS_FRAME_SHAPE, dtype=np.int8)
        self._ref_luma = np.zeros((32, 32), dtype=np.uint8)
        self._initialized = False

    def reset(self) -> None:
        self._ref_frame.fill(0)
        self._ref_luma.fill(0)
        self._initialized = False

    def process(self, curr_chw32: np.ndarray) -> Tuple[np.ndarray, DVSFrameStats]:
        frame = np.asarray(curr_chw32, dtype=np.int8)
        if frame.shape != DVS_FRAME_SHAPE:
            raise ValueError(f"Shape esperado {DVS_FRAME_SHAPE}, recebido {frame.shape}.")

        total_sites = DVS_PLANE_SIZE if self.mode == "on_off_luma" else DVS_TOTAL_BYTES
        out_event = np.zeros(DVS_FRAME_SHAPE, dtype=np.int8)

        if not self._initialized:
            if self.mode == "on_off_luma":
                self._ref_luma = _compute_luma_y(frame)
            else:
                self._ref_frame = frame.copy()
            self._initialized = True
            return out_event, DVSFrameStats(
                total_sites=total_sites,
                on_events=0,
                off_events=0,
                sparsity=1.0,
                is_static_frame=True,
            )

        if self.mode == "on_off_luma":
            y = _compute_luma_y(frame)
            diff = y.astype(np.int16) - self._ref_luma.astype(np.int16)

            on_mask = diff >= self.threshold
            off_mask = (~on_mask) & (diff <= -self.threshold)

            if self.binary_events:
                out_event[0, on_mask] = np.int8(self.event_amplitude)
                out_event[1, off_mask] = np.int8(self.event_amplitude)
            else:
                out_event[0, on_mask] = np.clip(diff[on_mask], 0, 127).astype(np.int8)
                out_event[1, off_mask] = np.clip(-diff[off_mask].astype(np.int32), 0, 127).astype(np.int8)

            active_mask = on_mask | off_mask
            self._ref_luma[active_mask] = y[active_mask]

            on_events = int(np.count_nonzero(on_mask))
            off_events = int(np.count_nonzero(off_mask))
        else:
            diff = frame.astype(np.int16) - self._ref_frame.astype(np.int16)

            on_mask = diff >= self.threshold
            off_mask = (~on_mask) & (diff <= -self.threshold)

            if self.binary_events:
                out_event[on_mask] = np.int8(self.event_amplitude)
                out_event[off_mask] = np.int8(-self.event_amplitude)
            else:
                out_event[on_mask] = np.clip(diff[on_mask], -127, 127).astype(np.int8)
                out_event[off_mask] = np.clip(diff[off_mask], -127, 127).astype(np.int8)

            active_mask = on_mask | off_mask
            self._ref_frame[active_mask] = frame[active_mask]

            on_events = int(np.count_nonzero(on_mask))
            off_events = int(np.count_nonzero(off_mask))

        total_events = on_events + off_events
        sparsity = 1.0 - (float(total_events) / float(total_sites))
        is_static = total_events < self.min_active_events

        return out_event, DVSFrameStats(
            total_sites=total_sites,
            on_events=on_events,
            off_events=off_events,
            sparsity=sparsity,
            is_static_frame=is_static,
        )


def generate_saccade_sequence(
    img_chw32: np.ndarray,
    num_frames: int = 8,
    max_shift: int = 2,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Sintetiza uma sequência de micro-sacadas a partir de uma imagem CIFAR-10 CHW (3, 32, 32).
    O frame 0 corresponde à imagem original sem deslocamento; os frames seguintes aplicam
    deslocamentos inteiros (dy, dx) em [-max_shift, max_shift] com replicação de borda.
    """
    img = np.asarray(img_chw32, dtype=np.int8)
    if img.shape != DVS_FRAME_SHAPE:
        raise ValueError(f"Shape esperado {DVS_FRAME_SHAPE}, recebido {img.shape}.")
    if num_frames < 1:
        raise ValueError("num_frames deve ser >= 1.")
    if max_shift < 0:
        raise ValueError("max_shift deve ser >= 0.")

    rng = np.random.default_rng(seed)
    seq = np.empty((num_frames, *DVS_FRAME_SHAPE), dtype=np.int8)
    seq[0] = img

    _, h, w = DVS_FRAME_SHAPE
    for t in range(1, num_frames):
        if max_shift == 0:
            seq[t] = img
            continue
        dy = int(rng.integers(-max_shift, max_shift + 1))
        dx = int(rng.integers(-max_shift, max_shift + 1))
        ys = np.clip(np.arange(h) - dy, 0, h - 1)
        xs = np.clip(np.arange(w) - dx, 0, w - 1)
        seq[t] = img[:, ys[:, None], xs[None, :]]

    return seq
