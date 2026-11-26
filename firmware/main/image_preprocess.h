/**
 * SpikeVision-S3: Pré-processamento de Imagem 100% Inteiro (Zero Malloc)
 *
 * Converte frames RGB565, RGB888 e Gray8 de resoluções arbitrárias para o
 * tensor planar CHW 3x32x32 quantizado em [0..127], compatível com snn_engine_infer().
 */

#pragma once

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define IMG_PREPROC_WIDTH   32
#define IMG_PREPROC_HEIGHT  32
#define IMG_PREPROC_CHANS   3
#define IMG_PREPROC_PLANE   (IMG_PREPROC_WIDTH * IMG_PREPROC_HEIGHT)
#define IMG_PREPROC_BYTES   (IMG_PREPROC_CHANS * IMG_PREPROC_PLANE)

/**
 * Converte imagem RGB565 (interleaved) para tensor planar CHW 3x32x32 int8_t [0..127].
 * Aplica center-crop quadrado quando src_w != src_h e box/area downsampling inteiro.
 *
 * @param rgb565     Ponteiro para buffer RGB565 (src_w * src_h pixels).
 * @param src_w      Largura da imagem de origem (> 0).
 * @param src_h      Altura da imagem de origem (> 0).
 * @param swap_bytes true se os bytes de cada pixel 16-bit estiverem em big-endian (comum em DMA OV2640/OV5640).
 * @param out_chw32  Buffer de saída de 3072 bytes (3 * 32 * 32) em formato plano CHW [0..127].
 */
void img_rgb565_to_chw32(
    const uint16_t* rgb565,
    int src_w,
    int src_h,
    bool swap_bytes,
    int8_t* out_chw32);

/**
 * Converte imagem RGB888 (interleaved R,G,B) para tensor planar CHW 3x32x32 int8_t [0..127].
 * Aplica center-crop quadrado quando src_w != src_h e box/area downsampling inteiro.
 *
 * @param rgb888     Ponteiro para buffer RGB888 (src_w * src_h * 3 bytes).
 * @param src_w      Largura da imagem de origem (> 0).
 * @param src_h      Altura da imagem de origem (> 0).
 * @param out_chw32  Buffer de saída de 3072 bytes (3 * 32 * 32) em formato plano CHW [0..127].
 */
void img_rgb888_to_chw32(
    const uint8_t* rgb888,
    int src_w,
    int src_h,
    int8_t* out_chw32);

/**
 * Converte imagem Grayscale 8-bit para tensor planar CHW 3x32x32 int8_t [0..127]
 * replicando o canal de luminância nos 3 planos (R = G = B).
 *
 * @param gray8      Ponteiro para buffer Gray8 (src_w * src_h bytes).
 * @param src_w      Largura da imagem de origem (> 0).
 * @param src_h      Altura da imagem de origem (> 0).
 * @param out_chw32  Buffer de saída de 3072 bytes (3 * 32 * 32) em formato plano CHW [0..127].
 */
void img_gray8_to_chw32(
    const uint8_t* gray8,
    int src_w,
    int src_h,
    int8_t* out_chw32);

#ifdef __cplusplus
}
#endif
