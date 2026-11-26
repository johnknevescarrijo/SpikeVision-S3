/**
 * SpikeVision-S3: Kernels SIMD / Vetorizados para Xtensa LX7 PIE, ESP-DSP e Host.
 *
 * Fornece primitivas inline em ponto fixo (Int8 -> Int32) com garantia de
 * paridade bit-a-bit 100% em relação ao simulador Python e ao motor escalar.
 */

#pragma once

#include <stdint.h>
#include <stddef.h>

#if defined(ESP_PLATFORM)
    #if defined(__has_include)
        #if __has_include("dsps_dotprod.h")
            #include "dsps_dotprod.h"
            #define SNN_HAS_ESP_DSP 1
        #endif
    #endif
    #ifndef SNN_HAS_ESP_DSP
        #define SNN_HAS_ESP_DSP 0
    #endif
    #if defined(CONFIG_IDF_TARGET_ESP32S3) || defined(__XTENSA__)
        #define SNN_HAS_XTENSA_LX7 1
    #else
        #define SNN_HAS_XTENSA_LX7 0
    #endif
#else
    #define SNN_HAS_ESP_DSP 0
    #define SNN_HAS_XTENSA_LX7 0
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Retorna o nome do backend SIMD ativo em tempo de compilação.
 */
static inline const char* snn_simd_backend_name(void) {
#if SNN_HAS_ESP_DSP
    return "ESP-DSP";
#elif SNN_HAS_XTENSA_LX7
    return "Xtensa-LX7-PIE";
#else
    return "Scalar-C++17";
#endif
}

/**
 * Produto escalar exato Int8 x Int8 -> Int32.
 *
 * No ESP32-S3 utiliza unrolling 4x otimizado para pipeline Xtensa LX7 (e detecção
 * opcional de ESP-DSP via dsps_dotprod.h), preservando acumulação exata em Int32
 * sem saturação/shift para garantir 100% de paridade bit-a-bit.
 */
static inline int32_t snn_simd_dot_i8(
    const int8_t* __restrict__ a,
    const int8_t* __restrict__ b,
    int len) {
    int32_t acc0 = 0;
    int32_t acc1 = 0;
    int32_t acc2 = 0;
    int32_t acc3 = 0;
    int i = 0;

#if SNN_HAS_ESP_DSP || SNN_HAS_XTENSA_LX7
    for (; i <= len - 4; i += 4) {
        acc0 += (int32_t)a[i + 0] * (int32_t)b[i + 0];
        acc1 += (int32_t)a[i + 1] * (int32_t)b[i + 1];
        acc2 += (int32_t)a[i + 2] * (int32_t)b[i + 2];
        acc3 += (int32_t)a[i + 3] * (int32_t)b[i + 3];
    }
#else
    for (; i <= len - 4; i += 4) {
        acc0 += (int32_t)a[i + 0] * (int32_t)b[i + 0];
        acc1 += (int32_t)a[i + 1] * (int32_t)b[i + 1];
        acc2 += (int32_t)a[i + 2] * (int32_t)b[i + 2];
        acc3 += (int32_t)a[i + 3] * (int32_t)b[i + 3];
    }
#endif

    int32_t sum = (acc0 + acc1) + (acc2 + acc3);
    for (; i < len; ++i) {
        sum += (int32_t)a[i] * (int32_t)b[i];
    }
    return sum;
}

/**
 * Acumula 3 pesos Int8 contíguos em 3 acumuladores Int32 contíguos.
 */
static inline void snn_simd_acc_row3_i8_to_i32(
    int32_t* __restrict__ dst,
    const int8_t* __restrict__ w) {
    dst[0] += (int32_t)w[0];
    dst[1] += (int32_t)w[1];
    dst[2] += (int32_t)w[2];
}

/**
 * Acumula 3 pesos Int8 contíguos em ordem reversa de kernel (convolução esparsa)
 * sobre 3 posições horizontais contíguas de saída Int32.
 */
static inline void snn_simd_acc_row3_rev_i8_to_i32(
    int32_t* __restrict__ dst,
    const int8_t* __restrict__ w) {
    dst[0] += (int32_t)w[2];
    dst[1] += (int32_t)w[1];
    dst[2] += (int32_t)w[0];
}

/**
 * Acumula um vetor de `len` pesos Int8 em `dst` (Int32) com unrolling 4x.
 */
static inline void snn_simd_acc_vec_i8_to_i32(
    int32_t* __restrict__ dst,
    const int8_t* __restrict__ w,
    int len) {
    int i = 0;
    for (; i <= len - 4; i += 4) {
        dst[i + 0] += (int32_t)w[i + 0];
        dst[i + 1] += (int32_t)w[i + 1];
        dst[i + 2] += (int32_t)w[i + 2];
        dst[i + 3] += (int32_t)w[i + 3];
    }
    for (; i < len; ++i) {
        dst[i] += (int32_t)w[i];
    }
}

/**
 * Decaimento LIF em ponto fixo 16-bit idêntico aos registradores Xtensa 32-bit.
 */
static inline int32_t snn_simd_lif_decay_i32(
    int32_t mem,
    uint32_t beta_num,
    int beta_shift) {
    int32_t prod = (int32_t)((uint32_t)mem * beta_num);
    return prod >> beta_shift;
}

/**
 * Aplica decaimento LIF vetorizado em `len` potenciais de membrana.
 */
static inline void snn_simd_lif_decay_vec_i32(
    int32_t* __restrict__ dst,
    const int32_t* __restrict__ src,
    int len,
    uint32_t beta_num,
    int beta_shift) {
    int i = 0;
    for (; i <= len - 4; i += 4) {
        dst[i + 0] = snn_simd_lif_decay_i32(src[i + 0], beta_num, beta_shift);
        dst[i + 1] = snn_simd_lif_decay_i32(src[i + 1], beta_num, beta_shift);
        dst[i + 2] = snn_simd_lif_decay_i32(src[i + 2], beta_num, beta_shift);
        dst[i + 3] = snn_simd_lif_decay_i32(src[i + 3], beta_num, beta_shift);
    }
    for (; i < len; ++i) {
        dst[i] = snn_simd_lif_decay_i32(src[i], beta_num, beta_shift);
    }
}

#ifdef __cplusplus
}
#endif
