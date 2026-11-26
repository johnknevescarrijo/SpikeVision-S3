/**
 * SpikeVision-S3: Motor de Inferência Neuromórfica em C/C++ para ESP32-S3.
 *
 * Implementa execução de redes neurais pulsadas (CSNN) quantizadas em Int8/Int32
 * com aritmética bit-accurate ao simulador Python, orientada a eventos (SOPs)
 * e alocação 100% estática na SRAM interna.
 */

#include "snn_engine.h"
#include "snn_simd.h"
#include <string.h>

#if defined(ESP_PLATFORM)
    #include "esp_timer.h"
    #include "esp_cpu.h"
#else
    #include <chrono>
#endif

/* =========================================================================
 * Buffers Estáticos de Estado e Ativação na SRAM Interna (Alinhados em 16 bytes)
 * Zero chamadas a malloc/free durante a inferência.
 * ========================================================================= */

#define SNN_BITPACK_WORDS(n) (((n) + 31) / 32)
#define L1_SPK_ELEMS         (CONV1_OUT_CHANNELS * POOL1_OUT_H * POOL1_OUT_W)
#define L2_SPK_ELEMS         (CONV2_OUT_CHANNELS * POOL2_OUT_H * POOL2_OUT_W)
#define L3_SPK_ELEMS         (CONV3_OUT_CHANNELS * POOL3_OUT_H * POOL3_OUT_W)

#define L1_SPK_WORDS         SNN_BITPACK_WORDS(L1_SPK_ELEMS)
#define L2_SPK_WORDS         SNN_BITPACK_WORDS(L2_SPK_ELEMS)
#define L3_SPK_WORDS         SNN_BITPACK_WORDS(L3_SPK_ELEMS)

#define L2_CUR_ELEMS         (CONV2_OUT_CHANNELS * 16 * 16)
#define L3_CUR_ELEMS         (CONV3_OUT_CHANNELS * 8 * 8)
#define SCRATCH_CUR_ELEMS    ((L2_CUR_ELEMS > L3_CUR_ELEMS) ? L2_CUR_ELEMS : L3_CUR_ELEMS)

/* Scratchpad compartilhado de correntes pós-sinápticas (Camadas 2 e 3): 32 KB */
SPIKEVISION_ALIGN16 static int32_t s_scratch_cur[SCRATCH_CUR_ELEMS];

/* Camada 1: MaxPool 16x16x16 persistente para os T timesteps + estado LIF + spikes bit-packed */
SPIKEVISION_ALIGN16 static int32_t  s_pool1[CONV1_OUT_CHANNELS][POOL1_OUT_H][POOL1_OUT_W];
SPIKEVISION_ALIGN16 static int32_t  s_mem1[CONV1_OUT_CHANNELS][POOL1_OUT_H][POOL1_OUT_W];
SPIKEVISION_ALIGN16 static uint32_t s_spk1[L1_SPK_WORDS];

/* Camada 2: estado LIF 32x8x8 + spikes bit-packed (MaxPool fundido no loop LIF) */
SPIKEVISION_ALIGN16 static int32_t  s_mem2[CONV2_OUT_CHANNELS][POOL2_OUT_H][POOL2_OUT_W];
SPIKEVISION_ALIGN16 static uint32_t s_spk2[L2_SPK_WORDS];

/* Camada 3: estado LIF 32x4x4 + spikes bit-packed (MaxPool fundido no loop LIF) */
SPIKEVISION_ALIGN16 static int32_t  s_mem3[CONV3_OUT_CHANNELS][POOL3_OUT_H][POOL3_OUT_W];
SPIKEVISION_ALIGN16 static uint32_t s_spk3[L3_SPK_WORDS];

/* Camada 4: Linear Head */
SPIKEVISION_ALIGN16 static int32_t s_mem4[FC_OUT_FEATURES];
SPIKEVISION_ALIGN16 static int32_t s_spk_counts[FC_OUT_FEATURES];

static inline void spk_clear_all(uint32_t* buf, size_t num_words) {
    memset(buf, 0, num_words * sizeof(uint32_t));
}

static inline void spk_set(uint32_t* buf, int idx) {
    buf[idx >> 5] |= (1u << (idx & 31));
}

/* =========================================================================
 * Dinâmica Leaky LIF em Ponto Fixo 16-bit
 * ========================================================================= */

/**
 * Aplica decaimento com aritmética inteira nativa de hardware de 32 bits.
 * Reproduz bit-a-bit os registradores da CPU Xtensa do ESP32-S3.
 */
static inline int32_t lif_decay_32(int32_t mem) {
    return snn_simd_lif_decay_i32(mem, (uint32_t)SPIKEVISION_BETA_NUM, SPIKEVISION_BETA_SHIFT);
}

static inline int32_t vth_l1(int c) {
#ifdef SPIKEVISION_PER_CHANNEL_QUANT
    return VTH_L1_CH[c];
#else
    (void)c;
    return SPIKEVISION_L1_VTH;
#endif
}

static inline int32_t vth_l2(int c) {
#ifdef SPIKEVISION_PER_CHANNEL_QUANT
    return VTH_L2_CH[c];
#else
    (void)c;
    return SPIKEVISION_L2_VTH;
#endif
}

static inline int32_t vth_l3(int c) {
#ifdef SPIKEVISION_PER_CHANNEL_QUANT
    return VTH_L3_CH[c];
#else
    (void)c;
    return SPIKEVISION_L3_VTH;
#endif
}

static inline int32_t vth_fc(int c) {
#ifdef SPIKEVISION_PER_CHANNEL_QUANT
    return VTH_FC_CH[c];
#else
    (void)c;
    return SPIKEVISION_L4_VTH;
#endif
}

/* =========================================================================
 * API do Motor Neuromórfico
 * ========================================================================= */

void snn_engine_init(void) {
    memset(s_scratch_cur, 0, sizeof(s_scratch_cur));

    memset(s_pool1, 0, sizeof(s_pool1));
    memset(s_mem1, 0, sizeof(s_mem1));
    spk_clear_all(s_spk1, L1_SPK_WORDS);

    memset(s_mem2, 0, sizeof(s_mem2));
    spk_clear_all(s_spk2, L2_SPK_WORDS);

    memset(s_mem3, 0, sizeof(s_mem3));
    spk_clear_all(s_spk3, L3_SPK_WORDS);

    memset(s_mem4, 0, sizeof(s_mem4));
    memset(s_spk_counts, 0, sizeof(s_spk_counts));
}

size_t snn_engine_get_sram_bytes(void) {
    return sizeof(s_scratch_cur) +
           sizeof(s_pool1) + sizeof(s_mem1) + sizeof(s_spk1) +
           sizeof(s_mem2) + sizeof(s_spk2) +
           sizeof(s_mem3) + sizeof(s_spk3) +
           sizeof(s_mem4) + sizeof(s_spk_counts);
}

size_t snn_engine_get_sram_usage(void) {
    return snn_engine_get_sram_bytes();
}

const char* snn_engine_get_simd_backend(void) {
    return snn_simd_backend_name();
}

static inline bool should_early_exit(
    int32_t timesteps_done,
    int32_t total_timesteps,
    const int32_t* spk_counts,
    const snn_early_exit_cfg_t* cfg) {
    if (!cfg || !cfg->enabled) {
        return false;
    }
    int32_t remaining = total_timesteps - timesteps_done;
    if (remaining <= 0) {
        return false;
    }
    int32_t min_t = (cfg->min_timesteps < 1) ? 1 : cfg->min_timesteps;
    if (timesteps_done < min_t) {
        return false;
    }

    int32_t top1 = spk_counts[0];
    int32_t top2 = (FC_OUT_FEATURES > 1) ? spk_counts[1] : 0;
    if (FC_OUT_FEATURES > 1 && top2 > top1) {
        int32_t tmp = top1;
        top1 = top2;
        top2 = tmp;
    }
    for (int c = 2; c < FC_OUT_FEATURES; ++c) {
        int32_t s = spk_counts[c];
        if (s > top1) {
            top2 = top1;
            top1 = s;
        } else if (s > top2) {
            top2 = s;
        }
    }

    int32_t gap = top1 - top2;
    if (cfg->margin > 0) {
        return gap >= cfg->margin;
    }
    /* Modo lossless matemático: impossível a 2ª classe alcançar a líder nos timesteps restantes */
    return gap > remaining;
}

static inline int32_t conv1_pixel(const uint8_t* image_chw, int och, int32_t bias, int y, int x) {
    int8_t patch[27];
    int p = 0;
    for (int ich = 0; ich < CONV1_IN_CHANNELS; ich++) {
        const uint8_t* in_plane = image_chw + ich * (32 * 32);
        for (int ky = 0; ky < 3; ky++) {
            int iy = y + ky - 1;
            if (iy < 0 || iy >= 32) {
                patch[p++] = 0;
                patch[p++] = 0;
                patch[p++] = 0;
                continue;
            }
            const uint8_t* row = in_plane + iy * 32;
            for (int kx = 0; kx < 3; kx++) {
                int ix = x + kx - 1;
                patch[p++] = (ix < 0 || ix >= 32) ? 0 : (int8_t)row[ix];
            }
        }
    }
    return bias + snn_simd_dot_i8(patch, &conv1_weights[och * 27], 27);
}

void snn_engine_infer(const uint8_t* image_chw, SNNInferenceResult* result) {
    snn_engine_infer_adaptive(image_chw, result, nullptr);
}

void snn_infer_adaptive(
    const uint8_t* image_chw,
    SNNInferenceResult* result,
    const snn_early_exit_cfg_t* early_exit_cfg) {
    snn_engine_infer_adaptive(image_chw, result, early_exit_cfg);
}

void snn_engine_infer_adaptive(
    const uint8_t* image_chw,
    SNNInferenceResult* result,
    const snn_early_exit_cfg_t* early_exit_cfg) {
    if (!image_chw || !result) {
        return;
    }

    /* 1. Medição inicial de tempo e ciclos */
#if defined(ESP_PLATFORM)
    uint64_t t_start = esp_timer_get_time();
    uint32_t cycles_start = esp_cpu_get_cycle_count();
#else
    auto t_start = std::chrono::high_resolution_clock::now();
#endif

    /* 2. Zera potenciais de membrana e acumuladores de disparo */
    memset(s_mem1, 0, sizeof(s_mem1));
    memset(s_mem2, 0, sizeof(s_mem2));
    memset(s_mem3, 0, sizeof(s_mem3));
    memset(s_mem4, 0, sizeof(s_mem4));
    memset(s_spk_counts, 0, sizeof(s_spk_counts));

    uint64_t active_sops = 0;
    uint32_t total_spks_l1 = 0;
    uint32_t total_spks_l2 = 0;
    uint32_t total_spks_l3 = 0;
    uint32_t total_spks_l4 = 0;

    /* 3. CAMADA 1: Direct Coding Convolucional + MaxPool2d(2) Fundidos (Calculada 1x por Imagem)
     * Como a imagem de entrada é estática ao longo dos 8 timesteps, a convolução
     * 3x3 + bias + MaxPool2d é pré-calculada uma única vez direto para s_pool1,
     * eliminando completamente o buffer intermediário s_cur1 (64 KB)!
     */
    for (int och = 0; och < CONV1_OUT_CHANNELS; och++) {
        int32_t bias = conv1_bias[och];
        for (int py = 0; py < POOL1_OUT_H; py++) {
            int y0 = py * 2;
            for (int px = 0; px < POOL1_OUT_W; px++) {
                int x0 = px * 2;
                int32_t m = conv1_pixel(image_chw, och, bias, y0, x0);
                int32_t v01 = conv1_pixel(image_chw, och, bias, y0, x0 + 1);
                if (v01 > m) m = v01;
                int32_t v10 = conv1_pixel(image_chw, och, bias, y0 + 1, x0);
                if (v10 > m) m = v10;
                int32_t v11 = conv1_pixel(image_chw, och, bias, y0 + 1, x0 + 1);
                if (v11 > m) m = v11;
                s_pool1[och][py][px] = m;
            }
        }
    }

    /* 4. LOOP TEMPORAL NEUROMÓRFICO (T = 8 Timesteps) */
    int32_t timesteps_executed = 0;
    bool early_exited = false;
    for (int t = 0; t < SPIKEVISION_TIMESTEPS; t++) {
        timesteps_executed = t + 1;

        /* --- CAMADA 1: Leaky LIF (16 x 16 x 16) com Bit-Packing --- */
        spk_clear_all(s_spk1, L1_SPK_WORDS);
        for (int och = 0; och < CONV1_OUT_CHANNELS; och++) {
            int32_t vth = vth_l1(och);
            for (int y = 0; y < POOL1_OUT_H; y++) {
                for (int x = 0; x < POOL1_OUT_W; x++) {
                    int32_t m = s_mem1[och][y][x];
                    int32_t r = (m >= vth) ? 1 : 0;
                    m = lif_decay_32(m) + s_pool1[och][y][x] - r * vth;
                    if (m >= vth) {
                        int idx = (och * POOL1_OUT_H + y) * POOL1_OUT_W + x;
                        spk_set(s_spk1, idx);
                        total_spks_l1++;
                    }
                    s_mem1[och][y][x] = m;
                }
            }
        }

        /* --- CAMADA 2: Convolução Esparsa Orientada a Eventos (SOP) com __builtin_ctz ---
         * Pula blocos inteiros de 32 neurônios silenciosos com if (mask == 0) continue;
         */
        for (int och = 0; och < CONV2_OUT_CHANNELS; och++) {
            int32_t b = conv2_bias[och];
            int32_t* cur_plane = &s_scratch_cur[och * (16 * 16)];
            for (int i = 0; i < 16 * 16; i++) {
                cur_plane[i] = b;
            }
        }

        for (int word_idx = 0; word_idx < L1_SPK_WORDS; word_idx++) {
            uint32_t mask = s_spk1[word_idx];
            if (mask == 0) continue;

            while (mask) {
                int bit = __builtin_ctz(mask);
                mask &= (mask - 1);
                int idx = (word_idx << 5) + bit;

                int ich = idx / (POOL1_OUT_H * POOL1_OUT_W);
                int rem = idx % (POOL1_OUT_H * POOL1_OUT_W);
                int iy = rem / POOL1_OUT_W;
                int ix = rem % POOL1_OUT_W;

                for (int ky = 0; ky < 3; ky++) {
                    int y = iy - ky + 1;
                    if (y < 0 || y >= 16) continue;
                    if (ix >= 1 && ix <= 14) {
                        int out_base = y * 16 + (ix - 1);
                        for (int och = 0; och < CONV2_OUT_CHANNELS; och++) {
                            const int8_t* w_row = &conv2_weights[((och * 16 + ich) * 3 + ky) * 3];
                            snn_simd_acc_row3_rev_i8_to_i32(&s_scratch_cur[och * (16 * 16) + out_base], w_row);
                        }
                        active_sops += 3 * CONV2_OUT_CHANNELS;
                    } else {
                        for (int kx = 0; kx < 3; kx++) {
                            int x = ix - kx + 1;
                            if (x < 0 || x >= 16) continue;

                            int out_offset = y * 16 + x;
                            for (int och = 0; och < CONV2_OUT_CHANNELS; och++) {
                                int8_t w = conv2_weights[((och * 16 + ich) * 3 + ky) * 3 + kx];
                                s_scratch_cur[och * (16 * 16) + out_offset] += w;
                                active_sops++;
                            }
                        }
                    }
                }
            }
        }

        /* Camada 2: MaxPool2d(2) fundido ao Leaky LIF (32 x 8 x 8) com Bit-Packing */
        spk_clear_all(s_spk2, L2_SPK_WORDS);
        for (int och = 0; och < CONV2_OUT_CHANNELS; och++) {
            int32_t vth = vth_l2(och);
            const int32_t* cur_plane = &s_scratch_cur[och * (16 * 16)];
            for (int py = 0; py < POOL2_OUT_H; py++) {
                int y0 = py * 2;
                for (int px = 0; px < POOL2_OUT_W; px++) {
                    int x0 = px * 2;
                    int32_t pool_val = cur_plane[y0 * 16 + x0];
                    int32_t v01 = cur_plane[y0 * 16 + (x0 + 1)];
                    if (v01 > pool_val) pool_val = v01;
                    int32_t v10 = cur_plane[(y0 + 1) * 16 + x0];
                    if (v10 > pool_val) pool_val = v10;
                    int32_t v11 = cur_plane[(y0 + 1) * 16 + (x0 + 1)];
                    if (v11 > pool_val) pool_val = v11;

                    int32_t m = s_mem2[och][py][px];
                    int32_t r = (m >= vth) ? 1 : 0;
                    m = lif_decay_32(m) + pool_val - r * vth;
                    if (m >= vth) {
                        int idx = (och * POOL2_OUT_H + py) * POOL2_OUT_W + px;
                        spk_set(s_spk2, idx);
                        total_spks_l2++;
                    }
                    s_mem2[och][py][px] = m;
                }
            }
        }

        /* --- CAMADA 3: Convolução Esparsa Orientada a Eventos (SOP) com __builtin_ctz --- */
        for (int och = 0; och < CONV3_OUT_CHANNELS; och++) {
            int32_t b = conv3_bias[och];
            int32_t* cur_plane = &s_scratch_cur[och * (8 * 8)];
            for (int i = 0; i < 8 * 8; i++) {
                cur_plane[i] = b;
            }
        }

        for (int word_idx = 0; word_idx < L2_SPK_WORDS; word_idx++) {
            uint32_t mask = s_spk2[word_idx];
            if (mask == 0) continue;

            while (mask) {
                int bit = __builtin_ctz(mask);
                mask &= (mask - 1);
                int idx = (word_idx << 5) + bit;

                int ich = idx / (POOL2_OUT_H * POOL2_OUT_W);
                int rem = idx % (POOL2_OUT_H * POOL2_OUT_W);
                int iy = rem / POOL2_OUT_W;
                int ix = rem % POOL2_OUT_W;

                for (int ky = 0; ky < 3; ky++) {
                    int y = iy - ky + 1;
                    if (y < 0 || y >= 8) continue;
                    if (ix >= 1 && ix <= 6) {
                        int out_base = y * 8 + (ix - 1);
                        for (int och = 0; och < CONV3_OUT_CHANNELS; och++) {
                            const int8_t* w_row = &conv3_weights[((och * 32 + ich) * 3 + ky) * 3];
                            snn_simd_acc_row3_rev_i8_to_i32(&s_scratch_cur[och * (8 * 8) + out_base], w_row);
                        }
                        active_sops += 3 * CONV3_OUT_CHANNELS;
                    } else {
                        for (int kx = 0; kx < 3; kx++) {
                            int x = ix - kx + 1;
                            if (x < 0 || x >= 8) continue;

                            int out_offset = y * 8 + x;
                            for (int och = 0; och < CONV3_OUT_CHANNELS; och++) {
                                int8_t w = conv3_weights[((och * 32 + ich) * 3 + ky) * 3 + kx];
                                s_scratch_cur[och * (8 * 8) + out_offset] += w;
                                active_sops++;
                            }
                        }
                    }
                }
            }
        }

        /* Camada 3: MaxPool2d(2) fundido ao Leaky LIF (32 x 4 x 4) com Bit-Packing */
        spk_clear_all(s_spk3, L3_SPK_WORDS);
        for (int och = 0; och < CONV3_OUT_CHANNELS; och++) {
            int32_t vth = vth_l3(och);
            const int32_t* cur_plane = &s_scratch_cur[och * (8 * 8)];
            for (int py = 0; py < POOL3_OUT_H; py++) {
                int y0 = py * 2;
                for (int px = 0; px < POOL3_OUT_W; px++) {
                    int x0 = px * 2;
                    int32_t pool_val = cur_plane[y0 * 8 + x0];
                    int32_t v01 = cur_plane[y0 * 8 + (x0 + 1)];
                    if (v01 > pool_val) pool_val = v01;
                    int32_t v10 = cur_plane[(y0 + 1) * 8 + x0];
                    if (v10 > pool_val) pool_val = v10;
                    int32_t v11 = cur_plane[(y0 + 1) * 8 + (x0 + 1)];
                    if (v11 > pool_val) pool_val = v11;

                    int32_t m = s_mem3[och][py][px];
                    int32_t r = (m >= vth) ? 1 : 0;
                    m = lif_decay_32(m) + pool_val - r * vth;
                    if (m >= vth) {
                        int idx = (och * POOL3_OUT_H + py) * POOL3_OUT_W + px;
                        spk_set(s_spk3, idx);
                        total_spks_l3++;
                    }
                    s_mem3[och][py][px] = m;
                }
            }
        }

        /* --- CAMADA 4: Head Linear (FC_IN_FEATURES -> FC_OUT_FEATURES) com Bit-Packing ---
         * Itera palavra a palavra sobre s_spk3, pulando blocos de 32 neurônios silenciosos.
         */
        int32_t cur4[FC_OUT_FEATURES];
        for (int c = 0; c < FC_OUT_FEATURES; ++c) {
            cur4[c] = fc_bias[c];
        }

        for (int word_idx = 0; word_idx < L3_SPK_WORDS; ++word_idx) {
            uint32_t mask = s_spk3[word_idx];
            if (mask == 0) continue;

            while (mask) {
                int bit = __builtin_ctz(mask);
                mask &= (mask - 1);
                int idx = (word_idx << 5) + bit;

                int8_t w_vec[FC_OUT_FEATURES];
                for (int c = 0; c < FC_OUT_FEATURES; ++c) {
                    w_vec[c] = fc_weights[c * FC_IN_FEATURES + idx];
                }
                snn_simd_acc_vec_i8_to_i32(cur4, w_vec, FC_OUT_FEATURES);
                active_sops += FC_OUT_FEATURES;
            }
        }

        /* Camada 4: Leaky LIF (FC_OUT_FEATURES Classes) */
        for (int c = 0; c < FC_OUT_FEATURES; c++) {
            int32_t vth = vth_fc(c);
            int32_t m = s_mem4[c];
            int32_t r = (m >= vth) ? 1 : 0;
            m = lif_decay_32(m) + cur4[c] - r * vth;
            uint8_t s = (m >= vth) ? 1 : 0;
            s_mem4[c] = m;
            s_spk_counts[c] += s;
            total_spks_l4 += s;
        }

        if (should_early_exit(timesteps_executed, SPIKEVISION_TIMESTEPS, s_spk_counts, early_exit_cfg)) {
            early_exited = true;
            break;
        }
    }

    /* 5. Decisão de Classificação: Maior contagem de spikes (desempate por maior potencial final de membrana) */
    for (int c = 0; c < FC_OUT_FEATURES; ++c) {
        result->spike_counts[c] = s_spk_counts[c];
        result->final_mem[c]    = s_mem4[c];
    }
    int best_c = 0;
    int32_t max_spk = s_spk_counts[0];
    int32_t best_mem = s_mem4[0];
    for (int c = 1; c < FC_OUT_FEATURES; ++c) {
        if (s_spk_counts[c] > max_spk ||
            (s_spk_counts[c] == max_spk && s_mem4[c] > best_mem)) {
            max_spk  = s_spk_counts[c];
            best_mem = s_mem4[c];
            best_c   = c;
        }
    }
    result->predicted_class = best_c;

    /* 6. Medição final de tempo e ciclos */
#if defined(ESP_PLATFORM)
    result->latency_us = esp_timer_get_time() - t_start;
    result->cpu_cycles = esp_cpu_get_cycle_count() - cycles_start;
#else
    auto t_end = std::chrono::high_resolution_clock::now();
    result->latency_us = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count();
    result->cpu_cycles = (uint32_t)(result->latency_us * 240); // Estimativa para 240 MHz
#endif

    /* 7. Preenchimento da Estrutura de Retorno */
    result->total_spikes_l1 = total_spks_l1;
    result->total_spikes_l2 = total_spks_l2;
    result->total_spikes_l3 = total_spks_l3;
    result->total_spikes_l4 = total_spks_l4;
    result->total_spikes = total_spks_l1 + total_spks_l2 + total_spks_l3 + total_spks_l4;

    /* Total de avaliações de neurônios nos timesteps efetivamente executados */
    const uint32_t total_neurons = (6656 + FC_OUT_FEATURES) * (uint32_t)timesteps_executed;
    result->sparsity_percent = 100.0f * (1.0f - ((float)result->total_spikes / (float)total_neurons));

    /* Total de MACs densos equivalentes:
     * L1: 16 * 32 * 32 * (3 * 3 * 3) = 442.368 (feita 1x)
     * L2: T * (32 * 16 * 16) * (16 * 3 * 3) = 9.437.184
     * L3: T * (32 * 8 * 8) * (32 * 3 * 3) = 4.718.592
     * L4: T * FC_IN_FEATURES * FC_OUT_FEATURES
     * Camadas 1..3 = 14.598.144
     */
    const uint64_t dense_macs = 14598144ULL + ((uint64_t)SPIKEVISION_TIMESTEPS * FC_IN_FEATURES * FC_OUT_FEATURES);
    result->dense_macs = dense_macs;
    result->active_sops = active_sops;

    /* Redução de operações: consideramos que em L1 foram 442.368 MACs e nas demais apenas active_sops */
    uint64_t total_ops = active_sops + 442368ULL;
    result->sop_reduction_percent = 100.0f * (1.0f - ((float)total_ops / (float)dense_macs));
    result->timesteps_executed = timesteps_executed;
    result->early_exited = early_exited;
}
