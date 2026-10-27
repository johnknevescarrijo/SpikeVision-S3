/**
 * SpikeVision-S3: Motor de Inferência Neuromórfica em C/C++ para ESP32-S3.
 *
 * Implementa execução de redes neurais pulsadas (CSNN) quantizadas em Int8/Int32
 * com aritmética bit-accurate ao simulador Python, orientada a eventos (SOPs)
 * e alocação 100% estática na SRAM interna.
 */

#include "snn_engine.h"
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

/* Camada 1: Entrada 3x32x32 -> Conv 16x32x32 -> MaxPool 16x16x16 */
SPIKEVISION_ALIGN16 static int32_t s_cur1[CONV1_OUT_CHANNELS][32][32];
SPIKEVISION_ALIGN16 static int32_t s_pool1[CONV1_OUT_CHANNELS][POOL1_OUT_H][POOL1_OUT_W];
SPIKEVISION_ALIGN16 static int32_t s_mem1[CONV1_OUT_CHANNELS][POOL1_OUT_H][POOL1_OUT_W];
SPIKEVISION_ALIGN16 static uint8_t s_spk1[CONV1_OUT_CHANNELS][POOL1_OUT_H][POOL1_OUT_W];

/* Camada 2: Conv 32x16x16 -> MaxPool 32x8x8 */
SPIKEVISION_ALIGN16 static int32_t s_cur2[CONV2_OUT_CHANNELS][16][16];
SPIKEVISION_ALIGN16 static int32_t s_pool2[CONV2_OUT_CHANNELS][POOL2_OUT_H][POOL2_OUT_W];
SPIKEVISION_ALIGN16 static int32_t s_mem2[CONV2_OUT_CHANNELS][POOL2_OUT_H][POOL2_OUT_W];
SPIKEVISION_ALIGN16 static uint8_t s_spk2[CONV2_OUT_CHANNELS][POOL2_OUT_H][POOL2_OUT_W];

/* Camada 3: Conv 32x8x8 -> MaxPool 32x4x4 */
SPIKEVISION_ALIGN16 static int32_t s_cur3[CONV3_OUT_CHANNELS][8][8];
SPIKEVISION_ALIGN16 static int32_t s_pool3[CONV3_OUT_CHANNELS][POOL3_OUT_H][POOL3_OUT_W];
SPIKEVISION_ALIGN16 static int32_t s_mem3[CONV3_OUT_CHANNELS][POOL3_OUT_H][POOL3_OUT_W];
SPIKEVISION_ALIGN16 static uint8_t s_spk3[CONV3_OUT_CHANNELS][POOL3_OUT_H][POOL3_OUT_W];

/* Camada 4: Linear Head 512 -> 2 */
SPIKEVISION_ALIGN16 static int32_t s_mem4[FC_OUT_FEATURES];
SPIKEVISION_ALIGN16 static int32_t s_spk_counts[FC_OUT_FEATURES];

/* =========================================================================
 * Dinâmica Leaky LIF em Ponto Fixo 16-bit
 * ========================================================================= */

/**
 * Aplica decaimento com aritmética inteira nativa de hardware de 32 bits.
 * Reproduz bit-a-bit os registradores da CPU Xtensa do ESP32-S3.
 */
static inline int32_t lif_decay_32(int32_t mem) {
    int32_t prod = (int32_t)((uint32_t)mem * (uint32_t)SPIKEVISION_BETA_NUM);
    return prod >> SPIKEVISION_BETA_SHIFT;
}

/* =========================================================================
 * API do Motor Neuromórfico
 * ========================================================================= */

void snn_engine_init(void) {
    memset(s_cur1, 0, sizeof(s_cur1));
    memset(s_pool1, 0, sizeof(s_pool1));
    memset(s_mem1, 0, sizeof(s_mem1));
    memset(s_spk1, 0, sizeof(s_spk1));

    memset(s_cur2, 0, sizeof(s_cur2));
    memset(s_pool2, 0, sizeof(s_pool2));
    memset(s_mem2, 0, sizeof(s_mem2));
    memset(s_spk2, 0, sizeof(s_spk2));

    memset(s_cur3, 0, sizeof(s_cur3));
    memset(s_pool3, 0, sizeof(s_pool3));
    memset(s_mem3, 0, sizeof(s_mem3));
    memset(s_spk3, 0, sizeof(s_spk3));

    memset(s_mem4, 0, sizeof(s_mem4));
    memset(s_spk_counts, 0, sizeof(s_spk_counts));
}

size_t snn_engine_get_sram_usage(void) {
    return sizeof(s_cur1) + sizeof(s_pool1) + sizeof(s_mem1) + sizeof(s_spk1) +
           sizeof(s_cur2) + sizeof(s_pool2) + sizeof(s_mem2) + sizeof(s_spk2) +
           sizeof(s_cur3) + sizeof(s_pool3) + sizeof(s_mem3) + sizeof(s_spk3) +
           sizeof(s_mem4) + sizeof(s_spk_counts);
}

void snn_engine_infer(const uint8_t* image_chw, SNNInferenceResult* result) {
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

    /* 3. CAMADA 1: Direct Coding Convolucional (Calculada 1x por Imagem)
     * Como a imagem de entrada é estática ao longo dos 8 timesteps, a convolução
     * 3x3 + bias + MaxPool2d é pré-calculada uma única vez, gerando economia de 8x!
     */
    for (int och = 0; och < CONV1_OUT_CHANNELS; och++) {
        int32_t bias = conv1_bias[och];
        for (int y = 0; y < 32; y++) {
            for (int x = 0; x < 32; x++) {
                int32_t acc = bias;
                for (int ich = 0; ich < CONV1_IN_CHANNELS; ich++) {
                    const uint8_t* in_plane = image_chw + ich * (32 * 32);
                    for (int ky = 0; ky < 3; ky++) {
                        int iy = y + ky - 1;
                        if (iy < 0 || iy >= 32) continue;
                        for (int kx = 0; kx < 3; kx++) {
                            int ix = x + kx - 1;
                            if (ix < 0 || ix >= 32) continue;

                            int8_t w = conv1_weights[((och * 3 + ich) * 3 + ky) * 3 + kx];
                            uint8_t px = in_plane[iy * 32 + ix];
                            acc += (int32_t)px * (int32_t)w;
                        }
                    }
                }
                s_cur1[och][y][x] = acc;
            }
        }
    }

    /* MaxPool2d(2, stride=2) Camada 1: 32x32 -> 16x16 */
    for (int och = 0; och < CONV1_OUT_CHANNELS; och++) {
        for (int py = 0; py < POOL1_OUT_H; py++) {
            for (int px = 0; px < POOL1_OUT_W; px++) {
                int y0 = py * 2;
                int x0 = px * 2;
                int32_t m = s_cur1[och][y0][x0];
                if (s_cur1[och][y0][x0 + 1] > m) m = s_cur1[och][y0][x0 + 1];
                if (s_cur1[och][y0 + 1][x0] > m) m = s_cur1[och][y0 + 1][x0];
                if (s_cur1[och][y0 + 1][x0 + 1] > m) m = s_cur1[och][y0 + 1][x0 + 1];
                s_pool1[och][py][px] = m;
            }
        }
    }

    /* 4. LOOP TEMPORAL NEUROMÓRFICO (T = 8 Timesteps) */
    for (int t = 0; t < SPIKEVISION_TIMESTEPS; t++) {

        /* --- CAMADA 1: Leaky LIF (16 x 16 x 16) --- */
        for (int och = 0; och < CONV1_OUT_CHANNELS; och++) {
            for (int y = 0; y < POOL1_OUT_H; y++) {
                for (int x = 0; x < POOL1_OUT_W; x++) {
                    int32_t m = s_mem1[och][y][x];
                    int32_t r = (m >= SPIKEVISION_L1_VTH) ? 1 : 0;
                    m = lif_decay_32(m) + s_pool1[och][y][x] - r * SPIKEVISION_L1_VTH;
                    uint8_t s = (m >= SPIKEVISION_L1_VTH) ? 1 : 0;
                    s_mem1[och][y][x] = m;
                    s_spk1[och][y][x] = s;
                    total_spks_l1 += s;
                }
            }
        }

        /* --- CAMADA 2: Convolução Esparsa Orientada a Eventos (SOP) ---
         * Se spk1 == 0 (repouso / silêncio sináptico), nada é computado.
         * Se spk1 == 1, realizamos exclusivamente adições inteiras (SOPs).
         */
        for (int och = 0; och < CONV2_OUT_CHANNELS; och++) {
            int32_t b = conv2_bias[och];
            for (int y = 0; y < 16; y++) {
                for (int x = 0; x < 16; x++) {
                    s_cur2[och][y][x] = b;
                }
            }
        }

        for (int ich = 0; ich < CONV2_IN_CHANNELS; ich++) {
            for (int iy = 0; iy < 16; iy++) {
                for (int ix = 0; ix < 16; ix++) {
                    if (s_spk1[ich][iy][ix] == 0) continue; // Salto neuromórfico!

                    for (int ky = 0; ky < 3; ky++) {
                        int y = iy - ky + 1;
                        if (y < 0 || y >= 16) continue;
                        for (int kx = 0; kx < 3; kx++) {
                            int x = ix - kx + 1;
                            if (x < 0 || x >= 16) continue;

                            for (int och = 0; och < CONV2_OUT_CHANNELS; och++) {
                                int8_t w = conv2_weights[((och * 16 + ich) * 3 + ky) * 3 + kx];
                                s_cur2[och][y][x] += w;
                                active_sops++;
                            }
                        }
                    }
                }
            }
        }

        /* MaxPool2d(2, stride=2) Camada 2: 16x16 -> 8x8 */
        for (int och = 0; och < CONV2_OUT_CHANNELS; och++) {
            for (int py = 0; py < POOL2_OUT_H; py++) {
                for (int px = 0; px < POOL2_OUT_W; px++) {
                    int y0 = py * 2;
                    int x0 = px * 2;
                    int32_t m = s_cur2[och][y0][x0];
                    if (s_cur2[och][y0][x0 + 1] > m) m = s_cur2[och][y0][x0 + 1];
                    if (s_cur2[och][y0 + 1][x0] > m) m = s_cur2[och][y0 + 1][x0];
                    if (s_cur2[och][y0 + 1][x0 + 1] > m) m = s_cur2[och][y0 + 1][x0 + 1];
                    s_pool2[och][py][px] = m;
                }
            }
        }

        /* Camada 2: Leaky LIF (32 x 8 x 8) */
        for (int och = 0; och < CONV2_OUT_CHANNELS; och++) {
            for (int y = 0; y < POOL2_OUT_H; y++) {
                for (int x = 0; x < POOL2_OUT_W; x++) {
                    int32_t m = s_mem2[och][y][x];
                    int32_t r = (m >= SPIKEVISION_L2_VTH) ? 1 : 0;
                    m = lif_decay_32(m) + s_pool2[och][y][x] - r * SPIKEVISION_L2_VTH;
                    uint8_t s = (m >= SPIKEVISION_L2_VTH) ? 1 : 0;
                    s_mem2[och][y][x] = m;
                    s_spk2[och][y][x] = s;
                    total_spks_l2 += s;
                }
            }
        }

        /* --- CAMADA 3: Convolução Esparsa Orientada a Eventos (SOP) --- */
        for (int och = 0; och < CONV3_OUT_CHANNELS; och++) {
            int32_t b = conv3_bias[och];
            for (int y = 0; y < 8; y++) {
                for (int x = 0; x < 8; x++) {
                    s_cur3[och][y][x] = b;
                }
            }
        }

        for (int ich = 0; ich < CONV3_IN_CHANNELS; ich++) {
            for (int iy = 0; iy < 8; iy++) {
                for (int ix = 0; ix < 8; ix++) {
                    if (s_spk2[ich][iy][ix] == 0) continue; // Salto neuromórfico!

                    for (int ky = 0; ky < 3; ky++) {
                        int y = iy - ky + 1;
                        if (y < 0 || y >= 8) continue;
                        for (int kx = 0; kx < 3; kx++) {
                            int x = ix - kx + 1;
                            if (x < 0 || x >= 8) continue;

                            for (int och = 0; och < CONV3_OUT_CHANNELS; och++) {
                                int8_t w = conv3_weights[((och * 32 + ich) * 3 + ky) * 3 + kx];
                                s_cur3[och][y][x] += w;
                                active_sops++;
                            }
                        }
                    }
                }
            }
        }

        /* MaxPool2d(2, stride=2) Camada 3: 8x8 -> 4x4 */
        for (int och = 0; och < CONV3_OUT_CHANNELS; och++) {
            for (int py = 0; py < POOL3_OUT_H; py++) {
                for (int px = 0; px < POOL3_OUT_W; px++) {
                    int y0 = py * 2;
                    int x0 = px * 2;
                    int32_t m = s_cur3[och][y0][x0];
                    if (s_cur3[och][y0][x0 + 1] > m) m = s_cur3[och][y0][x0 + 1];
                    if (s_cur3[och][y0 + 1][x0] > m) m = s_cur3[och][y0 + 1][x0];
                    if (s_cur3[och][y0 + 1][x0 + 1] > m) m = s_cur3[och][y0 + 1][x0 + 1];
                    s_pool3[och][py][px] = m;
                }
            }
        }

        /* Camada 3: Leaky LIF (32 x 4 x 4) */
        for (int och = 0; och < CONV3_OUT_CHANNELS; och++) {
            for (int y = 0; y < POOL3_OUT_H; y++) {
                for (int x = 0; x < POOL3_OUT_W; x++) {
                    int32_t m = s_mem3[och][y][x];
                    int32_t r = (m >= SPIKEVISION_L3_VTH) ? 1 : 0;
                    m = lif_decay_32(m) + s_pool3[och][y][x] - r * SPIKEVISION_L3_VTH;
                    uint8_t s = (m >= SPIKEVISION_L3_VTH) ? 1 : 0;
                    s_mem3[och][y][x] = m;
                    s_spk3[och][y][x] = s;
                    total_spks_l3 += s;
                }
            }
        }

        /* --- CAMADA 4: Head Linear (FC_IN_FEATURES -> FC_OUT_FEATURES) Orientada a Spikes ---
         * Itera sobre os FC_IN_FEATURES neurônios de saída de spk3 (formato CHW: [32, 4, 4]).
         * Apenas se o neurônio disparou spike (s==1), somamos os pesos correspondentes.
         */
        int32_t cur4[FC_OUT_FEATURES];
        for (int c = 0; c < FC_OUT_FEATURES; ++c) {
            cur4[c] = fc_bias[c];
        }

        int idx = 0;
        for (int ch = 0; ch < CONV3_OUT_CHANNELS; ++ch) {
            for (int y = 0; y < 4; ++y) {
                for (int x = 0; x < 4; ++x) {
                    if (s_spk3[ch][y][x]) {
                        for (int c = 0; c < FC_OUT_FEATURES; ++c) {
                            cur4[c] += (int32_t)fc_weights[c * FC_IN_FEATURES + idx];
                        }
                        active_sops += FC_OUT_FEATURES;
                    }
                    idx++;
                }
            }
        }

        /* Camada 4: Leaky LIF (FC_OUT_FEATURES Classes) */
        for (int c = 0; c < FC_OUT_FEATURES; c++) {
            int32_t m = s_mem4[c];
            int32_t r = (m >= SPIKEVISION_L4_VTH) ? 1 : 0;
            m = lif_decay_32(m) + cur4[c] - r * SPIKEVISION_L4_VTH;
            uint8_t s = (m >= SPIKEVISION_L4_VTH) ? 1 : 0;
            s_mem4[c] = m;
            s_spk_counts[c] += s;
            total_spks_l4 += s;
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

    /* Total de avaliações de neurônios:
     * L1: 4096 * T
     * L2: 2048 * T
     * L3:  512 * T
     * L4: FC_OUT_FEATURES * T
     * Total = (6656 + FC_OUT_FEATURES) * SPIKEVISION_TIMESTEPS
     */
    const uint32_t total_neurons = (6656 + FC_OUT_FEATURES) * SPIKEVISION_TIMESTEPS;
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
}
