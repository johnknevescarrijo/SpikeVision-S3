/**
 * SpikeVision-S3: Ponto de Entrada Principal (ESP-IDF + Host Linux)
 *
 * Executa a bateria de testes de validação neuromórfica e paridade bit-a-bit
 * sobre as 10 amostras reais do dataset CIFAR-10.
 */

#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdbool.h>
#include "snn_engine.h"
#include "snn_pipeline.h"
#include "image_preprocess.h"
#include "camera_capture.h"
#include "dvs_encoder.h"
#include "test_samples.h"

#if defined(ESP_PLATFORM)
    #include "freertos/FreeRTOS.h"
    #include "freertos/task.h"
    #include "esp_system.h"
    #include "esp_chip_info.h"
    #include "esp_heap_caps.h"
    #include "esp_log.h"

    static const char* TAG = "SPIKEVISION_S3";
#endif

static void print_banner(void) {
    puts(
        "================================================================================\n"
        "       ____             _   __        ___        _                ____ ____     \n"
        "      / ___|  _ __     (_) | | _____ / \\ \\      (_)___  ___  _ __ / ___|___ \\   \n"
        "      \\___ \\ | '_ \\    | | | |/ / _ \\   / \\     | / __|/ _ \\| '_ \\ \\___ \\ __) |  \n"
        "       ___) || |_) |   | | |   <  __/  / _ \\    | \\__ \\ (_) | | | | ___) / __/   \n"
        "      |____/ | .__/    |_| |_|\\_\\___| /_/ \\_\\   |_|___/\\___/|_| |_||____/_____|  \n"
        "             |_|                                                                \n"
        "         Neuromorphic Event-Driven SNN Engine for Cat/Non-Cat Classification    \n"
        "================================================================================"
    );
}

static void print_system_info(void) {
#if defined(ESP_PLATFORM)
    esp_chip_info_t chip_info;
    esp_chip_info(&chip_info);

    printf("🔧 [ESP32-S3 Hardware Info]:\n");
    printf("  • Modelo do Chip:       ESP32-S3 (Revisão %d)\n", chip_info.revision);
    printf("  • Núcleos de CPU:       %d Cores Xtensa LX7 @ 240 MHz\n", chip_info.cores);
    printf("  • Recursos:             %s%s%s\n",
           (chip_info.features & CHIP_FEATURE_WIFI_BGN) ? "Wi-Fi " : "",
           (chip_info.features & CHIP_FEATURE_BLE) ? "BLE " : "",
           (chip_info.features & CHIP_FEATURE_EMB_FLASH) ? "Embedded Flash" : "External Flash");
    printf("  • Heap Interno Livre:   %" PRIu32 " bytes\n", (uint32_t)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
    printf("  • Heap Total Livre:     %" PRIu32 " bytes\n", (uint32_t)esp_get_free_heap_size());
#else
    printf("💻 [Host Linux Test Environment]:\n");
    printf("  • Plataforma:           Linux Host x86_64 / Native C++\n");
    printf("  • Modo de Operação:     Validação Bit-a-Bit Cruzada\n");
#endif
    const size_t sram_bytes = snn_engine_get_sram_bytes();
    printf("  • Backend SIMD:         %s\n", snn_engine_get_simd_backend());
    printf("  • Consumo SRAM Motor:   %zu bytes (%.2f KB) [100%% Estática, 0 Heap]\n",
           sram_bytes, (float)sram_bytes / 1024.0f);
    printf("  • Ocupação ROM Pesos:   15608 bytes (15.24 KB)\n\n");
}

static const char* get_class_name(int cls) {
#if (SPIKEVISION_NUM_CLASSES == 10)
    static const char* const kClassNames[10] = {
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    };
    if (cls >= 0 && cls < 10) return kClassNames[cls];
    return "unknown";
#else
    if (cls == 1) return "CAT";
    return "NON-CAT";
#endif
}

static int run_validation_suite(void) {
    printf("🚀 [Executando Bateria de Testes nas %d Amostras do CIFAR-10 (%d Classes)]...\n",
           TEST_SAMPLE_COUNT, SPIKEVISION_NUM_CLASSES);
    printf("--------------------------------------------------------------------------------------------------------------------\n");
    printf("| ID | Ground-Truth | Pred SNN     | Top Spikes | Top Mem  | Sparsity | SOPs Ativas | Latência  | Match PyTorch  |\n");
    printf("--------------------------------------------------------------------------------------------------------------------\n");

    int bit_accurate_matches = 0;
    int correct_predictions = 0;
    uint64_t total_latency_us = 0;
    uint64_t total_sops = 0;
    uint64_t total_dense_macs = 0;
    float total_sparsity = 0.0f;

    for (int i = 0; i < TEST_SAMPLE_COUNT; i++) {
        const SNNTestCase* tc = &TEST_SUITE[i];
        SNNInferenceResult result;

        snn_engine_infer(tc->image_data, &result);

        bool spk_match = true;
        bool mem_match = true;
        for (int c = 0; c < SPIKEVISION_NUM_CLASSES; ++c) {
            if (result.spike_counts[c] != tc->expected_spikes[c]) {
                spk_match = false;
            }
            if (result.final_mem[c] != tc->expected_mem[c]) {
                mem_match = false;
            }
        }
        bool pred_match = (result.predicted_class == tc->expected_prediction);
        bool bit_accurate = spk_match && mem_match && pred_match;

        if (bit_accurate) bit_accurate_matches++;
        if (result.predicted_class == tc->ground_truth) correct_predictions++;

        total_latency_us += result.latency_us;
        total_sops += result.active_sops;
        total_dense_macs += result.dense_macs;
        total_sparsity += result.sparsity_percent;

        const char* gt_str = get_class_name(tc->ground_truth);
        const char* pred_str = get_class_name(result.predicted_class);
        const char* status_str = bit_accurate ? "\033[32mPASS (BIT-ACC)\033[0m" : "\033[31mFAIL          \033[0m";

        int32_t top_spk = result.spike_counts[result.predicted_class];
        int32_t top_mem = result.final_mem[result.predicted_class];

        printf("| %02d | %-12s | %-12s |    %4" PRId32 "    |  %7" PRId32 " |  %5.1f%%  | %10" PRIu64 "  | %7.2f ms | %-14s |\n",
               i,
               gt_str,
               pred_str,
               top_spk,
               top_mem,
               result.sparsity_percent,
               result.active_sops,
               (float)result.latency_us / 1000.0f,
               status_str);
    }
    printf("--------------------------------------------------------------------------------------------------------------------\n\n");

    /* Relatório Consolidado */
    float avg_latency_ms = ((float)total_latency_us / (float)TEST_SAMPLE_COUNT) / 1000.0f;
    float fps = (avg_latency_ms > 0.0f) ? (1000.0f / avg_latency_ms) : 0.0f;
    float avg_sparsity = total_sparsity / (float)TEST_SAMPLE_COUNT;
    float avg_sops = (float)total_sops / (float)TEST_SAMPLE_COUNT;
    float avg_dense_macs = (float)total_dense_macs / (float)TEST_SAMPLE_COUNT;
    float reduction_percent = 100.0f * (1.0f - ((avg_sops + 442368.0f) / avg_dense_macs));

    printf("📊 [Resumo de Benchmark e Validação Cruzada]:\n");
    printf("  • Paridade Bit-a-Bit C++/Python:   %d/%d (%d%%)\n",
           bit_accurate_matches, TEST_SAMPLE_COUNT, (bit_accurate_matches * 100 / TEST_SAMPLE_COUNT));
    printf("  • Acurácia nas Amostras CIFAR-10:  %d/%d (%d%%)\n",
           correct_predictions, TEST_SAMPLE_COUNT, (correct_predictions * 100 / TEST_SAMPLE_COUNT));
    printf("  • Latência Média por Inferência:   %.2f ms (%.1f FPS)\n", avg_latency_ms, fps);
    printf("  • Esparsidade Média da Rede:       %.2f%% (Neurônios em Silêncio)\n", avg_sparsity);
    printf("  • Operações Sinápticas (SOPs):     %.0f SOPs (Média por Inferência)\n", avg_sops);
    printf("  • Operações Dense MAC Equivalente: %.0f MACs\n", avg_dense_macs);
    printf("  • Redução de Carga Computacional:  %.2f%% de Economia de Hardware!\n", reduction_percent);
    const size_t sram_bytes = snn_engine_get_sram_bytes();
    printf("  • Memória SRAM Interna Motor:      %zu bytes (%.2f KB) [100%% Estática]\n\n",
           sram_bytes, (float)sram_bytes / 1024.0f);

    if (bit_accurate_matches == TEST_SAMPLE_COUNT) {
        printf("\033[32m✔ SUCESSO ABSOLUTO: Todas as %d amostras apresentaram 100%% de paridade bit-a-bit!\033[0m\n\n",
               TEST_SAMPLE_COUNT);
        return 0;
    } else {
        printf("\033[31m✘ FALHA: Divergência detectada em %d amostras.\033[0m\n\n",
               TEST_SAMPLE_COUNT - bit_accurate_matches);
        return 1;
    }
}

static int run_early_exit_benchmark(void) {
    struct EarlyExitMode {
        const char* label;
        snn_early_exit_cfg_t cfg;
        bool require_lossless;
    };

    const EarlyExitMode modes[] = {
        { "Lossless (margin=0, min_t=2)",   { true, 2, 0 }, true },
        { "Agressivo (margin=2, min_t=2)",  { true, 2, 2 }, false },
    };

    /* Baseline full-T SOPs para cálculo exato de economia */
    uint64_t baseline_sops = 0;
    for (int i = 0; i < TEST_SAMPLE_COUNT; i++) {
        SNNInferenceResult base_res;
        snn_engine_infer(TEST_SUITE[i].image_data, &base_res);
        baseline_sops += base_res.active_sops;
    }

    printf("⚡ [Benchmark de Early-Exit Temporal Adaptativo (%d Amostras)]:\n", TEST_SAMPLE_COUNT);
    printf("--------------------------------------------------------------------------------------------------------------\n");
    printf("| Modo                          | t_exit Médio | Econ. Timesteps | Econ. SOPs | Latência Média | Concordância |\n");
    printf("--------------------------------------------------------------------------------------------------------------\n");

    int status = 0;
    for (size_t m = 0; m < sizeof(modes) / sizeof(modes[0]); m++) {
        uint64_t total_t = 0;
        uint64_t total_sops = 0;
        uint64_t total_lat_us = 0;
        int agreements = 0;

        for (int i = 0; i < TEST_SAMPLE_COUNT; i++) {
            const SNNTestCase* tc = &TEST_SUITE[i];
            SNNInferenceResult res;
            snn_engine_infer_adaptive(tc->image_data, &res, &modes[m].cfg);

            total_t += (uint64_t)res.timesteps_executed;
            total_sops += res.active_sops;
            total_lat_us += res.latency_us;
            if (res.predicted_class == tc->expected_prediction) {
                agreements++;
            }
        }

        float avg_t = (float)total_t / (float)TEST_SAMPLE_COUNT;
        float t_saving = 100.0f * (1.0f - (avg_t / (float)SPIKEVISION_TIMESTEPS));
        float sop_saving = (baseline_sops > 0)
            ? 100.0f * (1.0f - ((float)total_sops / (float)baseline_sops))
            : 0.0f;
        float avg_lat_ms = ((float)total_lat_us / (float)TEST_SAMPLE_COUNT) / 1000.0f;
        float agreement_pct = 100.0f * ((float)agreements / (float)TEST_SAMPLE_COUNT);

        printf("| %-29s | %5.2f / %d    |     %6.2f%%     |   %6.2f%%  |    %6.2f ms   |    %6.2f%%   |\n",
               modes[m].label,
               avg_t,
               SPIKEVISION_TIMESTEPS,
               t_saving,
               sop_saving,
               avg_lat_ms,
               agreement_pct);

        if (modes[m].require_lossless && agreements != TEST_SAMPLE_COUNT) {
            status = 1;
        }
    }
    printf("--------------------------------------------------------------------------------------------------------------\n\n");
    return status;
}

static int run_pipeline_validation(void) {
    printf("🔄 [Validação do Pipeline Assíncrono (Dual-Core / Worker Thread) - %d Amostras]:\n", TEST_SAMPLE_COUNT);

    uint64_t expected_sops[TEST_SAMPLE_COUNT];
    for (int i = 0; i < TEST_SAMPLE_COUNT; i++) {
        SNNInferenceResult sync_res;
        snn_engine_infer(TEST_SUITE[i].image_data, &sync_res);
        expected_sops[i] = sync_res.active_sops;
    }

    if (!snn_pipeline_start(nullptr)) {
        printf("\033[31m✘ FALHA: Não foi possível iniciar o pipeline assíncrono.\033[0m\n\n");
        return 1;
    }

    int matches = 0;
    int submitted = 0;
    int received = 0;

    /* Preenche até a profundidade da fila e drena em pipeline para exercitar concorrência */
    while (received < TEST_SAMPLE_COUNT) {
        while (submitted < TEST_SAMPLE_COUNT && (submitted - received) < SNN_PIPELINE_QUEUE_DEPTH) {
            if (!snn_pipeline_submit((uint32_t)submitted, (const int8_t*)TEST_SUITE[submitted].image_data, 5000)) {
                printf("\033[31m✘ FALHA: Timeout ao submeter frame %d no pipeline.\033[0m\n", submitted);
                snn_pipeline_stop();
                return 1;
            }
            submitted++;
        }

        snn_pipeline_output_t out;
        if (!snn_pipeline_receive(&out, 5000)) {
            printf("\033[31m✘ FALHA: Timeout ao receber frame %d do pipeline.\033[0m\n", received);
            snn_pipeline_stop();
            return 1;
        }

        const SNNTestCase* tc = &TEST_SUITE[received];
        bool ok = (out.frame_id == (uint32_t)received) &&
                  (out.result.predicted_class == tc->expected_prediction) &&
                  (out.result.active_sops == expected_sops[received]);
        for (int c = 0; c < SPIKEVISION_NUM_CLASSES; ++c) {
            if (out.result.spike_counts[c] != tc->expected_spikes[c] ||
                out.result.final_mem[c] != tc->expected_mem[c]) {
                ok = false;
            }
        }
        if (ok) {
            matches++;
        }
        received++;
    }

    snn_pipeline_stop();

    if (matches == TEST_SAMPLE_COUNT) {
        printf("  • Paridade Assíncrona (frame_id, pred, spikes, SOPs): \033[32m%d/%d (100%%)\033[0m\n\n",
               matches, TEST_SAMPLE_COUNT);
        return 0;
    }

    printf("  • Paridade Assíncrona: \033[31m%d/%d FALHA\033[0m\n\n", matches, TEST_SAMPLE_COUNT);
    return 1;
}

static bool verify_solid_chw32(const int8_t* chw, int8_t exp_r, int8_t exp_g, int8_t exp_b) {
    for (int i = 0; i < IMG_PREPROC_PLANE; ++i) {
        if (chw[i] < 0 ||
            chw[IMG_PREPROC_PLANE + i] < 0 ||
            chw[2 * IMG_PREPROC_PLANE + i] < 0) {
            return false;
        }
        if (chw[i] != exp_r ||
            chw[IMG_PREPROC_PLANE + i] != exp_g ||
            chw[2 * IMG_PREPROC_PLANE + i] != exp_b) {
            return false;
        }
    }
    return true;
}

static int run_camera_preprocess_validation(void) {
    printf("📷 [Validação de Câmera & Pré-Processamento RGB -> 32x32 CHW [0..127]]:\n");
    printf("  • Driver Hardware esp_camera: %s\n",
           camera_capture_has_hardware_driver() ? "Disponível (ESP-IDF)" : "Indisponível (Fallback Sintético CI)");

    static uint16_t s_rgb565_buf[160 * 120];
    static uint8_t  s_rgb888_buf[96 * 96 * 3];
    static uint8_t  s_gray8_buf[96 * 96];
    static int8_t   s_chw32[IMG_PREPROC_BYTES];

    int checks_passed = 0;
    int checks_total = 0;

    /* 1. Cores puras R, G, B, Branco e Preto em RGB565 (com e sem swap_bytes) */
    struct ColorCase565 {
        uint16_t px;
        int8_t r;
        int8_t g;
        int8_t b;
    };
    const ColorCase565 cases565[] = {
        { 0xF800u, 127,   0,   0 }, /* Vermelho puro */
        { 0x07E0u,   0, 127,   0 }, /* Verde puro    */
        { 0x001Fu,   0,   0, 127 }, /* Azul puro     */
        { 0xFFFFu, 127, 127, 127 }, /* Branco        */
        { 0x0000u,   0,   0,   0 }, /* Preto         */
    };

    bool rgb565_ok = true;
    for (size_t c = 0; c < sizeof(cases565) / sizeof(cases565[0]); ++c) {
        /* 96x96 -> 32x32 sem swap */
        for (int i = 0; i < 96 * 96; ++i) {
            s_rgb565_buf[i] = cases565[c].px;
        }
        img_rgb565_to_chw32(s_rgb565_buf, 96, 96, false, s_chw32);
        if (!verify_solid_chw32(s_chw32, cases565[c].r, cases565[c].g, cases565[c].b)) {
            rgb565_ok = false;
        }

        /* 96x96 -> 32x32 com swap_bytes=true (bytes invertidos na entrada) */
        const uint16_t swapped_px = static_cast<uint16_t>((cases565[c].px >> 8) | (cases565[c].px << 8));
        for (int i = 0; i < 96 * 96; ++i) {
            s_rgb565_buf[i] = swapped_px;
        }
        img_rgb565_to_chw32(s_rgb565_buf, 96, 96, true, s_chw32);
        if (!verify_solid_chw32(s_chw32, cases565[c].r, cases565[c].g, cases565[c].b)) {
            rgb565_ok = false;
        }
    }
    checks_total++;
    if (rgb565_ok) checks_passed++;

    /* 2. Cores puras R, G, B, Branco e Preto em RGB888 (96x96 -> 32x32) */
    struct ColorCase888 {
        uint8_t r8, g8, b8;
        int8_t  r,  g,  b;
    };
    const ColorCase888 cases888[] = {
        { 255,   0,   0, 127,   0,   0 },
        {   0, 255,   0,   0, 127,   0 },
        {   0,   0, 255,   0,   0, 127 },
        { 255, 255, 255, 127, 127, 127 },
        {   0,   0,   0,   0,   0,   0 },
    };

    bool rgb888_ok = true;
    for (size_t c = 0; c < sizeof(cases888) / sizeof(cases888[0]); ++c) {
        for (int i = 0; i < 96 * 96; ++i) {
            s_rgb888_buf[i * 3 + 0] = cases888[c].r8;
            s_rgb888_buf[i * 3 + 1] = cases888[c].g8;
            s_rgb888_buf[i * 3 + 2] = cases888[c].b8;
        }
        img_rgb888_to_chw32(s_rgb888_buf, 96, 96, s_chw32);
        if (!verify_solid_chw32(s_chw32, cases888[c].r, cases888[c].g, cases888[c].b)) {
            rgb888_ok = false;
        }
    }
    checks_total++;
    if (rgb888_ok) checks_passed++;

    /* 3. Gray8 (96x96 -> 32x32) replicado nos 3 canais CHW */
    for (int i = 0; i < 96 * 96; ++i) {
        s_gray8_buf[i] = 255;
    }
    img_gray8_to_chw32(s_gray8_buf, 96, 96, s_chw32);
    checks_total++;
    if (verify_solid_chw32(s_chw32, 127, 127, 127)) checks_passed++;

    /* 4. Center-crop + box average em 160x120 -> 32x32 (bordas laterais vermelhas descartadas, centro verde preservado) */
    for (int y = 0; y < 120; ++y) {
        for (int x = 0; x < 160; ++x) {
            const bool in_center = (x >= 20 && x < 140);
            s_rgb565_buf[y * 160 + x] = in_center ? 0x07E0u : 0xF800u;
        }
    }
    img_rgb565_to_chw32(s_rgb565_buf, 160, 120, false, s_chw32);
    checks_total++;
    if (verify_solid_chw32(s_chw32, 0, 127, 0)) checks_passed++;

    /* 5. Fluxo completo camera_capture_grab_chw32 -> snn_engine_infer */
    camera_capture_cfg_t cam_cfg = camera_capture_default_cfg(CAM_PRESET_ESP32S3_EYE);
    cam_cfg.enable_synthetic_fallback = true;
    uint32_t capture_us = 0;
    bool e2e_ok = false;

    if (camera_capture_init(&cam_cfg)) {
        if (camera_capture_grab_chw32(s_chw32, &capture_us)) {
            bool bounds_ok = true;
            for (int i = 0; i < IMG_PREPROC_BYTES; ++i) {
                if (s_chw32[i] < 0) {
                    bounds_ok = false;
                    break;
                }
            }
            if (bounds_ok) {
                SNNInferenceResult cam_res;
                snn_engine_infer(reinterpret_cast<const uint8_t*>(s_chw32), &cam_res);
                if (cam_res.predicted_class >= 0 &&
                    cam_res.predicted_class < SPIKEVISION_NUM_CLASSES &&
                    cam_res.timesteps_executed == SPIKEVISION_TIMESTEPS) {
                    e2e_ok = true;
                }
            }
        }
        camera_capture_deinit();
    }
    checks_total++;
    if (e2e_ok) checks_passed++;

    if (checks_passed == checks_total) {
        printf("  • Testes RGB565/RGB888/Gray8 + Center-Crop 160x120 + Captura->SNN: \033[32m%d/%d PASS\033[0m (cap=%" PRIu32 " us)\n\n",
               checks_passed, checks_total, capture_us);
        return 0;
    }

    printf("  • Testes Câmera & Pré-processamento: \033[31m%d/%d FALHA\033[0m\n\n", checks_passed, checks_total);
    return 1;
}

static int run_dvs_encoder_validation(void) {
    printf("⚡ [Validação Pseudo-DVS ΔFrame Encoder (Zero Malloc)]:\n");

    static dvs_encoder_state_t dvs_state;
    static int8_t frame_a[DVS_FRAME_BYTES];
    static int8_t frame_b[DVS_FRAME_BYTES];
    static int8_t evt_out[DVS_FRAME_BYTES];

    /* Frame A: padrão base [0..127] */
    for (int i = 0; i < DVS_FRAME_BYTES; ++i) {
        frame_a[i] = static_cast<int8_t>((i * 7) % 64 + 20);
    }
    /* Frame B: metade superior com +30 (ON), parte inferior com -25 (OFF) */
    for (int i = 0; i < DVS_FRAME_BYTES; ++i) {
        if (i < 512) {
            frame_b[i] = static_cast<int8_t>(frame_a[i] + 30);
        } else if (i >= DVS_FRAME_PLANE && i < DVS_FRAME_PLANE + 512) {
            frame_b[i] = static_cast<int8_t>(frame_a[i] - 25);
        } else if (i >= 2 * DVS_FRAME_PLANE && i < 2 * DVS_FRAME_PLANE + 256) {
            frame_b[i] = static_cast<int8_t>(frame_a[i] + 3); /* Abaixo do threshold=8 */
        } else {
            frame_b[i] = frame_a[i];
        }
    }

    int checks_passed = 0;
    int checks_total = 0;
    dvs_frame_stats_t stats;

    /* 1. DVS_MODE_SIGNED_DELTA (binário): init -> estático -> movimento -> retorno estático */
    dvs_encoder_cfg_t cfg_sd = dvs_encoder_default_cfg(DVS_MODE_SIGNED_DELTA);
    dvs_encoder_init(&dvs_state, &cfg_sd);

    /* 1º frame (init) */
    dvs_encoder_process(&dvs_state, frame_a, evt_out, &stats);
    bool sd_init_ok = (stats.on_events == 0 && stats.off_events == 0 &&
                       stats.sparsity == 1.0f && stats.is_static_frame &&
                       stats.total_sites == DVS_FRAME_BYTES);
    for (int i = 0; i < DVS_FRAME_BYTES && sd_init_ok; ++i) {
        if (evt_out[i] != 0) sd_init_ok = false;
    }

    /* 2º frame idêntico -> estático */
    dvs_encoder_process(&dvs_state, frame_a, evt_out, &stats);
    bool sd_static_ok = (stats.on_events == 0 && stats.off_events == 0 &&
                         stats.sparsity == 1.0f && stats.is_static_frame);

    /* 3º frame com movimento -> 512 ON (+127), 512 OFF (-127), sub-threshold ignorado */
    dvs_encoder_process(&dvs_state, frame_b, evt_out, &stats);
    bool sd_motion_ok = (stats.on_events == 512 && stats.off_events == 512 &&
                         !stats.is_static_frame && stats.sparsity < 1.0f);
    for (int i = 0; i < 512 && sd_motion_ok; ++i) {
        if (evt_out[i] != 127) sd_motion_ok = false;
    }
    for (int i = DVS_FRAME_PLANE; i < DVS_FRAME_PLANE + 512 && sd_motion_ok; ++i) {
        if (evt_out[i] != -127) sd_motion_ok = false;
    }
    for (int i = 2 * DVS_FRAME_PLANE; i < 2 * DVS_FRAME_PLANE + 256 && sd_motion_ok; ++i) {
        if (evt_out[i] != 0) sd_motion_ok = false;
    }

    /* 4º frame igual ao 3º -> referência atualizada para sítios disparados, volta a 100% esparso */
    dvs_encoder_process(&dvs_state, frame_b, evt_out, &stats);
    bool sd_settled_ok = (stats.on_events == 0 && stats.off_events == 0 &&
                          stats.sparsity == 1.0f && stats.is_static_frame);

    checks_total++;
    if (sd_init_ok && sd_static_ok && sd_motion_ok && sd_settled_ok) checks_passed++;

    /* 2. DVS_MODE_SIGNED_DELTA (contínuo, binary_events=false) */
    cfg_sd.binary_events = false;
    dvs_encoder_init(&dvs_state, &cfg_sd);
    dvs_encoder_process(&dvs_state, frame_a, evt_out, &stats);
    dvs_encoder_process(&dvs_state, frame_b, evt_out, &stats);
    bool sd_cont_ok = (stats.on_events == 512 && stats.off_events == 512 &&
                       evt_out[0] == 30 && evt_out[DVS_FRAME_PLANE] == -25);
    checks_total++;
    if (sd_cont_ok) checks_passed++;

    /* 3. DVS_MODE_ON_OFF_LUMA: init -> estático -> movimento -> settle -> snn_engine_infer */
    /* Para ON_OFF_LUMA, variamos todos os 3 canais igualmente em 2 regiões distintas para garantir delta de Y */
    static int8_t luma_b[DVS_FRAME_BYTES];
    for (int p = 0; p < DVS_FRAME_PLANE; ++p) {
        int8_t delta = 0;
        if (p < 128) delta = 30;        /* ON */
        else if (p < 256) delta = -20;  /* OFF */
        luma_b[p] = static_cast<int8_t>(frame_a[p] + delta);
        luma_b[DVS_FRAME_PLANE + p] = static_cast<int8_t>(frame_a[DVS_FRAME_PLANE + p] + delta);
        luma_b[2 * DVS_FRAME_PLANE + p] = static_cast<int8_t>(frame_a[2 * DVS_FRAME_PLANE + p] + delta);
    }

    dvs_encoder_cfg_t cfg_luma = dvs_encoder_default_cfg(DVS_MODE_ON_OFF_LUMA);
    dvs_encoder_init(&dvs_state, &cfg_luma);

    dvs_encoder_process(&dvs_state, frame_a, evt_out, &stats);
    bool luma_init_ok = (stats.total_sites == DVS_FRAME_PLANE && stats.is_static_frame && stats.sparsity == 1.0f);

    dvs_encoder_process(&dvs_state, frame_a, evt_out, &stats);
    bool luma_static_ok = (stats.on_events == 0 && stats.off_events == 0 && stats.is_static_frame);

    dvs_encoder_process(&dvs_state, luma_b, evt_out, &stats);
    bool luma_motion_ok = (stats.on_events == 128 && stats.off_events == 128 && !stats.is_static_frame);
    for (int p = 0; p < 128 && luma_motion_ok; ++p) {
        if (evt_out[p] != 127 || evt_out[DVS_FRAME_PLANE + p] != 0 || evt_out[2 * DVS_FRAME_PLANE + p] != 0) {
            luma_motion_ok = false;
        }
    }
    for (int p = 128; p < 256 && luma_motion_ok; ++p) {
        if (evt_out[p] != 0 || evt_out[DVS_FRAME_PLANE + p] != 127 || evt_out[2 * DVS_FRAME_PLANE + p] != 0) {
            luma_motion_ok = false;
        }
    }

    SNNInferenceResult dvs_snn_res;
    snn_engine_infer(reinterpret_cast<const uint8_t*>(evt_out), &dvs_snn_res);
    bool luma_snn_ok = (dvs_snn_res.predicted_class >= 0 &&
                        dvs_snn_res.predicted_class < SPIKEVISION_NUM_CLASSES &&
                        dvs_snn_res.timesteps_executed == SPIKEVISION_TIMESTEPS);

    dvs_encoder_process(&dvs_state, luma_b, evt_out, &stats);
    bool luma_settled_ok = (stats.on_events == 0 && stats.off_events == 0 &&
                            stats.sparsity == 1.0f && stats.is_static_frame);

    checks_total++;
    if (luma_init_ok && luma_static_ok && luma_motion_ok && luma_snn_ok && luma_settled_ok) {
        checks_passed++;
    }

    if (checks_passed == checks_total) {
        printf("  • Testes Pseudo-DVS (SIGNED_DELTA + ON_OFF_LUMA + SNN): \033[32m%d/%d PASS\033[0m (classe=%d, sparsity=%.1f%%)\n\n",
               checks_passed, checks_total, dvs_snn_res.predicted_class, stats.sparsity * 100.0f);
        return 0;
    }

    printf("  • Testes Pseudo-DVS: \033[31m%d/%d FALHA\033[0m\n\n", checks_passed, checks_total);
    return 1;
}

#if defined(ESP_PLATFORM)
extern "C" void app_main(void) {
    print_banner();
    print_system_info();
    snn_engine_init();
    run_validation_suite();
    run_pipeline_validation();
    run_early_exit_benchmark();
    run_camera_preprocess_validation();
    run_dvs_encoder_validation();

    ESP_LOGI(TAG, "SpikeVision-S3 Neuromorphic Engine executado com sucesso.");
    vTaskDelay(pdMS_TO_TICKS(1000));
}
#else
int main(int argc, char** argv) {
    (void)argc;
    (void)argv;
    print_banner();
    print_system_info();
    snn_engine_init();
    int rc = run_validation_suite();
    int pipe_rc = run_pipeline_validation();
    int ee_rc = run_early_exit_benchmark();
    int cam_rc = run_camera_preprocess_validation();
    int dvs_rc = run_dvs_encoder_validation();
    return (rc != 0 || pipe_rc != 0 || ee_rc != 0 || cam_rc != 0 || dvs_rc != 0) ? 1 : 0;
}
#endif
