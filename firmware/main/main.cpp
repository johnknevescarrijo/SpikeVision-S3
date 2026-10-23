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
    printf("  • Consumo SRAM Motor:   %zu bytes (%.2f KB) [100%% Estática, 0 Heap]\n",
           snn_engine_get_sram_usage(), (float)snn_engine_get_sram_usage() / 1024.0f);
    printf("  • Ocupação ROM Pesos:   15608 bytes (15.24 KB)\n\n");
}

static int run_validation_suite(void) {
    printf("🚀 [Executando Bateria de Testes nas 10 Amostras do CIFAR-10]...\n");
    printf("------------------------------------------------------------------------------------------------------------------------------------\n");
    printf("| ID | Ground-Truth | Pred SNN | Spikes [N, C] | Membrana Final   | Esparsidade | SOPs Ativas | Latência  | Ciclos CPU | Status       |\n");
    printf("------------------------------------------------------------------------------------------------------------------------------------\n");

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

        bool spk_match = (result.spike_counts[0] == tc->expected_spikes[0] &&
                          result.spike_counts[1] == tc->expected_spikes[1]);
        bool mem_match = (result.final_mem[0] == tc->expected_mem[0] &&
                          result.final_mem[1] == tc->expected_mem[1]);
        bool pred_match = (result.predicted_class == tc->expected_prediction);
        bool bit_accurate = spk_match && mem_match && pred_match;

        if (bit_accurate) bit_accurate_matches++;
        if (result.predicted_class == tc->ground_truth) correct_predictions++;

        total_latency_us += result.latency_us;
        total_sops += result.active_sops;
        total_dense_macs += result.dense_macs;
        total_sparsity += result.sparsity_percent;

        const char* gt_str = (tc->ground_truth == 1) ? "GATO    " : "NAO-GATO";
        const char* pred_str = (result.predicted_class == 1) ? "GATO    " : "NAO-GATO";
        const char* status_str = bit_accurate ? "\033[32mPASS (BIT-ACC)\033[0m" : "\033[31mFAIL          \033[0m";

        printf("| %02d | %s     | %s |   [%" PRId32 ", %" PRId32 "]     | [%6" PRId32 ", %6" PRId32 "] |   %5.1f%%    | %10" PRIu64 "  | %7.2f ms | %10" PRIu32 " | %s |\n",
               i,
               gt_str,
               pred_str,
               result.spike_counts[0], result.spike_counts[1],
               result.final_mem[0], result.final_mem[1],
               result.sparsity_percent,
               result.active_sops,
               (float)result.latency_us / 1000.0f,
               result.cpu_cycles,
               status_str);
    }
    printf("------------------------------------------------------------------------------------------------------------------------------------\n\n");

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
    printf("  • Memória Flash Necessária:        ~15.24 KB\n");
    printf("  • Memória SRAM Interna Necessária: ~162.50 KB (100%% alocação estática)\n\n");

    if (bit_accurate_matches == TEST_SAMPLE_COUNT) {
        printf("\033[32m✔ SUCESSO ABSOLUTO: Todas as 10 amostras apresentaram 100%% de paridade bit-a-bit!\033[0m\n\n");
        return 0;
    } else {
        printf("\033[31m✘ FALHA: Divergência detectada em %d amostras.\033[0m\n\n",
               TEST_SAMPLE_COUNT - bit_accurate_matches);
        return 1;
    }
}

#if defined(ESP_PLATFORM)
extern "C" void app_main(void) {
    print_banner();
    print_system_info();
    snn_engine_init();
    run_validation_suite();

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
    return run_validation_suite();
}
#endif
