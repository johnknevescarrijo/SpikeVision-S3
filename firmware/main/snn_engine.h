#pragma once

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "weights.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Estrutura contendo o resultado completo e as métricas de inferência da CSNN.
 */
typedef struct {
    int predicted_class;          /**< 0 = Nao-Gato (Non-Cat), 1 = Gato (Cat) */
    int32_t spike_counts[2];      /**< Spikes acumulados na saída: [non_cat, cat] */
    int32_t final_mem[2];         /**< Potencial final de membrana no timestep T: [non_cat, cat] */

    /* Métricas neuromórficas de disparo e esparsidade */
    uint32_t total_spikes_l1;     /**< Spikes totais disparados na Camada 1 */
    uint32_t total_spikes_l2;     /**< Spikes totais disparados na Camada 2 */
    uint32_t total_spikes_l3;     /**< Spikes totais disparados na Camada 3 */
    uint32_t total_spikes_l4;     /**< Spikes totais disparados na Camada 4 */
    uint32_t total_spikes;        /**< Soma de todos os spikes da rede nos T timesteps */

    /* Eficiência computacional: SOPs (Synaptic Operations) vs MACs Densos */
    uint64_t active_sops;         /**< Operações de soma executadas sob disparo de spikes */
    uint64_t dense_macs;          /**< Total de MACs que uma CNN densa equivalente executaria */
    float sparsity_percent;       /**< % de inatividade sináptica (1.0 - firing_rate) */
    float sop_reduction_percent;  /**< % de redução de operações vs modelo denso */

    /* Benchmarks de tempo e ciclos de clock */
    uint64_t latency_us;          /**< Tempo total de inferência em microssegundos */
    uint32_t cpu_cycles;          /**< Ciclos de CPU consumidos (ESP32-S3) */
} SNNInferenceResult;

/**
 * Inicializa os buffers do motor de inferência SNN.
 */
void snn_engine_init(void);

/**
 * Executa a inferência completa da SNN sobre uma imagem planar 3x32x32 uint8_t [0..127].
 *
 * @param image_chw Ponteiro para os dados da imagem em formato plano CHW (3 * 32 * 32 bytes).
 * @param result    Estrutura onde os resultados, predição e benchmarks serão gravados.
 */
void snn_engine_infer(const uint8_t* image_chw, SNNInferenceResult* result);

/**
 * Retorna o consumo total de memória estática (SRAM) alocada pelos buffers do motor.
 */
size_t snn_engine_get_sram_usage(void);

#ifdef __cplusplus
}
#endif
