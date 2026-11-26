/**
 * SpikeVision-S3: Pipeline Assíncrono Dual-Core FreeRTOS (ESP32-S3) / Worker Thread (Host).
 *
 * Arquitetura produtor-consumidor com buffers estáticos (zero malloc no loop quente):
 *  - ESP32-S3: Core 0 livre para I/O (câmera, Wi-Fi, Serial) e Core 1 dedicado à inferência SNN.
 *  - Host Linux: mesma API sobre std::thread/std::mutex/std::condition_variable.
 */

#pragma once

#include <stdint.h>
#include <stdbool.h>
#include "snn_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SNN_PIPELINE_QUEUE_DEPTH 4
#define SNN_PIPELINE_FRAME_BYTES (SPIKEVISION_IN_CHANNELS * SPIKEVISION_INPUT_HEIGHT * SPIKEVISION_INPUT_WIDTH)

/**
 * Resultado assíncrono produzido pelo pipeline de inferência SNN.
 */
typedef struct {
    uint32_t frame_id;
    SNNInferenceResult result;
} snn_pipeline_output_t;

/**
 * Inicia a task/thread de inferência SNN e as filas estáticas do pipeline.
 *
 * @param early_exit_cfg Configuração opcional de early-exit (nullptr desabilita).
 * @return true em caso de sucesso, false se já ativo ou em erro de inicialização.
 */
bool snn_pipeline_start(const snn_early_exit_cfg_t* early_exit_cfg);

/**
 * Submete um frame 3x32x32 para inferência assíncrona.
 *
 * @param frame_id   Identificador sequencial do frame.
 * @param input_chw  Ponteiro para o tensor CHW (3072 bytes, valores [0..127]).
 * @param timeout_ms Tempo máximo de espera em ms caso a fila esteja cheia.
 * @return true se enfileirado, false se timeout/inativo/argumento inválido.
 */
bool snn_pipeline_submit(uint32_t frame_id, const int8_t* input_chw, uint32_t timeout_ms);

/**
 * Recebe o próximo resultado processado pelo pipeline.
 *
 * @param out        Estrutura de saída preenchida com frame_id e SNNInferenceResult.
 * @param timeout_ms Tempo máximo de espera em ms caso não haja resultado pronto.
 * @return true se um resultado foi recebido, false em timeout ou encerramento.
 */
bool snn_pipeline_receive(snn_pipeline_output_t* out, uint32_t timeout_ms);

/**
 * Encerra o pipeline de forma segura e aguarda o término da task/thread trabalhadora.
 */
void snn_pipeline_stop(void);

#ifdef __cplusplus
}
#endif
