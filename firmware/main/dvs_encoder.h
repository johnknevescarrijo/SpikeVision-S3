/**
 * SpikeVision-S3: Encoder Neuromórfico Pseudo-DVS por Diferença Temporal (Zero Malloc)
 *
 * Converte sequências de frames CHW 3x32x32 (int8_t [0..127]) em mapas de eventos
 * esparsos ON/OFF compatíveis diretamente com snn_engine_infer().
 */

#pragma once

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DVS_FRAME_WIDTH     32
#define DVS_FRAME_HEIGHT    32
#define DVS_FRAME_CHANNELS  3
#define DVS_FRAME_PLANE     (DVS_FRAME_WIDTH * DVS_FRAME_HEIGHT)
#define DVS_FRAME_BYTES     (DVS_FRAME_CHANNELS * DVS_FRAME_PLANE)

/**
 * Modo de operação do encoder Pseudo-DVS.
 */
typedef enum {
    DVS_MODE_SIGNED_DELTA = 0,  /**< Diferença temporal por canal RGB: +amp/-amp/0 nos 3 canais */
    DVS_MODE_ON_OFF_LUMA  = 1   /**< Luminância inteira Y=(77R+150G+29B)>>8: ch0=ON, ch1=OFF, ch2=0 */
} dvs_mode_t;

/**
 * Configuração do encoder Pseudo-DVS.
 */
typedef struct {
    dvs_mode_t mode;            /**< Modo de codificação temporal */
    int16_t threshold;          /**< Limiar de disparo |curr - ref| >= threshold (padrão: 8) */
    int8_t event_amplitude;     /**< Amplitude de saída para eventos binários (padrão: 127) */
    bool binary_events;         /**< true: emite +/-event_amplitude; false: emite clamp(diff, -127, 127) */
    uint32_t min_active_events; /**< Mínimo de eventos para considerar frame não-estático (padrão: 16) */
} dvs_encoder_cfg_t;

/**
 * Estado interno estático do encoder Pseudo-DVS (zero alocação dinâmica).
 */
typedef struct {
    dvs_encoder_cfg_t cfg;
    int8_t ref_frame[DVS_FRAME_BYTES];   /**< Referência CHW para DVS_MODE_SIGNED_DELTA */
    uint8_t ref_luma[DVS_FRAME_PLANE];   /**< Referência de luminância para DVS_MODE_ON_OFF_LUMA */
    bool initialized;                    /**< true após processar o 1º frame de referência */
} dvs_encoder_state_t;

/**
 * Estatísticas de esparsidade e atividade de um frame de eventos.
 */
typedef struct {
    uint32_t total_sites;       /**< 3072 para SIGNED_DELTA; 1024 para ON_OFF_LUMA */
    uint32_t on_events;         /**< Quantidade de eventos positivos (ON) */
    uint32_t off_events;        /**< Quantidade de eventos negativos (OFF) */
    float sparsity;             /**< Fração de sítios inativos [0.0f .. 1.0f] */
    bool is_static_frame;       /**< true se (on_events + off_events) < min_active_events */
} dvs_frame_stats_t;

/**
 * Retorna a configuração padrão para o modo especificado.
 */
dvs_encoder_cfg_t dvs_encoder_default_cfg(dvs_mode_t mode);

/**
 * Inicializa o estado do encoder Pseudo-DVS com a configuração informada.
 * Se cfg == NULL, utiliza dvs_encoder_default_cfg(DVS_MODE_SIGNED_DELTA).
 */
void dvs_encoder_init(dvs_encoder_state_t* state, const dvs_encoder_cfg_t* cfg);

/**
 * Reinicia o estado de referência do encoder mantendo a configuração atual.
 */
void dvs_encoder_reset(dvs_encoder_state_t* state);

/**
 * Processa um frame CHW 3x32x32 e gera o tensor de eventos int8_t[3072].
 * No 1º frame após init/reset, apenas inicializa a referência, zera out_event_chw32,
 * e preenche stats com sparsity=1.0f e is_static_frame=true.
 */
void dvs_encoder_process(
    dvs_encoder_state_t* state,
    const int8_t* curr_chw32,
    int8_t* out_event_chw32,
    dvs_frame_stats_t* stats);

#ifdef __cplusplus
}
#endif
