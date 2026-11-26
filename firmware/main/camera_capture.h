/**
 * SpikeVision-S3: Driver de Captura de Câmera (ESP32-S3 OV2640/OV5640 + Fallback CI)
 *
 * Integra esp32-camera diretamente com o pipeline RGB565 -> 32x32 CHW [0..127].
 * Quando esp_camera.h não estiver disponível (Host Linux ou ESP-IDF sem o componente),
 * fornece fallback limpo com gerador de frame sintético opcional para testes de CI.
 */

#pragma once

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Presets de pinagem para placas ESP32-S3 com câmera DVP.
 */
typedef enum {
    CAM_PRESET_ESP32S3_EYE = 0,
    CAM_PRESET_XIAO_ESP32S3_SENSE = 1,
    CAM_PRESET_FREENOVE_S3_WROOM = 2,
    CAM_PRESET_CUSTOM = 3
} camera_board_preset_t;

/**
 * Mapeamento de pinos GPIO para interface paralela DVP da câmera.
 */
typedef struct {
    int pwdn;
    int reset;
    int xclk;
    int siod;
    int sioc;
    int d7;
    int d6;
    int d5;
    int d4;
    int d3;
    int d2;
    int d1;
    int d0;
    int vsync;
    int href;
    int pclk;
} camera_capture_pins_t;

/**
 * Configuração do módulo de captura de câmera.
 */
typedef struct {
    camera_board_preset_t preset;
    int xclk_freq_hz;              /**< Frequência XCLK em Hz (0 = padrão 20 MHz) */
    bool swap_rgb565_bytes;        /**< Inverte bytes do RGB565 (padrão: true para OV2640/OV5640) */
    bool use_qqvga;                /**< false = FRAMESIZE_96X96, true = FRAMESIZE_QQVGA (160x120) */
    bool enable_synthetic_fallback;/**< true = gera frame sintético quando esp_camera.h não existir */
    camera_capture_pins_t custom_pins; /**< Usado apenas quando preset == CAM_PRESET_CUSTOM */
} camera_capture_cfg_t;

/**
 * Retorna uma configuração padrão preenchida para o preset informado.
 */
camera_capture_cfg_t camera_capture_default_cfg(camera_board_preset_t preset);

/**
 * Inicializa o driver de câmera (ou modo sintético de fallback).
 *
 * @param cfg Ponteiro para configuração (nullptr usa CAM_PRESET_ESP32S3_EYE com fallback habilitado).
 * @return true se inicializado com sucesso.
 */
bool camera_capture_init(const camera_capture_cfg_t* cfg);

/**
 * Captura um frame e converte diretamente para tensor planar CHW 3x32x32 int8_t [0..127].
 *
 * @param out_chw32      Buffer de saída de 3072 bytes (3 * 32 * 32).
 * @param out_capture_us Opcional; se não-nulo, recebe a latência de captura + pré-processamento em us.
 * @return true em caso de sucesso, false se não inicializado ou falha na captura.
 */
bool camera_capture_grab_chw32(int8_t* out_chw32, uint32_t* out_capture_us);

/**
 * Libera os recursos do driver de câmera.
 */
void camera_capture_deinit(void);

/**
 * Retorna true se o firmware foi compilado com o driver real esp_camera.h.
 */
bool camera_capture_has_hardware_driver(void);

#ifdef __cplusplus
}
#endif
