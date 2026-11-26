/**
 * SpikeVision-S3: Protocolo Serial Framed Zero-Malloc (USB-CDC / UART / Host Pipe)
 *
 * Framing binário na entrada:
 *   MAGIC(2) = {'S', 'V'} | CMD(1) | FLAGS(1) | LEN(2 uint16 LE) | PAYLOAD(LEN) | CRC16_CCITT(2 uint16 LE)
 *
 * CRC-16/CCITT-FALSE (poly=0x1021, init=0xFFFF) calculado sobre:
 *   CMD(1) | FLAGS(1) | LEN(2 LE) | PAYLOAD(LEN)
 *
 * Respostas na saída em formato JSON-line ('\n').
 */

#pragma once

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "dvs_encoder.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SERIAL_PROTO_MAGIC_0        'S'
#define SERIAL_PROTO_MAGIC_1        'V'
#define SERIAL_PROTO_VERSION        "1.0"

/* Suporta até CMD_INFER_RGB888 64x64: 4 bytes (w,h) + 64*64*3 = 12292 bytes */
#define SERIAL_PROTO_MAX_DIM        64
#define SERIAL_PROTO_MAX_PAYLOAD    (4 + SERIAL_PROTO_MAX_DIM * SERIAL_PROTO_MAX_DIM * 3)
#define SERIAL_PROTO_TX_BUF_SIZE    512

typedef enum {
    CMD_PING         = 0x01,
    CMD_INFER_CHW32  = 0x02,
    CMD_INFER_RGB888 = 0x03,
    CMD_RESET_DVS    = 0x04
} serial_cmd_t;

/* Flags para CMD_INFER_CHW32 e CMD_INFER_RGB888 */
#define SERIAL_FLAG_EE_LOSSLESS     0x01  /* Early-Exit Lossless (margin=0, min_t=2) */
#define SERIAL_FLAG_EE_AGGRESSIVE   0x02  /* Early-Exit Agressivo (margin=2, min_t=2) */
#define SERIAL_FLAG_USE_DVS         0x04  /* Pré-processa via dvs_encoder_process */

typedef enum {
    PROTO_STATE_MAGIC_0 = 0,
    PROTO_STATE_MAGIC_1,
    PROTO_STATE_CMD,
    PROTO_STATE_FLAGS,
    PROTO_STATE_LEN_LO,
    PROTO_STATE_LEN_HI,
    PROTO_STATE_PAYLOAD,
    PROTO_STATE_CRC_LO,
    PROTO_STATE_CRC_HI
} serial_proto_state_t;

/** Callback para leitura de 1 byte (retorna 0..255, ou -1 em EOF/erro). */
typedef int (*serial_read_byte_fn)(void* ctx);

/** Callback para escrita de resposta JSON-line. */
typedef void (*serial_write_str_fn)(void* ctx, const char* str, size_t len);

/**
 * Contexto estático do parser/executor do protocolo (Zero Malloc).
 */
typedef struct {
    serial_proto_state_t state;
    uint8_t cmd;
    uint8_t flags;
    uint16_t len;
    uint16_t payload_idx;
    uint16_t rx_crc;
    uint16_t calc_crc;

    uint8_t payload[SERIAL_PROTO_MAX_PAYLOAD];
    int8_t chw_buf[DVS_FRAME_BYTES];
    int8_t dvs_buf[DVS_FRAME_BYTES];
    char tx_buf[SERIAL_PROTO_TX_BUF_SIZE];

    dvs_encoder_state_t dvs_state;
    bool dvs_initialized;

    serial_write_str_fn write_fn;
    void* write_ctx;
} serial_protocol_t;

/**
 * Calcula CRC-16/CCITT-FALSE (poly=0x1021, init=0xFFFF) sobre um buffer.
 */
uint16_t serial_protocol_crc16(const uint8_t* data, size_t len);

/**
 * Inicializa o contexto do protocolo serial e o encoder DVS interno.
 */
void serial_protocol_init(
    serial_protocol_t* proto,
    serial_write_str_fn write_fn,
    void* write_ctx);

/**
 * Alimenta 1 byte na máquina de estados do protocolo.
 * Dispara a execução do comando e envia a resposta JSON ao completar o frame.
 */
void serial_protocol_feed_byte(serial_protocol_t* proto, uint8_t byte);

/**
 * Executa o loop de leitura/processamento até read_fn retornar -1.
 */
void serial_protocol_run(
    serial_protocol_t* proto,
    serial_read_byte_fn read_fn,
    void* read_ctx);

/**
 * Executa o protocolo serial sobre stdin/stdout (Host --serial ou ESP32-S3 VFS).
 */
void serial_protocol_run_stdio(void);

#ifdef __cplusplus
}
#endif
