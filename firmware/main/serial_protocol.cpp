/**
 * SpikeVision-S3: Implementação do Protocolo Serial Framed Zero-Malloc
 */

#include "serial_protocol.h"
#include "snn_engine.h"
#include "image_preprocess.h"

#include <stdio.h>
#include <string.h>
#include <inttypes.h>

#if defined(ESP_PLATFORM)
    #include "freertos/FreeRTOS.h"
    #include "freertos/task.h"
    #include <errno.h>
#endif

static uint16_t crc16_update(uint16_t crc, uint8_t byte) {
    crc ^= static_cast<uint16_t>(byte) << 8;
    for (int i = 0; i < 8; ++i) {
        if (crc & 0x8000U) {
            crc = static_cast<uint16_t>((crc << 1) ^ 0x1021U);
        } else {
            crc = static_cast<uint16_t>(crc << 1);
        }
    }
    return crc;
}

uint16_t serial_protocol_crc16(const uint8_t* data, size_t len) {
    uint16_t crc = 0xFFFFU;
    for (size_t i = 0; i < len; ++i) {
        crc = crc16_update(crc, data[i]);
    }
    return crc;
}

static void proto_reset_parser(serial_protocol_t* proto) {
    proto->state = PROTO_STATE_MAGIC_0;
    proto->cmd = 0;
    proto->flags = 0;
    proto->len = 0;
    proto->payload_idx = 0;
    proto->rx_crc = 0;
    proto->calc_crc = 0xFFFFU;
}

static void proto_send(serial_protocol_t* proto, const char* str, size_t len) {
    if (proto->write_fn != nullptr && len > 0) {
        proto->write_fn(proto->write_ctx, str, len);
    }
}

static void proto_send_error(serial_protocol_t* proto, const char* code) {
    int n = snprintf(proto->tx_buf, sizeof(proto->tx_buf),
                     "{\"type\":\"error\",\"code\":\"%s\"}\n", code);
    if (n > 0) {
        size_t len = (static_cast<size_t>(n) < sizeof(proto->tx_buf))
                         ? static_cast<size_t>(n)
                         : sizeof(proto->tx_buf) - 1;
        proto_send(proto, proto->tx_buf, len);
    }
}

static void proto_ensure_dvs(serial_protocol_t* proto) {
    if (!proto->dvs_initialized) {
        dvs_encoder_cfg_t cfg = dvs_encoder_default_cfg(DVS_MODE_SIGNED_DELTA);
        dvs_encoder_init(&proto->dvs_state, &cfg);
        proto->dvs_initialized = true;
    }
}

static void proto_execute_infer(serial_protocol_t* proto, const int8_t* chw_in) {
    const uint8_t flags = proto->flags;
    const bool use_dvs = (flags & SERIAL_FLAG_USE_DVS) != 0;

    dvs_frame_stats_t dvs_stats;
    const int8_t* infer_ptr = chw_in;

    if (use_dvs) {
        proto_ensure_dvs(proto);
        dvs_encoder_process(&proto->dvs_state, chw_in, proto->dvs_buf, &dvs_stats);
        infer_ptr = proto->dvs_buf;
    }

    snn_early_exit_cfg_t ee_cfg = { false, 2, 0 };
    const snn_early_exit_cfg_t* ee_ptr = nullptr;
    if (flags & SERIAL_FLAG_EE_AGGRESSIVE) {
        ee_cfg.enabled = true;
        ee_cfg.min_timesteps = 2;
        ee_cfg.margin = 2;
        ee_ptr = &ee_cfg;
    } else if (flags & SERIAL_FLAG_EE_LOSSLESS) {
        ee_cfg.enabled = true;
        ee_cfg.min_timesteps = 2;
        ee_cfg.margin = 0;
        ee_ptr = &ee_cfg;
    }

    SNNInferenceResult res;
    snn_engine_infer_adaptive(reinterpret_cast<const uint8_t*>(infer_ptr), &res, ee_ptr);

    char* p = proto->tx_buf;
    size_t rem = sizeof(proto->tx_buf);
    int n = snprintf(p, rem, "{\"type\":\"infer\",\"pred\":%d,\"spike_counts\":[",
                     res.predicted_class);
    if (n <= 0 || static_cast<size_t>(n) >= rem) return;
    p += n;
    rem -= static_cast<size_t>(n);

    for (int c = 0; c < SPIKEVISION_NUM_CLASSES; ++c) {
        n = snprintf(p, rem, "%s%" PRId32, (c == 0) ? "" : ",", res.spike_counts[c]);
        if (n <= 0 || static_cast<size_t>(n) >= rem) return;
        p += n;
        rem -= static_cast<size_t>(n);
    }

    n = snprintf(
        p, rem,
        "],\"timesteps_executed\":%" PRId32 ",\"early_exited\":%s,"
        "\"total_sops\":%" PRIu64 ",\"sparsity\":%.2f,\"latency_us\":%" PRIu64,
        res.timesteps_executed,
        res.early_exited ? "true" : "false",
        res.active_sops,
        static_cast<double>(res.sparsity_percent),
        res.latency_us);
    if (n <= 0 || static_cast<size_t>(n) >= rem) return;
    p += n;
    rem -= static_cast<size_t>(n);

    if (use_dvs) {
        n = snprintf(
            p, rem,
            ",\"dvs_sparsity\":%.2f,\"dvs_on\":%" PRIu32 ",\"dvs_off\":%" PRIu32 ",\"dvs_static\":%s",
            static_cast<double>(dvs_stats.sparsity * 100.0f),
            dvs_stats.on_events,
            dvs_stats.off_events,
            dvs_stats.is_static_frame ? "true" : "false");
        if (n <= 0 || static_cast<size_t>(n) >= rem) return;
        p += n;
        rem -= static_cast<size_t>(n);
    }

    n = snprintf(p, rem, "}\n");
    if (n <= 0 || static_cast<size_t>(n) >= rem) return;
    p += n;

    proto_send(proto, proto->tx_buf, static_cast<size_t>(p - proto->tx_buf));
}

static void proto_dispatch(serial_protocol_t* proto) {
    switch (proto->cmd) {
        case CMD_PING: {
            if (proto->len != 0) {
                proto_send_error(proto, "invalid_length");
                return;
            }
            int n = snprintf(
                proto->tx_buf, sizeof(proto->tx_buf),
                "{\"type\":\"pong\",\"version\":\"%s\",\"num_classes\":%d,"
                "\"timesteps\":%d,\"sram_bytes\":%zu,\"simd\":\"%s\"}\n",
                SERIAL_PROTO_VERSION,
                SPIKEVISION_NUM_CLASSES,
                SPIKEVISION_TIMESTEPS,
                snn_engine_get_sram_bytes(),
                snn_engine_get_simd_backend());
            if (n > 0) {
                size_t len = (static_cast<size_t>(n) < sizeof(proto->tx_buf))
                                 ? static_cast<size_t>(n)
                                 : sizeof(proto->tx_buf) - 1;
                proto_send(proto, proto->tx_buf, len);
            }
            break;
        }

        case CMD_INFER_CHW32: {
            if (proto->len != DVS_FRAME_BYTES) {
                proto_send_error(proto, "invalid_length");
                return;
            }
            memcpy(proto->chw_buf, proto->payload, DVS_FRAME_BYTES);
            proto_execute_infer(proto, proto->chw_buf);
            break;
        }

        case CMD_INFER_RGB888: {
            if (proto->len < 4) {
                proto_send_error(proto, "invalid_length");
                return;
            }
            uint16_t w = static_cast<uint16_t>(proto->payload[0]) |
                         (static_cast<uint16_t>(proto->payload[1]) << 8);
            uint16_t h = static_cast<uint16_t>(proto->payload[2]) |
                         (static_cast<uint16_t>(proto->payload[3]) << 8);
            if (w == 0 || h == 0 || w > SERIAL_PROTO_MAX_DIM || h > SERIAL_PROTO_MAX_DIM) {
                proto_send_error(proto, "invalid_dimensions");
                return;
            }
            size_t expected = 4U + static_cast<size_t>(w) * static_cast<size_t>(h) * 3U;
            if (proto->len != expected) {
                proto_send_error(proto, "invalid_length");
                return;
            }
            img_rgb888_to_chw32(proto->payload + 4, static_cast<int>(w), static_cast<int>(h), proto->chw_buf);
            proto_execute_infer(proto, proto->chw_buf);
            break;
        }

        case CMD_RESET_DVS: {
            if (proto->len != 0) {
                proto_send_error(proto, "invalid_length");
                return;
            }
            proto_ensure_dvs(proto);
            dvs_encoder_reset(&proto->dvs_state);
            static const char kAck[] = "{\"type\":\"ack\",\"cmd\":\"reset_dvs\"}\n";
            proto_send(proto, kAck, sizeof(kAck) - 1);
            break;
        }

        default:
            proto_send_error(proto, "unknown_cmd");
            break;
    }
}

void serial_protocol_init(
    serial_protocol_t* proto,
    serial_write_str_fn write_fn,
    void* write_ctx) {
    memset(proto, 0, sizeof(*proto));
    proto->write_fn = write_fn;
    proto->write_ctx = write_ctx;
    proto_ensure_dvs(proto);
    proto_reset_parser(proto);
}

void serial_protocol_feed_byte(serial_protocol_t* proto, uint8_t byte) {
    switch (proto->state) {
        case PROTO_STATE_MAGIC_0:
            if (byte == SERIAL_PROTO_MAGIC_0) {
                proto->state = PROTO_STATE_MAGIC_1;
            }
            break;

        case PROTO_STATE_MAGIC_1:
            if (byte == SERIAL_PROTO_MAGIC_1) {
                proto->calc_crc = 0xFFFFU;
                proto->state = PROTO_STATE_CMD;
            } else if (byte != SERIAL_PROTO_MAGIC_0) {
                proto->state = PROTO_STATE_MAGIC_0;
            }
            break;

        case PROTO_STATE_CMD:
            proto->cmd = byte;
            proto->calc_crc = crc16_update(proto->calc_crc, byte);
            proto->state = PROTO_STATE_FLAGS;
            break;

        case PROTO_STATE_FLAGS:
            proto->flags = byte;
            proto->calc_crc = crc16_update(proto->calc_crc, byte);
            proto->state = PROTO_STATE_LEN_LO;
            break;

        case PROTO_STATE_LEN_LO:
            proto->len = byte;
            proto->calc_crc = crc16_update(proto->calc_crc, byte);
            proto->state = PROTO_STATE_LEN_HI;
            break;

        case PROTO_STATE_LEN_HI:
            proto->len |= static_cast<uint16_t>(byte) << 8;
            proto->calc_crc = crc16_update(proto->calc_crc, byte);
            if (proto->len > SERIAL_PROTO_MAX_PAYLOAD) {
                proto_send_error(proto, "payload_too_large");
                proto_reset_parser(proto);
            } else if (proto->len == 0) {
                proto->payload_idx = 0;
                proto->state = PROTO_STATE_CRC_LO;
            } else {
                proto->payload_idx = 0;
                proto->state = PROTO_STATE_PAYLOAD;
            }
            break;

        case PROTO_STATE_PAYLOAD:
            proto->payload[proto->payload_idx++] = byte;
            proto->calc_crc = crc16_update(proto->calc_crc, byte);
            if (proto->payload_idx >= proto->len) {
                proto->state = PROTO_STATE_CRC_LO;
            }
            break;

        case PROTO_STATE_CRC_LO:
            proto->rx_crc = byte;
            proto->state = PROTO_STATE_CRC_HI;
            break;

        case PROTO_STATE_CRC_HI:
            proto->rx_crc |= static_cast<uint16_t>(byte) << 8;
            if (proto->rx_crc == proto->calc_crc) {
                proto_dispatch(proto);
            } else {
                proto_send_error(proto, "bad_crc");
            }
            proto_reset_parser(proto);
            break;
    }
}

void serial_protocol_run(
    serial_protocol_t* proto,
    serial_read_byte_fn read_fn,
    void* read_ctx) {
    if (proto == nullptr || read_fn == nullptr) return;
    for (;;) {
        int ch = read_fn(read_ctx);
        if (ch < 0) {
            break;
        }
        serial_protocol_feed_byte(proto, static_cast<uint8_t>(ch));
    }
}

static int stdio_read_byte(void* ctx) {
    (void)ctx;
#if defined(ESP_PLATFORM)
    for (;;) {
        int ch = fgetc(stdin);
        if (ch != EOF) {
            return ch;
        }
        clearerr(stdin);
        vTaskDelay(pdMS_TO_TICKS(1));
    }
#else
    return fgetc(stdin);
#endif
}

static void stdio_write_str(void* ctx, const char* str, size_t len) {
    (void)ctx;
    fwrite(str, 1, len, stdout);
    fflush(stdout);
}

void serial_protocol_run_stdio(void) {
    static serial_protocol_t s_proto;
    setvbuf(stdin, nullptr, _IONBF, 0);
    setvbuf(stdout, nullptr, _IONBF, 0);
    serial_protocol_init(&s_proto, stdio_write_str, nullptr);
    serial_protocol_run(&s_proto, stdio_read_byte, nullptr);
}
