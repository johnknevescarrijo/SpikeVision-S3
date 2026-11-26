/**
 * SpikeVision-S3: Implementação do Encoder Neuromórfico Pseudo-DVS (Zero Malloc)
 */

#include "dvs_encoder.h"
#include <string.h>

static inline int8_t clamp_i8(int32_t v, int32_t lo, int32_t hi) {
    if (v < lo) return static_cast<int8_t>(lo);
    if (v > hi) return static_cast<int8_t>(hi);
    return static_cast<int8_t>(v);
}

static inline uint8_t compute_luma_y(int8_t r_i8, int8_t g_i8, int8_t b_i8) {
    const int32_t r = (r_i8 < 0) ? 0 : static_cast<int32_t>(r_i8);
    const int32_t g = (g_i8 < 0) ? 0 : static_cast<int32_t>(g_i8);
    const int32_t b = (b_i8 < 0) ? 0 : static_cast<int32_t>(b_i8);
    return static_cast<uint8_t>((77 * r + 150 * g + 29 * b) >> 8);
}

dvs_encoder_cfg_t dvs_encoder_default_cfg(dvs_mode_t mode) {
    dvs_encoder_cfg_t cfg;
    cfg.mode = mode;
    cfg.threshold = 8;
    cfg.event_amplitude = 127;
    cfg.binary_events = true;
    cfg.min_active_events = 16;
    return cfg;
}

void dvs_encoder_init(dvs_encoder_state_t* state, const dvs_encoder_cfg_t* cfg) {
    if (!state) return;
    if (cfg) {
        state->cfg = *cfg;
    } else {
        state->cfg = dvs_encoder_default_cfg(DVS_MODE_SIGNED_DELTA);
    }
    dvs_encoder_reset(state);
}

void dvs_encoder_reset(dvs_encoder_state_t* state) {
    if (!state) return;
    memset(state->ref_frame, 0, sizeof(state->ref_frame));
    memset(state->ref_luma, 0, sizeof(state->ref_luma));
    state->initialized = false;
}

void dvs_encoder_process(
    dvs_encoder_state_t* state,
    const int8_t* curr_chw32,
    int8_t* out_event_chw32,
    dvs_frame_stats_t* stats)
{
    if (!state || !curr_chw32 || !out_event_chw32) return;

    const uint32_t total_sites = (state->cfg.mode == DVS_MODE_ON_OFF_LUMA)
                                     ? DVS_FRAME_PLANE
                                     : DVS_FRAME_BYTES;

    if (!state->initialized) {
        if (state->cfg.mode == DVS_MODE_ON_OFF_LUMA) {
            for (int p = 0; p < DVS_FRAME_PLANE; ++p) {
                state->ref_luma[p] = compute_luma_y(
                    curr_chw32[p],
                    curr_chw32[DVS_FRAME_PLANE + p],
                    curr_chw32[2 * DVS_FRAME_PLANE + p]);
            }
        } else {
            memcpy(state->ref_frame, curr_chw32, DVS_FRAME_BYTES);
        }
        memset(out_event_chw32, 0, DVS_FRAME_BYTES);
        state->initialized = true;

        if (stats) {
            stats->total_sites = total_sites;
            stats->on_events = 0;
            stats->off_events = 0;
            stats->sparsity = 1.0f;
            stats->is_static_frame = true;
        }
        return;
    }

    uint32_t on_events = 0;
    uint32_t off_events = 0;
    const int16_t thresh = state->cfg.threshold;
    const int8_t amp_pos = state->cfg.event_amplitude;
    const int8_t amp_neg = static_cast<int8_t>(-static_cast<int16_t>(state->cfg.event_amplitude));
    const bool binary = state->cfg.binary_events;

    if (state->cfg.mode == DVS_MODE_ON_OFF_LUMA) {
        int8_t* out_on = out_event_chw32;
        int8_t* out_off = out_event_chw32 + DVS_FRAME_PLANE;
        int8_t* out_ch2 = out_event_chw32 + 2 * DVS_FRAME_PLANE;

        for (int p = 0; p < DVS_FRAME_PLANE; ++p) {
            const uint8_t y = compute_luma_y(
                curr_chw32[p],
                curr_chw32[DVS_FRAME_PLANE + p],
                curr_chw32[2 * DVS_FRAME_PLANE + p]);
            const int16_t diff = static_cast<int16_t>(y) - static_cast<int16_t>(state->ref_luma[p]);

            if (diff >= thresh) {
                out_on[p] = binary ? amp_pos : clamp_i8(diff, 0, 127);
                out_off[p] = 0;
                state->ref_luma[p] = y;
                ++on_events;
            } else if (diff <= -thresh) {
                out_on[p] = 0;
                out_off[p] = binary ? amp_pos : clamp_i8(-static_cast<int32_t>(diff), 0, 127);
                state->ref_luma[p] = y;
                ++off_events;
            } else {
                out_on[p] = 0;
                out_off[p] = 0;
            }
            out_ch2[p] = 0;
        }
    } else {
        for (int i = 0; i < DVS_FRAME_BYTES; ++i) {
            const int16_t diff = static_cast<int16_t>(curr_chw32[i]) - static_cast<int16_t>(state->ref_frame[i]);

            if (diff >= thresh) {
                out_event_chw32[i] = binary ? amp_pos : clamp_i8(diff, -127, 127);
                state->ref_frame[i] = curr_chw32[i];
                ++on_events;
            } else if (diff <= -thresh) {
                out_event_chw32[i] = binary ? amp_neg : clamp_i8(diff, -127, 127);
                state->ref_frame[i] = curr_chw32[i];
                ++off_events;
            } else {
                out_event_chw32[i] = 0;
            }
        }
    }

    if (stats) {
        const uint32_t total_events = on_events + off_events;
        stats->total_sites = total_sites;
        stats->on_events = on_events;
        stats->off_events = off_events;
        stats->sparsity = 1.0f - (static_cast<float>(total_events) / static_cast<float>(total_sites));
        stats->is_static_frame = (total_events < state->cfg.min_active_events);
    }
}
