#include "image_preprocess.h"

namespace {

static inline int8_t quantize_u8_to_127(uint32_t u8) {
    /* Equivalente exato em aritmética inteira a round(u8 * 127.0 / 255.0) em [0..127] */
    uint32_t q = (u8 * 127u + 127u) / 255u;
    if (q > 127u) {
        q = 127u;
    }
    return static_cast<int8_t>(q);
}

template <typename ReadPixelFn>
static void downsample_crop_to_chw32(
    int src_w,
    int src_h,
    int8_t* out_chw32,
    ReadPixelFn read_pixel) {
    const int crop_size = (src_w < src_h) ? src_w : src_h;
    const int crop_x0 = (src_w - crop_size) / 2;
    const int crop_y0 = (src_h - crop_size) / 2;

    int8_t* out_r = out_chw32;
    int8_t* out_g = out_chw32 + IMG_PREPROC_PLANE;
    int8_t* out_b = out_chw32 + (2 * IMG_PREPROC_PLANE);

    for (int oy = 0; oy < IMG_PREPROC_HEIGHT; ++oy) {
        int y0 = crop_y0 + (oy * crop_size) / IMG_PREPROC_HEIGHT;
        int y1 = crop_y0 + ((oy + 1) * crop_size) / IMG_PREPROC_HEIGHT;
        if (y1 <= y0) {
            y1 = y0 + 1;
        }
        if (y1 > src_h) {
            y1 = src_h;
        }

        for (int ox = 0; ox < IMG_PREPROC_WIDTH; ++ox) {
            int x0 = crop_x0 + (ox * crop_size) / IMG_PREPROC_WIDTH;
            int x1 = crop_x0 + ((ox + 1) * crop_size) / IMG_PREPROC_WIDTH;
            if (x1 <= x0) {
                x1 = x0 + 1;
            }
            if (x1 > src_w) {
                x1 = src_w;
            }

            uint32_t sum_r = 0;
            uint32_t sum_g = 0;
            uint32_t sum_b = 0;
            uint32_t count = 0;

            for (int y = y0; y < y1; ++y) {
                const int row_offset = y * src_w;
                for (int x = x0; x < x1; ++x) {
                    uint32_t r = 0;
                    uint32_t g = 0;
                    uint32_t b = 0;
                    read_pixel(row_offset + x, r, g, b);
                    sum_r += r;
                    sum_g += g;
                    sum_b += b;
                    ++count;
                }
            }

            const uint32_t half = count / 2u;
            const uint32_t avg_r = (sum_r + half) / count;
            const uint32_t avg_g = (sum_g + half) / count;
            const uint32_t avg_b = (sum_b + half) / count;

            const int dst_idx = oy * IMG_PREPROC_WIDTH + ox;
            out_r[dst_idx] = quantize_u8_to_127(avg_r);
            out_g[dst_idx] = quantize_u8_to_127(avg_g);
            out_b[dst_idx] = quantize_u8_to_127(avg_b);
        }
    }
}

}  // namespace

extern "C" void img_rgb565_to_chw32(
    const uint16_t* rgb565,
    int src_w,
    int src_h,
    bool swap_bytes,
    int8_t* out_chw32) {
    if (rgb565 == nullptr || out_chw32 == nullptr || src_w <= 0 || src_h <= 0) {
        return;
    }

    downsample_crop_to_chw32(src_w, src_h, out_chw32,
        [rgb565, swap_bytes](int idx, uint32_t& r, uint32_t& g, uint32_t& b) {
            uint16_t px = rgb565[idx];
            if (swap_bytes) {
                px = static_cast<uint16_t>((px >> 8) | (px << 8));
            }
            const uint32_t r5 = (px >> 11) & 0x1Fu;
            const uint32_t g6 = (px >> 5) & 0x3Fu;
            const uint32_t b5 = px & 0x1Fu;

            r = (r5 << 3) | (r5 >> 2);
            g = (g6 << 2) | (g6 >> 4);
            b = (b5 << 3) | (b5 >> 2);
        });
}

extern "C" void img_rgb888_to_chw32(
    const uint8_t* rgb888,
    int src_w,
    int src_h,
    int8_t* out_chw32) {
    if (rgb888 == nullptr || out_chw32 == nullptr || src_w <= 0 || src_h <= 0) {
        return;
    }

    downsample_crop_to_chw32(src_w, src_h, out_chw32,
        [rgb888](int idx, uint32_t& r, uint32_t& g, uint32_t& b) {
            const int base = idx * 3;
            r = rgb888[base + 0];
            g = rgb888[base + 1];
            b = rgb888[base + 2];
        });
}

extern "C" void img_gray8_to_chw32(
    const uint8_t* gray8,
    int src_w,
    int src_h,
    int8_t* out_chw32) {
    if (gray8 == nullptr || out_chw32 == nullptr || src_w <= 0 || src_h <= 0) {
        return;
    }

    downsample_crop_to_chw32(src_w, src_h, out_chw32,
        [gray8](int idx, uint32_t& r, uint32_t& g, uint32_t& b) {
            const uint32_t v = gray8[idx];
            r = v;
            g = v;
            b = v;
        });
}
