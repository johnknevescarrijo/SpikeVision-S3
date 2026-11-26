#include "camera_capture.h"
#include "image_preprocess.h"

#if defined(ESP_PLATFORM) && __has_include("esp_camera.h")
    #define SPIKEVISION_HAS_ESP_CAMERA 1
    #include "esp_camera.h"
    #include "esp_timer.h"
#else
    #define SPIKEVISION_HAS_ESP_CAMERA 0
    #if defined(ESP_PLATFORM)
        #include "esp_timer.h"
    #else
        #include <chrono>
    #endif
#endif

namespace {

static bool s_cam_initialized = false;
static bool s_cam_hw_active = false;
static camera_capture_cfg_t s_cam_cfg = {};
static uint32_t s_synth_frame_counter = 0;

static uint64_t get_time_us(void) {
#if defined(ESP_PLATFORM)
    return static_cast<uint64_t>(esp_timer_get_time());
#else
    using namespace std::chrono;
    return static_cast<uint64_t>(
        duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count());
#endif
}

static camera_capture_pins_t get_preset_pins(camera_board_preset_t preset) {
    switch (preset) {
        case CAM_PRESET_ESP32S3_EYE:
            return {
                -1, -1, 15, 4, 5,
                16, 17, 18, 12, 10, 8, 9, 11,
                6, 7, 13
            };
        case CAM_PRESET_XIAO_ESP32S3_SENSE:
            return {
                -1, -1, 10, 40, 39,
                48, 11, 12, 14, 16, 18, 17, 15,
                38, 47, 13
            };
        case CAM_PRESET_FREENOVE_S3_WROOM:
            return {
                -1, -1, 15, 4, 5,
                16, 17, 18, 12, 10, 8, 9, 11,
                6, 7, 13
            };
        case CAM_PRESET_CUSTOM:
        default:
            return {
                -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1
            };
    }
}

static void generate_synthetic_chw32(int8_t* out_chw32, uint32_t frame_idx, bool swap_bytes) {
    /* Buffer compacto 32x32 RGB565 na stack (2048 bytes) para exercitar img_rgb565_to_chw32 */
    uint16_t synth_rgb565[IMG_PREPROC_WIDTH * IMG_PREPROC_HEIGHT];
    const uint32_t shift = frame_idx & 0x1Fu;

    for (int y = 0; y < IMG_PREPROC_HEIGHT; ++y) {
        for (int x = 0; x < IMG_PREPROC_WIDTH; ++x) {
            const uint16_t r5 = static_cast<uint16_t>((x + shift) & 0x1Fu);
            const uint16_t g6 = static_cast<uint16_t>(((y * 2) + shift) & 0x3Fu);
            const uint16_t b5 = static_cast<uint16_t>(((x ^ y) + shift) & 0x1Fu);
            uint16_t px = static_cast<uint16_t>((r5 << 11) | (g6 << 5) | b5);
            if (swap_bytes) {
                px = static_cast<uint16_t>((px >> 8) | (px << 8));
            }
            synth_rgb565[y * IMG_PREPROC_WIDTH + x] = px;
        }
    }

    img_rgb565_to_chw32(synth_rgb565, IMG_PREPROC_WIDTH, IMG_PREPROC_HEIGHT, swap_bytes, out_chw32);
}

}  // namespace

extern "C" camera_capture_cfg_t camera_capture_default_cfg(camera_board_preset_t preset) {
    camera_capture_cfg_t cfg = {};
    cfg.preset = preset;
    cfg.xclk_freq_hz = 20000000;
    cfg.swap_rgb565_bytes = true;
    cfg.use_qqvga = false;
    cfg.enable_synthetic_fallback = true;
    cfg.custom_pins = get_preset_pins(preset);
    return cfg;
}

extern "C" bool camera_capture_has_hardware_driver(void) {
#if SPIKEVISION_HAS_ESP_CAMERA
    return true;
#else
    return false;
#endif
}

extern "C" bool camera_capture_init(const camera_capture_cfg_t* cfg) {
    if (s_cam_initialized) {
        camera_capture_deinit();
    }

    s_cam_cfg = (cfg != nullptr) ? *cfg : camera_capture_default_cfg(CAM_PRESET_ESP32S3_EYE);
    if (s_cam_cfg.xclk_freq_hz <= 0) {
        s_cam_cfg.xclk_freq_hz = 20000000;
    }

#if SPIKEVISION_HAS_ESP_CAMERA
    const camera_capture_pins_t pins = (s_cam_cfg.preset == CAM_PRESET_CUSTOM)
        ? s_cam_cfg.custom_pins
        : get_preset_pins(s_cam_cfg.preset);

    camera_config_t esp_cfg = {};
    esp_cfg.pin_pwdn = pins.pwdn;
    esp_cfg.pin_reset = pins.reset;
    esp_cfg.pin_xclk = pins.xclk;
    esp_cfg.pin_sccb_sda = pins.siod;
    esp_cfg.pin_sccb_scl = pins.sioc;
    esp_cfg.pin_d7 = pins.d7;
    esp_cfg.pin_d6 = pins.d6;
    esp_cfg.pin_d5 = pins.d5;
    esp_cfg.pin_d4 = pins.d4;
    esp_cfg.pin_d3 = pins.d3;
    esp_cfg.pin_d2 = pins.d2;
    esp_cfg.pin_d1 = pins.d1;
    esp_cfg.pin_d0 = pins.d0;
    esp_cfg.pin_vsync = pins.vsync;
    esp_cfg.pin_href = pins.href;
    esp_cfg.pin_pclk = pins.pclk;

    esp_cfg.xclk_freq_hz = s_cam_cfg.xclk_freq_hz;
    esp_cfg.ledc_timer = LEDC_TIMER_0;
    esp_cfg.ledc_channel = LEDC_CHANNEL_0;
    esp_cfg.pixel_format = PIXFORMAT_RGB565;
    esp_cfg.frame_size = s_cam_cfg.use_qqvga ? FRAMESIZE_QQVGA : FRAMESIZE_96X96;
    esp_cfg.jpeg_quality = 12;
    esp_cfg.fb_count = 2;
    esp_cfg.fb_location = CAMERA_FB_IN_PSRAM;
    esp_cfg.grab_mode = CAMERA_GRAB_LATEST;

    if (esp_camera_init(&esp_cfg) == ESP_OK) {
        s_cam_hw_active = true;
        s_cam_initialized = true;
        return true;
    }

    if (!s_cam_cfg.enable_synthetic_fallback) {
        s_cam_hw_active = false;
        s_cam_initialized = false;
        return false;
    }
#endif

    if (!s_cam_cfg.enable_synthetic_fallback) {
        s_cam_hw_active = false;
        s_cam_initialized = false;
        return false;
    }

    s_cam_hw_active = false;
    s_cam_initialized = true;
    s_synth_frame_counter = 0;
    return true;
}

extern "C" bool camera_capture_grab_chw32(int8_t* out_chw32, uint32_t* out_capture_us) {
    if (!s_cam_initialized || out_chw32 == nullptr) {
        return false;
    }

    const uint64_t t_start = get_time_us();

#if SPIKEVISION_HAS_ESP_CAMERA
    if (s_cam_hw_active) {
        camera_fb_t* fb = esp_camera_fb_get();
        if (fb == nullptr) {
            return false;
        }

        bool ok = true;
        if (fb->format == PIXFORMAT_RGB565) {
            img_rgb565_to_chw32(
                reinterpret_cast<const uint16_t*>(fb->buf),
                static_cast<int>(fb->width),
                static_cast<int>(fb->height),
                s_cam_cfg.swap_rgb565_bytes,
                out_chw32);
        } else if (fb->format == PIXFORMAT_RGB888) {
            img_rgb888_to_chw32(
                fb->buf,
                static_cast<int>(fb->width),
                static_cast<int>(fb->height),
                out_chw32);
        } else if (fb->format == PIXFORMAT_GRAYSCALE) {
            img_gray8_to_chw32(
                fb->buf,
                static_cast<int>(fb->width),
                static_cast<int>(fb->height),
                out_chw32);
        } else {
            ok = false;
        }

        esp_camera_fb_return(fb);
        if (!ok) {
            return false;
        }

        if (out_capture_us != nullptr) {
            *out_capture_us = static_cast<uint32_t>(get_time_us() - t_start);
        }
        return true;
    }
#endif

    generate_synthetic_chw32(out_chw32, s_synth_frame_counter++, s_cam_cfg.swap_rgb565_bytes);
    if (out_capture_us != nullptr) {
        *out_capture_us = static_cast<uint32_t>(get_time_us() - t_start);
    }
    return true;
}

extern "C" void camera_capture_deinit(void) {
    if (!s_cam_initialized) {
        return;
    }

#if SPIKEVISION_HAS_ESP_CAMERA
    if (s_cam_hw_active) {
        esp_camera_deinit();
    }
#endif

    s_cam_hw_active = false;
    s_cam_initialized = false;
}
