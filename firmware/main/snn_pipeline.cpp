/**
 * SpikeVision-S3: Implementação do Pipeline Dual-Core FreeRTOS / Host Thread.
 */

#include "snn_pipeline.h"
#include <string.h>

typedef struct {
    uint32_t frame_id;
    uint8_t data[SNN_PIPELINE_FRAME_BYTES];
    bool stop_sentinel;
} snn_pipeline_input_t;

static snn_early_exit_cfg_t s_early_exit_cfg = { false, 2, 0 };
static bool s_has_early_exit = false;

#if defined(ESP_PLATFORM)

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "freertos/semphr.h"

#define SNN_PIPELINE_TASK_STACK_SIZE 8192
#define SNN_PIPELINE_TASK_PRIORITY   5
#define SNN_PIPELINE_INFERENCE_CORE  1

static QueueHandle_t s_in_queue = nullptr;
static QueueHandle_t s_out_queue = nullptr;
static TaskHandle_t s_worker_task = nullptr;
static SemaphoreHandle_t s_done_sem = nullptr;
static volatile bool s_running = false;

static TickType_t ms_to_ticks(uint32_t timeout_ms) {
    if (timeout_ms == 0xFFFFFFFFu) {
        return portMAX_DELAY;
    }
    return pdMS_TO_TICKS(timeout_ms);
}

static void snn_pipeline_worker_task(void* arg) {
    (void)arg;
    snn_pipeline_input_t item;

    while (true) {
        if (xQueueReceive(s_in_queue, &item, portMAX_DELAY) != pdTRUE) {
            continue;
        }
        if (item.stop_sentinel) {
            break;
        }

        snn_pipeline_output_t out;
        out.frame_id = item.frame_id;
        snn_engine_infer_adaptive(
            item.data,
            &out.result,
            s_has_early_exit ? &s_early_exit_cfg : nullptr);

        xQueueSend(s_out_queue, &out, portMAX_DELAY);
    }

    if (s_done_sem != nullptr) {
        xSemaphoreGive(s_done_sem);
    }
    s_worker_task = nullptr;
    vTaskDelete(nullptr);
}

bool snn_pipeline_start(const snn_early_exit_cfg_t* early_exit_cfg) {
    if (s_running) {
        return false;
    }

    if (early_exit_cfg != nullptr) {
        s_early_exit_cfg = *early_exit_cfg;
        s_has_early_exit = true;
    } else {
        s_has_early_exit = false;
    }

    snn_engine_init();

    s_in_queue = xQueueCreate(SNN_PIPELINE_QUEUE_DEPTH, sizeof(snn_pipeline_input_t));
    s_out_queue = xQueueCreate(SNN_PIPELINE_QUEUE_DEPTH, sizeof(snn_pipeline_output_t));
    s_done_sem = xSemaphoreCreateBinary();

    if (s_in_queue == nullptr || s_out_queue == nullptr || s_done_sem == nullptr) {
        if (s_in_queue != nullptr) { vQueueDelete(s_in_queue); s_in_queue = nullptr; }
        if (s_out_queue != nullptr) { vQueueDelete(s_out_queue); s_out_queue = nullptr; }
        if (s_done_sem != nullptr) { vSemaphoreDelete(s_done_sem); s_done_sem = nullptr; }
        return false;
    }

    s_running = true;
    BaseType_t ok = xTaskCreatePinnedToCore(
        snn_pipeline_worker_task,
        "snn_infer_c1",
        SNN_PIPELINE_TASK_STACK_SIZE,
        nullptr,
        SNN_PIPELINE_TASK_PRIORITY,
        &s_worker_task,
        SNN_PIPELINE_INFERENCE_CORE);

    if (ok != pdPASS) {
        s_running = false;
        vQueueDelete(s_in_queue); s_in_queue = nullptr;
        vQueueDelete(s_out_queue); s_out_queue = nullptr;
        vSemaphoreDelete(s_done_sem); s_done_sem = nullptr;
        return false;
    }

    return true;
}

bool snn_pipeline_submit(uint32_t frame_id, const int8_t* input_chw, uint32_t timeout_ms) {
    if (!s_running || input_chw == nullptr || s_in_queue == nullptr) {
        return false;
    }

    snn_pipeline_input_t item;
    item.frame_id = frame_id;
    item.stop_sentinel = false;
    memcpy(item.data, input_chw, SNN_PIPELINE_FRAME_BYTES);

    return xQueueSend(s_in_queue, &item, ms_to_ticks(timeout_ms)) == pdTRUE;
}

bool snn_pipeline_receive(snn_pipeline_output_t* out, uint32_t timeout_ms) {
    if (out == nullptr || s_out_queue == nullptr) {
        return false;
    }
    return xQueueReceive(s_out_queue, out, ms_to_ticks(timeout_ms)) == pdTRUE;
}

void snn_pipeline_stop(void) {
    if (!s_running) {
        return;
    }
    s_running = false;

    if (s_in_queue != nullptr) {
        snn_pipeline_input_t sentinel;
        memset(&sentinel, 0, sizeof(sentinel));
        sentinel.stop_sentinel = true;
        xQueueSend(s_in_queue, &sentinel, portMAX_DELAY);
    }

    if (s_done_sem != nullptr) {
        xSemaphoreTake(s_done_sem, portMAX_DELAY);
        vSemaphoreDelete(s_done_sem);
        s_done_sem = nullptr;
    }
    if (s_in_queue != nullptr) {
        vQueueDelete(s_in_queue);
        s_in_queue = nullptr;
    }
    if (s_out_queue != nullptr) {
        vQueueDelete(s_out_queue);
        s_out_queue = nullptr;
    }
}

#else /* Host Linux (std::thread + ring buffer estático) */

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>

static snn_pipeline_input_t s_in_ring[SNN_PIPELINE_QUEUE_DEPTH];
static snn_pipeline_output_t s_out_ring[SNN_PIPELINE_QUEUE_DEPTH];
static size_t s_in_head = 0;
static size_t s_in_tail = 0;
static size_t s_in_count = 0;
static size_t s_out_head = 0;
static size_t s_out_tail = 0;
static size_t s_out_count = 0;

static std::mutex s_mtx;
static std::condition_variable s_cv_in;
static std::condition_variable s_cv_out;
static std::thread s_worker;
static bool s_running = false;
static bool s_stop_requested = false;

static void snn_pipeline_worker_thread(void) {
    while (true) {
        snn_pipeline_input_t item;
        {
            std::unique_lock<std::mutex> lock(s_mtx);
            s_cv_in.wait(lock, []() {
                return s_in_count > 0 || s_stop_requested;
            });

            if (s_stop_requested && s_in_count == 0) {
                break;
            }

            item = s_in_ring[s_in_head];
            s_in_head = (s_in_head + 1) % SNN_PIPELINE_QUEUE_DEPTH;
            s_in_count--;
        }
        s_cv_in.notify_all();

        if (item.stop_sentinel) {
            break;
        }

        snn_pipeline_output_t out;
        out.frame_id = item.frame_id;
        snn_engine_infer_adaptive(
            item.data,
            &out.result,
            s_has_early_exit ? &s_early_exit_cfg : nullptr);

        {
            std::unique_lock<std::mutex> lock(s_mtx);
            s_cv_out.wait(lock, []() {
                return s_out_count < SNN_PIPELINE_QUEUE_DEPTH || s_stop_requested;
            });

            if (s_stop_requested) {
                break;
            }

            s_out_ring[s_out_tail] = out;
            s_out_tail = (s_out_tail + 1) % SNN_PIPELINE_QUEUE_DEPTH;
            s_out_count++;
        }
        s_cv_out.notify_all();
    }
}

bool snn_pipeline_start(const snn_early_exit_cfg_t* early_exit_cfg) {
    std::lock_guard<std::mutex> lock(s_mtx);
    if (s_running) {
        return false;
    }

    if (early_exit_cfg != nullptr) {
        s_early_exit_cfg = *early_exit_cfg;
        s_has_early_exit = true;
    } else {
        s_has_early_exit = false;
    }

    s_in_head = 0;
    s_in_tail = 0;
    s_in_count = 0;
    s_out_head = 0;
    s_out_tail = 0;
    s_out_count = 0;
    s_stop_requested = false;
    s_running = true;

    snn_engine_init();
    s_worker = std::thread(snn_pipeline_worker_thread);
    return true;
}

bool snn_pipeline_submit(uint32_t frame_id, const int8_t* input_chw, uint32_t timeout_ms) {
    if (input_chw == nullptr) {
        return false;
    }

    std::unique_lock<std::mutex> lock(s_mtx);
    if (!s_running || s_stop_requested) {
        return false;
    }

    auto pred = []() {
        return s_in_count < SNN_PIPELINE_QUEUE_DEPTH || s_stop_requested;
    };

    if (timeout_ms == 0xFFFFFFFFu) {
        s_cv_in.wait(lock, pred);
    } else {
        if (!s_cv_in.wait_for(lock, std::chrono::milliseconds(timeout_ms), pred)) {
            return false;
        }
    }

    if (s_stop_requested || !s_running) {
        return false;
    }

    snn_pipeline_input_t& slot = s_in_ring[s_in_tail];
    slot.frame_id = frame_id;
    slot.stop_sentinel = false;
    memcpy(slot.data, input_chw, SNN_PIPELINE_FRAME_BYTES);

    s_in_tail = (s_in_tail + 1) % SNN_PIPELINE_QUEUE_DEPTH;
    s_in_count++;

    lock.unlock();
    s_cv_in.notify_all();
    return true;
}

bool snn_pipeline_receive(snn_pipeline_output_t* out, uint32_t timeout_ms) {
    if (out == nullptr) {
        return false;
    }

    std::unique_lock<std::mutex> lock(s_mtx);
    auto pred = []() {
        return s_out_count > 0 || !s_running;
    };

    if (timeout_ms == 0xFFFFFFFFu) {
        s_cv_out.wait(lock, pred);
    } else {
        if (!s_cv_out.wait_for(lock, std::chrono::milliseconds(timeout_ms), pred)) {
            return false;
        }
    }

    if (s_out_count == 0) {
        return false;
    }

    *out = s_out_ring[s_out_head];
    s_out_head = (s_out_head + 1) % SNN_PIPELINE_QUEUE_DEPTH;
    s_out_count--;

    lock.unlock();
    s_cv_out.notify_all();
    return true;
}

void snn_pipeline_stop(void) {
    {
        std::lock_guard<std::mutex> lock(s_mtx);
        if (!s_running && !s_worker.joinable()) {
            return;
        }
        s_stop_requested = true;
        s_running = false;
    }
    s_cv_in.notify_all();
    s_cv_out.notify_all();

    if (s_worker.joinable()) {
        s_worker.join();
    }
}

#endif
