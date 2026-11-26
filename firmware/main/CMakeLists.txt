idf_component_register(
    SRCS "main.cpp" "snn_engine.cpp" "snn_pipeline.cpp" "image_preprocess.cpp" "camera_capture.cpp" "dvs_encoder.cpp"
    INCLUDE_DIRS "." "../include" "../test_samples"
    REQUIRES esp_timer freertos
)
