idf_component_register(
    SRCS "main.cpp" "snn_engine.cpp" "snn_pipeline.cpp"
    INCLUDE_DIRS "." "../include" "../test_samples"
    REQUIRES esp_timer freertos
)
