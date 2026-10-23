# SpikeVision-S3 Firmware Makefile (Host Test & ESP-IDF Helper)

CXX ?= g++
CXXFLAGS = -O3 -Wall -Wextra -std=c++17 -Iinclude -Itest_samples -Imain
HOST_BIN = build_host/spikevision_s3_host

SRCS = main/main.cpp main/snn_engine.cpp

.PHONY: all test host clean idf idf-build idf-flash idf-monitor

all: host

host: $(HOST_BIN)

$(HOST_BIN): $(SRCS)
	@mkdir -p build_host
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(HOST_BIN)
	@echo "Binário host compilado com sucesso: $(HOST_BIN)"

test: host
	@echo "Executando bateria de validação bit-a-bit no Host Linux..."
	./$(HOST_BIN)

idf: idf-build

idf-build:
	@echo "Compilando firmware via ESP-IDF para ESP32-S3..."
	bash -c "source /home/john/esp/esp-idf/export.sh && idf.py set-target esp32s3 && idf.py build"

idf-flash:
	@echo "Gravando firmware no ESP32-S3..."
	bash -c "source /home/john/esp/esp-idf/export.sh && idf.py flash"

idf-monitor:
	@echo "Abrindo monitor serial ESP-IDF..."
	bash -c "source /home/john/esp/esp-idf/export.sh && idf.py monitor"

clean:
	rm -rf build_host build
