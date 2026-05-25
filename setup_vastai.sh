#!/bin/bash
# Vast.ai RTX 4090 setup script for Clairvoyant benchmark
# Run once after SSH into the instance:  bash setup_vastai.sh
set -e

echo "=== [1/6] System deps ==="
apt-get update -qq
apt-get install -y -qq curl wget git golang-go

echo "=== [2/6] Install libonnxruntime ==="
ONNX_VERSION="1.18.1"
wget -q "https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-linux-x64-${ONNX_VERSION}.tgz"
tar -xzf "onnxruntime-linux-x64-${ONNX_VERSION}.tgz"
cp "onnxruntime-linux-x64-${ONNX_VERSION}/lib/libonnxruntime.so.${ONNX_VERSION}" /usr/local/lib/
ln -sf "/usr/local/lib/libonnxruntime.so.${ONNX_VERSION}" /usr/local/lib/libonnxruntime.so
ldconfig
echo "libonnxruntime installed"

echo "=== [3/6] Install Ollama ==="
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &>/tmp/ollama.log &
sleep 5
echo "Ollama started"

echo "=== [4/6] Pull models ==="
ollama pull gemma3:4b
ollama pull llama3.1:8b
echo "Models ready"

echo "=== [5/6] Clone repo and build Clairvoyant ==="
git clone https://github.com/Aravind0403/clairvoyant-scheduler.git /opt/clairvoyant
cd /opt/clairvoyant/scheduler

# Set ONNX lib path
export ONNX_LIB_PATH=/usr/local/lib/libonnxruntime.so
export CGO_LDFLAGS="-L/usr/local/lib"
export CGO_CFLAGS="-I$(ls -d /root/onnxruntime-linux-x64-*/include 2>/dev/null | head -1)"

go build -o clairvoyant ./cmd/main.go 2>/tmp/go_build.log \
  || go build -o clairvoyant . 2>>/tmp/go_build.log \
  || { echo "Build failed — check /tmp/go_build.log"; cat /tmp/go_build.log; }

echo "Clairvoyant built"

echo "=== [6/6] Done ==="
echo ""
echo "Next steps:"
echo "  1. Start Clairvoyant:"
echo "     cd /opt/clairvoyant/scheduler"
echo "     ONNX_MODEL_PATH=../model/predictor.onnx ONNX_LIB_PATH=/usr/local/lib/libonnxruntime.so STARVATION_TIMEOUT_SEC=15 ./clairvoyant &"
echo ""
echo "  2. Run benchmark (both conditions, 5 runs each):"
echo "     cd /opt/clairvoyant/profiler"
echo "     python3 benchmark.py --model gemma3:4b --runs 5"
echo "     python3 benchmark.py --model llama3.1:8b --runs 5 --out benchmark_results_llama.csv --summary benchmark_summary_llama.csv"
echo ""
echo "  3. Copy results back:"
echo "     scp -P <port> root@<ip>:/opt/clairvoyant/profiler/benchmark_summary*.csv ."
