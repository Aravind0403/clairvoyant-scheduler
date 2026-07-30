#!/usr/bin/env bash
# ==============================================================================
# setup_gcp_l4.sh — One-click setup & benchmark runner for GCP NVIDIA L4 GPU
# ==============================================================================
# Spins up Ollama, Clairvoyant Go Scheduler proxy, and runs trace_replay.py
# across arrival rates \rho \in {0.4, 0.6, 0.8}.
#
# Usage (on GCP Compute Engine Ubuntu 22.04 L4 instance):
#   chmod +x setup_gcp_l4.sh
#   ./setup_gcp_l4.sh
# ==============================================================================

set -euo pipefail

echo "=================================================="
echo " 1. Installing System Dependencies & NVIDIA Drivers"
echo "=================================================="
sudo apt-get update -y
sudo apt-get install -y git curl python3-pip python3-venv golang-go build-essential nvidia-driver-535 nvidia-utils-535 || true

# Install Ollama
if ! command -v ollama &> /dev/null; then
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
fi

echo "=================================================="
echo " 2. Setting up Python Environment"
echo "=================================================="
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install pandas numpy xgboost onnxruntime scikit-learn requests

echo "=================================================="
echo " 3. Pulling Target LLM Models into Ollama"
echo "=================================================="
ollama serve &
OLLAMA_PID=$!
sleep 5

echo "Pulling Gemma3:4b..."
ollama pull gemma3:4b || true
echo "Pulling Llama3.1:8b..."
ollama pull llama3.1:8b || true

echo "=================================================="
echo " 4. Building & Starting Clairvoyant Go Proxy"
echo "=================================================="
cd scheduler
go build -o clairvoyant ./cmd/main.go
cd ..

# Start Clairvoyant Proxy on :8080 forwarding to Ollama :11434
ONNX_LIB_PATH=$(find venv/lib -name "libonnxruntime.so*" | head -n 1)
echo "Using ONNX shared library: $ONNX_LIB_PATH"
ONNX_LIB_PATH=$ONNX_LIB_PATH ONNX_MODEL_PATH=model/predictor.onnx BACKEND_URL=http://localhost:11434 LISTEN_ADDR=:8080 ./scheduler/clairvoyant &
PROXY_PID=$!
sleep 3

echo "=================================================="
echo " 5. Running Real Workload Trace Replay Benchmarks"
echo "=================================================="
mkdir -p results

echo "--- [1/2] Running FCFS Baseline (Direct Ollama) at rho=0.8 ---"
python3 profiler/trace_replay.py \
    --endpoint http://localhost:11434/v1/chat/completions \
    --model gemma3:4b --rho 0.8 --n 150 \
    --label fcfs --save results/gcp_l4_fcfs_rho08.json

echo "--- [2/2] Running Clairvoyant SJF (Via Proxy) at rho=0.8 ---"
python3 profiler/trace_replay.py \
    --endpoint http://localhost:8080/v1/chat/completions \
    --model gemma3:4b --rho 0.8 --n 150 \
    --label sjf --save results/gcp_l4_sjf_rho08.json

echo "=================================================="
echo " Benchmark Run Complete!"
echo " Results saved to results/gcp_l4_fcfs_rho08.json and results/gcp_l4_sjf_rho08.json"
echo "=================================================="
