#!/usr/bin/env bash
# ==============================================================================
# setup_geekom_a9.sh — Automated Environment Setup & Smoke Test for GEEKOM A9 Mega
# ==============================================================================
# Hardware: AMD Ryzen AI Max+ 395 (128GB Unified Memory, ROCm)
# Usage:
#   chmod +x setup_geekom_a9.sh
#   ./setup_geekom_a9.sh
# ==============================================================================

set -euo pipefail

echo "=================================================="
echo " 1. Checking ROCm & GPU Device Access"
echo "=================================================="
if [ -e "/dev/kfd" ]; then
    echo "  ✓ /dev/kfd detected"
else
    echo "  ⚠ Warning: /dev/kfd not found. Verify ROCm user permissions."
fi

if command -v rocminfo &> /dev/null; then
    echo "  ✓ rocminfo found:"
    rocminfo | grep -E "Name:|Marketing Name:" | head -n 4 || true
else
    echo "  ⚠ rocminfo not in PATH (standard on non-root ROCm accounts)."
fi

echo "=================================================="
echo " 2. Setting up Python Virtual Environment"
echo "=================================================="
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "  ✓ Created python virtual environment in ./venv"
fi
source venv/bin/activate
pip install --upgrade pip -q
pip install numpy pandas onnxruntime requests -q
echo "  ✓ Python dependencies installed"

echo "=================================================="
echo " 3. Setting up libonnxruntime (Linux x64)"
echo "=================================================="
# Auto-detect libonnxruntime from python site-packages to avoid root install
ONNX_LIB_DIR=$(python3 -c "import onnxruntime, os; print(os.path.dirname(onnxruntime.__file__))")
if [ -f "${ONNX_LIB_DIR}/capi/libonnxruntime.so" ]; then
    export ONNX_LIB_PATH="${ONNX_LIB_DIR}/capi/libonnxruntime.so"
else
    export ONNX_LIB_PATH=$(find "${ONNX_LIB_DIR}/capi" -name "libonnxruntime.so*" | head -n 1)
fi
echo "  ✓ Using ONNX runtime library: ${ONNX_LIB_PATH}"

echo "=================================================="
echo " 4. Building Clairvoyant Go Scheduler Binary"
echo "=================================================="
cd scheduler
export CGO_ENABLED=1
go build -o clairvoyant ./cmd/main.go
echo "  ✓ Clairvoyant binary built successfully at ./scheduler/clairvoyant"
cd ..

echo "=================================================="
echo " 5. Smoke Test: Verifying Predictor Inference"
echo "=================================================="
cd scheduler
ONNX_LIB_PATH="${ONNX_LIB_PATH}" go test -v -run TestPredictorProbabilities ./internal/predictor
cd ..

echo "=================================================="
echo " Setup & Smoke Test Complete!"
echo "=================================================="
echo "To start the scheduler proxy on :8080 forwarding to llama.cpp-server (:8000):"
echo "  ONNX_LIB_PATH=\"${ONNX_LIB_PATH}\" ONNX_MODEL_PATH=model/predictor.onnx BACKEND_URL=http://localhost:8000 ./scheduler/clairvoyant"

