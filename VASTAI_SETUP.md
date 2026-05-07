# Vast.ai Setup Reference — Clairvoyant Scheduler

Accurate as of 2026-04-30. RTX 4090, vLLM template.

---

## 1. Instance Setup

**Template:** vLLM (not NVIDIA CUDA)
**GPU:** 1× RTX 4090, 24GB VRAM
**Reliability:** ≥ 99.5%

---

## 2. SSH In

```bash
ssh -i ~/.ssh/id_ed25519_vast -p <port> root@<ip>
```

Port and IP come from the Vast.ai Connect button on the instance page.

---

## 3. Start vLLM (on the instance)

```bash
export HF_TOKEN=your_token_here

vllm serve google/gemma-3-4b-it \
  --port 8000 \
  --max-model-len 4096
```

First run downloads the model (~3GB). Subsequent runs use cache.
Wait for: `INFO: Application startup complete.`

Run in background for benchmarking:
```bash
export HF_TOKEN=your_token_here

nohup vllm serve google/gemma-3-4b-it \
  --port 8000 \
  --max-model-len 4096 > /tmp/vllm.log 2>&1 &

# Wait and verify
sleep 30
curl -s http://localhost:8000/health
```

---

## 4. Verify vLLM

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"google/gemma-3-4b-it","messages":[{"role":"user","content":"hi"}]}' \
  | head -c 200
```

---

## 5. Transfer Files from Mac

Run this on your **Mac**, not the instance:

```bash
# From the scheduler directory
cd /Users/aravindsundaresan/Development/Research_Clavoiyrant/clairvoyant/scheduler

scp -i ~/.ssh/id_ed25519_vast -P <port> \
  test_queue_ordering.py \
  root@<ip>:~/clairvoyant/

scp -i ~/.ssh/id_ed25519_vast -P <port> \
  ../model/predictor.onnx \
  root@<ip>:~/clairvoyant/
```

---

## 6. Build Scheduler on Instance

```bash
# Install Go
wget -q https://go.dev/dl/go1.21.13.linux-amd64.tar.gz
tar -C /usr/local -xzf go1.21.13.linux-amd64.tar.gz
export PATH=$PATH:/usr/local/go/bin

# Install ONNX Runtime
wget -q https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-linux-x64-1.20.1.tgz
tar xf onnxruntime-linux-x64-1.20.1.tgz
cp onnxruntime-linux-x64-1.20.1/lib/libonnxruntime.so.1.20.1 /usr/local/lib/libonnxruntime.so
ldconfig

# Transfer source and build
# (source already transferred via scp in step 5 — transfer the scheduler/ dir too)
cd ~/clairvoyant/scheduler
go build -o clairvoyant-scheduler ./cmd/main.go
```

Transfer scheduler source from Mac:
```bash
scp -i ~/.ssh/id_ed25519_vast -P <port> -r \
  /Users/aravindsundaresan/Development/Research_Clavoiyrant/clairvoyant/scheduler \
  root@<ip>:~/clairvoyant/
```

---

## 7. Run Stage 3 Benchmark (vLLM + SJF)

```bash
cd ~/clairvoyant/scheduler

# Kill anything on 8080
kill $(lsof -ti :8080) 2>/dev/null; sleep 1

# Start scheduler pointing at vLLM
BACKEND_URL=http://localhost:8000 \
ONNX_MODEL_PATH=~/clairvoyant/predictor.onnx \
ONNX_LIB_PATH=/usr/local/lib/libonnxruntime.so \
STARVATION_TIMEOUT_SEC=90 \
./clairvoyant-scheduler > /tmp/scheduler.log 2>&1 &

sleep 3
grep "listening" /tmp/scheduler.log

# Run 5 times
./SJF_run_benchmark.sh 5
```

For FCFS baseline against vLLM (Stage 3 baseline):
```bash
for i in $(seq 1 5); do
  echo "=== FCFS Run $i / 5 ==="
  python3 ~/clairvoyant/scheduler/test_queue_ordering.py \
    --url http://localhost:8000/v1/chat/completions \
    --model google/gemma-3-4b-it
  sleep 5
done
```

---

## 8. Starvation Timeout Rule

```
τ = 3 × expected_short_latency (peak)
```

| Hardware | SHORT latency range | τ setting |
|----------|--------------------:|:---------:|
| Apple M1 (Ollama) | 30–80s | 120s |
| RTX 4090 (Ollama) | 8–40s | 90s |
| RTX 4090 (vLLM) | TBD | TBD (expect lower — measure first) |

---

## 9. Save Results Before Destroying

```bash
cat /tmp/scheduler.log > ~/clairvoyant/stage3_scheduler.log
# Copy benchmark output to a file during the run:
./SJF_run_benchmark.sh 5 | tee ~/clairvoyant/stage3_results.txt
```

Then scp results back to Mac:
```bash
scp -i ~/.ssh/id_ed25519_vast -P <port> \
  root@<ip>:~/clairvoyant/stage3_results.txt \
  /Users/aravindsundaresan/Development/Research_Clavoiyrant/clairvoyant/
```

**Destroy the instance immediately after** — storage costs $0.07/day stopped.
