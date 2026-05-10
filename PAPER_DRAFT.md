# Clairvoyant: SLA-Predictive Inference Batching for Head-of-Line Blocking Mitigation in LLM Serving

## 1. Abstract (~150 words)
*(To be written last)*

## 2. Introduction
Modern Large Language Model (LLM) serving infrastructure is fundamentally bottlenecked by the extreme variance in output generation lengths. A typical production workload mixes short, factual queries (e.g., "What is the capital of France?", requiring <10 tokens) with long, complex generation tasks (e.g., "Write a React tic-tac-toe application", requiring >1000 tokens). In single-concurrent-request deployment scenarios—such as edge AI, local enterprise servers, and resource-constrained environments running quantized Small Language Models (SLMs)—requests are processed via a standard First-Come-First-Served (FCFS) queue. This creates severe Head-of-Line Blocking (HOLB): a 2-second factual query can be delayed for minutes while waiting behind a long code generation task.

While cloud-scale LLM deployments (e.g., vLLM, Orca) mitigate HOLB natively using token-level continuous batching, these solutions require massive VRAM overhead to maintain concurrent KV-caches. For the rapidly growing ecosystem of memory-constrained edge and local deployments, continuous batching is impossible. These deployments must rely on serial or low-concurrency execution, leaving them highly vulnerable to HOLB.

Classical queueing theory provides a well-known solution to HOLB: Shortest-Job-First (SJF) scheduling. However, SJF requires knowing the processing time (output token length) of a job *before* it begins execution. For LLMs, this presents a paradox: the output length is unknown until the model finishes generating it. Previous work has attempted to estimate LLM complexity, but usually requires running the prompt through the heavy LLM itself, defeating the purpose of a low-latency scheduler.

In this paper, we present **Clairvoyant**, a drop-in, SLA-predictive inference proxy that mathematically eliminates HOLB in serial LLM deployments with effectively zero overhead. Clairvoyant introduces a novel predictive routing architecture that intercepts API requests and uses a highly optimized ONNX XGBoost classifier to predict the output length class (Short, Medium, Long) of a prompt based entirely on 19 lightweight lexical features. 

Our contributions are as follows:
1. **Zero-Overhead Predictive Routing Architecture:** We design a language-agnostic sidecar proxy that intercepts OpenAI-compatible requests and performs feature extraction and output-length prediction in <0.029ms, requiring zero modifications to the underlying LLM backend (e.g., Ollama).
2. **Lexical Sufficiency for Generative Intent:** We demonstrate that computationally expensive embeddings are unnecessary for length prediction. Using only 19 lexical features (such as `has_code_keyword` and specific `instruction_verbs`), Clairvoyant achieves >76% cross-distribution ranking accuracy, successfully generalizing across instruction-following datasets.
3. **Empirical Queue Management:** We implement an SJF min-heap with a mathematically derived starvation timeout ($\tau = 3 \times \mu_{short}$) that prevents indefinite blocking of long jobs. In real-world GPU benchmarks on an RTX 4090, Clairvoyant reduces average latency for short requests by 44% compared to FCFS, achieving 100% deterministic SJF ordering.
4. **Operational Boundaries of SJF:** We provide an intellectually honest evaluation of Clairvoyant against continuous-batching engines (vLLM), demonstrating that request-level SJF scheduling is an anti-pattern for concurrent backends, formally defining the scope of predictive proxies to resource-constrained and serial deployments.

## 3. Background (~0.5 page)
- HOLB taxonomy (Layer 1 vs Layer 2)
- SJF theory, starvation timeout
- Why Ollama/serial dispatch is the right scope

## 4. System Design (~1.5 pages)
- Sidecar proxy architecture
- Feature extraction pipeline
- ONNX inference (0.015ms latency)
- Go scheduler + starvation timeout

## 5. ML Predictor (~1 page)
- Feature engineering (19 features, two tiers)
- Training (XGBoost, 3-class labels)
- Dataset selection — the Long-class starvation finding

## 6. Evaluation (~2.5 pages)
- §6.1 Dataset study (5 families, starvation table)
- §6.2 Cross-distribution matrix + ablation
- §6.3 GPU benchmark (Vast.ai — FCFS vs SJF)

## 7. Related Work (~0.5 page)
- S3 (complementary, not competing)
- Orca, vLLM (Layer 2 — different problem)
- SJF in classical queueing theory

## 8. Limitations (~0.25 page)
- Serial dispatch only, vLLM out of scope
- Stage 3 regression note (honest)
- len//4 approximation

## 9. Conclusion (~0.25 page)
