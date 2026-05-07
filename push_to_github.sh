#!/bin/bash
set -e

REPO_URL="https://github.com/Aravind0403/clairvoyant-scheduler.git"

echo "==> Initialising git repo..."
git init

echo "==> Setting up remote..."
git remote add origin "$REPO_URL"

echo "==> Staging files..."
git add .

echo "==> Files to be committed:"
git status --short

echo ""
echo "==> Committing..."
git commit -m "Initial commit: Clairvoyant sidecar scheduler v0.55

- Go HTTP sidecar proxy with SJF priority queue and aging monitor
- ONNX-exported XGBoost predictor: 19 lexical features, 0.029ms inference
- 76.29% ranking accuracy, 100% SJF ordering on Apple M1
- Starvation timeout formula: tau = 3x expected short latency
- Documents XGBoost 2.x ONNX export bug + fix
- Paper in preparation: MLSys 2026"

echo "==> Pushing to GitHub..."
git branch -M main
git push -u origin main

echo ""
echo "Done. Repo live at: https://github.com/Aravind0403/clairvoyant-scheduler"
