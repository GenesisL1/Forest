#!/usr/bin/env bash
set -euo pipefail

# Build the C++ local trainer binary used by local_trainer_server.py (engine=cpp)
# Output: ./train_gl1f_cpp

cd "$(dirname "$0")"

if ! command -v g++ >/dev/null 2>&1; then
  echo "g++ not found. Install a C++17-capable compiler (g++/clang++) and retry." >&2
  exit 1
fi

g++ -O3 -std=c++17 -o train_gl1f_cpp cpp/train_gl1f_cpp.cpp
chmod +x train_gl1f_cpp

echo "Built: $(pwd)/train_gl1f_cpp"
