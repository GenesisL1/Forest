@echo off
REM Build the C++ local trainer binary used by local_trainer_server.py (engine=cpp)
REM Output: .\train_gl1f_cpp.exe
REM Requires a C++17-capable compiler in PATH (e.g., MinGW-w64 g++).

where g++ >nul 2>&1
if errorlevel 1 (
  echo g++ not found in PATH. Install MinGW-w64 (or another C++ compiler) and retry.
  exit /b 1
)

g++ -O3 -std=c++17 -o train_gl1f_cpp.exe cpp\train_gl1f_cpp.cpp

echo Built: %cd%\train_gl1f_cpp.exe
