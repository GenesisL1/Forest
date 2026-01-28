# C++ Local Trainer (GL1F/GL1X)

This folder contains a drop-in C++ training worker that mirrors `train_gl1f.py`, but runs much faster.

## Build (Linux/macOS)

From the project root:

```bash
./build_cpp_trainer.sh
```

This produces:

- `./train_gl1f_cpp`

## Build (Windows)

From the project root (with a C++17 g++ in PATH, e.g. MinGW-w64):

```bat
build_cpp_trainer.bat
```

This produces:

- `train_gl1f_cpp.exe`

## Run the local trainer server

The UI talks to `local_trainer_server.py`. Start it from the project root:

```bash
python local_trainer_server.py
```

The UI has three engines:

- **Browser** (no server)
- **Python** (local trainer server runs `train_gl1f.py`)
- **C++** (local trainer server runs `train_gl1f_cpp`)

By default the server looks for `./train_gl1f_cpp` (or `train_gl1f_cpp.exe` on Windows).
You can override with:

```bash
python local_trainer_server.py --cpp-train-bin /path/to/train_gl1f_cpp
```
