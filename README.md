# Ferris Mark VK

A Windows Vulkan 2D sprite rendering benchmark written in Rust.

![Ferris Mark VK](ferris-mark-vk.gif)

Inspired by Jacob's [gophermark](https://github.com/unitoftime/experiments/tree/master/gophermark) ([YouTube](https://www.youtube.com/watch?v=ZuVyxnpMZO4))
and [ebitengine-bunny-mark](https://github.com/sedyh/ebitengine-bunny-mark) (Artem Sedykh). With sprite batching inspired by [How I learned Vulkan and wrote a small game engine with it](https://edw.is/learning-vulkan/#drawing-many-sprites) (Elias Daler).

## Vulkan SDK recommended

Install on Windows 11 with winget:

```
winget install vulkansdk
```

## Run it

To run with 1000 sprites:

```
cargo run --release 1000
```

### Run Benchmark

With Git Bash on Windows:

```bash
./benchmark.sh
```

This will:
1. Build the app in release mode
2. Test various sprite counts (1000, 2000, 3000, etc.)
3. Summarize the results

### Ferris Mark VK Benchmark Results

Ryzen 9700X and Radeon 9070 XT running Radeon driver version 25.6.1.

Sprites | FPS     | Sprites/sec
--------|---------|------------
  10,000 |  2026.2 | 20,262,404
  20,000 |  1244.2 | 24,884,168
  30,000 |   914.9 | 27,445,988
  40,000 |   711.7 | 28,467,288
  50,000 |   580.7 | 29,033,536
  60,000 |   491.8 | 29,509,286
  70,000 |   420.6 | 29,443,856
  80,000 |   372.2 | 29,772,974
  90,000 |   336.0 | 30,241,962
 100,000 |   305.0 | 30,503,418
 150,000 |   196.3 | 29,442,926
 200,000 |   156.5 | 31,298,406
