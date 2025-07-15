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
   5000 |  2351.8 | 11759210
   6000 |  2340.9 | 14045133
   7000 |  2286.1 | 16002482
   8000 |  2278.9 | 18231350
   9000 |  2042.4 | 18381666
  10000 |  2026.2 | 20262404
  20000 |  1244.2 | 24884168
  30000 |   914.9 | 27445988
  40000 |   711.7 | 28467288
  50000 |   580.7 | 29033536
  60000 |   491.8 | 29509286
  70000 |   420.6 | 29443856
  80000 |   372.2 | 29772974
  90000 |   336.0 | 30241962
 100000 |   305.0 | 30503418
 150000 |   196.3 | 29442926
 200000 |   156.5 | 31298406
