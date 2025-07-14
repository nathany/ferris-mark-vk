#!/bin/bash
cargo build --release
rm -f benchmark_results.txt
for sprites in 5000 6000 7000 8000 9000 10000 20000 30000 40000 50000 60000 70000 80000 90000 100000 150000 200000; do
    echo "Testing $sprites sprites..."
    if [ -f "./target/release/ferris-mark-vk.exe" ]; then
        ./target/release/ferris-mark-vk.exe $sprites --frames 10100 >> benchmark_results.txt 2>&1
    else
        ./target/release/ferris-mark-vk $sprites --frames 10100 >> benchmark_results.txt 2>&1
    fi
done
echo ""
echo "Checking benchmark_results.txt..."
echo "BENCHMARK_RESULT lines found:"
grep "BENCHMARK_RESULT:" benchmark_results.txt || echo "No BENCHMARK_RESULT lines found"
echo ""
echo "Last few lines of benchmark_results.txt:"
tail -10 benchmark_results.txt
echo ""
echo "Ferris Mark VK Benchmark Results"
echo "================================"
echo "Sprites | FPS     | Sprites/sec"
echo "--------|---------|------------"
grep "BENCHMARK_RESULT:" benchmark_results.txt | sed 's/BENCHMARK_RESULT: \([0-9]*\) sprites, \([0-9.]*\) FPS, \([0-9]*\) sprites\/sec/\1 \2 \3/' | sort -n | while read sprites fps spritessec; do printf "%7s | %7s | %s\n" "$sprites" "$fps" "$spritessec"; done
