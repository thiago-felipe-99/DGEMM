#!/usr/bin/env python
import sys

import matplotlib.pyplot as plt


file_path = sys.argv[1]
max_ms1 = int(sys.argv[2])
max_ms2 = int(sys.argv[3])
max_ms3 = int(sys.argv[4])
max_ms4 = int(sys.argv[5])

algs = {}

with open(file_path, "r", encoding="utf-8") as file:
    for line in file:
        columns = line.strip().split(",")
        algs.setdefault(columns[0], [])
        algs[columns[0]] += [{
            "size": int(columns[1]),
            "ms": float(columns[2]),
            "float": float(columns[3])
        }]

dpi = 100
fig_width_px = 2000
fig_height_px = 1000
fig_size = (fig_width_px / dpi, fig_height_px / dpi)

fig1, (ax1, ax2) = plt.subplots(2, 1, dpi=dpi, figsize=fig_size)
fig2, (ax3, ax4) = plt.subplots(2, 1, dpi=dpi, figsize=fig_size)
fig3, (ax5, ax6) = plt.subplots(2, 1, dpi=dpi, figsize=fig_size)
fig4, (ax7, ax8) = plt.subplots(2, 1, dpi=dpi, figsize=fig_size)

ax1.set_ylabel("ms")
ax1.set_title("Tempo Para Calcular")
ax1.set_ylim(0, max_ms1)
ax2.set_ylabel("GFLOPS/segundo")
ax2.set_title("Desempenho")

ax3.set_ylabel("ms")
ax3.set_title("Tempo Para Calcular")
ax3.set_ylim(0, max_ms2)
ax4.set_ylabel("GFLOPS/segundo")
ax4.set_title("Desempenho")

ax5.set_ylabel("ms")
ax5.set_title("Tempo Para Calcular")
ax5.set_ylim(0, max_ms3)
ax6.set_ylabel("GFLOPS/segundo")
ax6.set_title("Desempenho")

ax7.set_ylabel("ms")
ax7.set_title("Tempo Para Calcular")
ax7.set_ylim(0, max_ms4)
ax8.set_ylabel("GFLOPS/segundo")
ax8.set_title("Desempenho")

otimizations = {
    "without":  ["simple",          "transpose",          "simd_manual",          "avx256",          "avx512"],
    "unroll":   ["simple_unroll",   "transpose_unroll",   "simd_manual_unroll",   "avx256_unroll",   "avx512_unroll"],
    "blocking": ["simple_blocking", "transpose_blocking", "simd_manual_blocking", "avx256_blocking", "avx512_blocking"],
    "parallel": ["simple_parallel", "transpose_parallel", "simd_manual_parallel", "avx256_parallel", "avx512_parallel"],
}

for key, average in algs.items():
    x = [d["size"] for d in average]
    y1 = [d["ms"] for d in average]
    y2 = [d["float"] for d in average]

    if key in otimizations["without"]:
        ax1.plot(x, y1, label=key)
        ax2.plot(x, y2, label=key)

    if key in otimizations["unroll"]:
        ax3.plot(x, y1, label=key)
        ax4.plot(x, y2, label=key)

    if key in otimizations["blocking"]:
        ax5.plot(x, y1, label=key)
        ax6.plot(x, y2, label=key)

    if key in otimizations["parallel"]:
        ax7.plot(x, y1, label=key)
        ax8.plot(x, y2, label=key)

ax1.legend()
ax3.legend()
ax5.legend()
ax7.legend()

fig_path_prefix = file_path.replace(".csv", "")
fig1.savefig(fig_path_prefix+".without.png")
fig2.savefig(fig_path_prefix+".unroll.png")
fig3.savefig(fig_path_prefix+".blocking.png")
fig4.savefig(fig_path_prefix+".parallel.png")

# plt.show()
