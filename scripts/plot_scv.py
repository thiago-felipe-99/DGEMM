#!/usr/bin/env python
import sys

import matplotlib.pyplot as plt


def calculate_line_averages(file_paths):
    averages = {}
    algs = {}

    for file_path in file_paths:
        with open(file_path, 'r') as file:
            for line in file:
                columns = line.strip().split(',')
                alg_name = columns[0]
                alg_size = columns[1]
                averages.setdefault(alg_name, {})
                averages[alg_name].setdefault(
                    alg_size, {"ms_sum": 0, "float_sum": 0, "count": 0})
                averages[alg_name][alg_size]["ms_sum"] += float(
                    columns[2])
                averages[alg_name][alg_size]["float_sum"] += float(
                    columns[3])
                averages[alg_name][alg_size]["count"] += 1

    for alg, info in averages.items():
        algs.setdefault(alg, [])
        for size, info in info.items():
            ms_avg = info["ms_sum"] / info["count"]
            float_avg = info["float_sum"] / info["count"]
            algs[alg] += [{"size": int(size),
                           "ms_avg": ms_avg, "float_avg": float_avg}]

    return algs


file_paths = sys.argv[1:]
averages = calculate_line_averages(file_paths)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

ax1.set_ylabel('ms')
ax1.set_title('Tempo Para Calcular')

ax2.set_ylabel('GFLOPS/segundo')
ax2.set_title('Performace')

for key, average in averages.items():
    x = [d['size'] for d in average]
    y1 = [d['ms_avg'] for d in average]
    y2 = [d['float_avg'] for d in average]
    ax1.plot(x, y1, label=key)
    ax2.plot(x, y2, label=key)

ax1.legend()
ax2.legend()
plt.show()
