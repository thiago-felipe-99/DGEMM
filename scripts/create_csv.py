#!/usr/bin/env python
import subprocess

import cpuinfo

info = cpuinfo.get_cpu_info()
AVX256_ENABLE = 'avx2' in info['flags'] or 'avx' in info['flags']
AVX512_ENABLE = 'avx512' in info['flags']


def create_command_build(name="dgemm", unroll=8, block_size=32):
    return ["gcc", "-O3", "-fopenmp", "-march=native", "src/main.c",
            "src/dgemm.c", "-o", name, "-DUNROLL="+str(unroll),
            "-DBLOCK_SIZE="+str(block_size), "-lm"]


def run_test(name="./out/dgemm", algs="simple", loop="32:1024:32", parallel=False):
    command = [name, "-d", algs, "-o", loop]
    if parallel:
        command += ["-p"]

    result = subprocess.run(
        command, capture_output=True, text=True, check=True)

    averages = {}

    for line in result.stdout.splitlines():
        columns = line.strip().split(',')
        alg_name = columns[0]
        averages.setdefault(
            alg_name, {"float_sum": 0, "count": 0, "float_avg": 0})
        averages[alg_name]["float_sum"] += float(columns[3])
        averages[alg_name]["count"] += 1

    for value in averages.values():
        value["float_avg"] = value["float_sum"] / value["count"]

    return averages


def achar_melhor_unroll():
    algs = "simple_unroll,transpose_unroll,simd_manual_unroll"
    loop = "32:512:32"

    if AVX256_ENABLE:
        algs += ',avx256_unroll'

    if AVX512_ENABLE:
        algs += ',avx512_unroll'

    melhores_unrolls = {}

    for i in range(5):
        print(f"inicializando amostra: {i}")
        min_unroll = 2
        max_unroll = 8
        current = 8
        current_result = {}

        while (max_unroll - min_unroll) > 1:
            print(f"inicializando teste para unroll={current}")
            name = "./out/dgemm_unroll" + str(current)
            subprocess.run(create_command_build(
                name=name, unroll=current, block_size=current*8), check=True)

            result = run_test(name, algs, loop, True)

            sum_float = 0
            count = 0
            for key, value in result.items():
                float_avg = value["float_avg"]
                current_float = current_result.get(
                    key, {}).get("float_avg", float_avg)
                sum_float += float_avg / current_float
                count += 1

            if sum_float / count >= 1:
                current_result = result
                min_unroll = current
                max_unroll = current*2
                current = current*2
            else:
                current = (max_unroll - min_unroll)//2 + min_unroll
                current = current - current % 2
                max_unroll = current

        melhores_unrolls.setdefault(current, 0)
        melhores_unrolls[current] += 1

    return max(melhores_unrolls, key=melhores_unrolls.get)


UNROLL = achar_melhor_unroll()
print(f"Melhor UNROLL {UNROLL}")
print("Binary built successfully!")
