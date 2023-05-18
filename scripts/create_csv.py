#!/usr/bin/env python
import time
import subprocess

import cpuinfo
from cpuinfo.cpuinfo import multiprocessing

info = cpuinfo.get_cpu_info()
AVX256_ENABLE = "avx2" in info["flags"] or "avx" in info["flags"]
AVX512_ENABLE = "avx512" in info["flags"]


def criar_comando_de_build(name="dgemm", unroll=8, block_size=32):
    return ["gcc", "-O3", "-fopenmp", "-march=native", "src/main.c",
            "src/dgemm.c", "-o", name, "-DUNROLL="+str(unroll),
            "-DBLOCK_SIZE="+str(block_size), "-lm"]


def rodar_teste_pegar_media(name="./out/dgemm", algs="simple",
                            loop="32:1024:32", parallel=False):
    command = [name, "-d", algs, "-o", loop]
    if parallel:
        command += ["-p"]

    result = subprocess.run(
        command, capture_output=True, text=True, check=True)

    averages = {}

    for line in result.stdout.splitlines():
        columns = line.strip().split(",")
        alg_name = columns[0]
        averages.setdefault(alg_name, {
            "gflops_sum": 0, "gflops_avg": 0,
            "ms_sum": 0, "ms_avg": 0, "count": 0
        })
        averages[alg_name]["ms_sum"] += float(columns[2])
        averages[alg_name]["gflops_sum"] += float(columns[3])
        averages[alg_name]["count"] += 1

    for value in averages.values():
        value["gflops_avg"] = value["gflops_sum"] / value["count"]
        value["ms_avg"] = value["ms_sum"] / value["count"]

    return averages


def rodar_teste(name="./out/dgemm", algs="simple",
                     loop="32:1024:32", parallel=False):
    command = [name, "-d", algs, "-o", loop]
    if parallel:
        command += ["-p"]

    result = subprocess.run(
        command, capture_output=True, text=True, check=True)

    averages = {}

    for line in result.stdout.splitlines():
        columns = line.strip().split(",")
        alg_name = columns[0]
        alg_size = columns[1]
        averages.setdefault(alg_name, {})
        averages[alg_name].setdefault(alg_size, {"gflops_sum": 0, "ms_sum": 0})
        averages[alg_name][alg_size]["ms_sum"] = float(columns[2])
        averages[alg_name][alg_size]["gflops_sum"] = float(columns[3])

    return averages


def achar_melhor_unroll():
    algs = "simple_unroll,transpose_unroll,simd_manual_unroll"
    loop = "32:512:32"

    if AVX256_ENABLE:
        algs += ",avx256_unroll"

    if AVX512_ENABLE:
        algs += ",avx512_unroll"

    melhores_unrolls = {}

    for i in range(5):
        print(f"inicializando amostra de unroll: {i}")
        min_unroll = 2
        max_unroll = 8
        current = 8
        current_result = {}

        while (max_unroll - min_unroll) > 1:
            print(f"inicializando teste para unroll: {current}")
            name = "./out/dgemm_unroll" + str(current)
            subprocess.run(criar_comando_de_build(
                name=name, unroll=current, block_size=current*8), check=True)

            result = rodar_teste_pegar_media(name, algs, loop, True)

            sum_float = 0
            count = 0
            for key, value in result.items():
                gflops_avg = value["gflops_avg"]
                current_float = current_result.get(
                    key, {}).get("gflops_avg", gflops_avg)
                sum_float += gflops_avg / current_float
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


def achar_melhor_block_size(unroll):
    algs = "simple_blocking,transpose_blocking,simd_manual_blocking"
    loop = "32:512:32"

    if AVX256_ENABLE:
        algs += ",avx256_blocking"
    if AVX512_ENABLE:
        algs += ",avx512_blocking"

    melhores_block_size = {}

    for i in range(5):
        print(f"inicializando amostra de block size: {i}")
        min_block_size = unroll
        max_block_size = unroll
        current = unroll
        if AVX512_ENABLE:
            current = current * 8
            max_block_size = unroll * 8
        else:
            current = current * 4
            max_block_size = unroll * 4

        current_result = {}
        already_pass = {current: 0}

        while (max_block_size - min_block_size) > 1 or already_pass[current] > 3:
            print(f"inicializando teste para blocksize: {current}")
            name = "./out/dgemm_unroll" + str(current)
            subprocess.run(criar_comando_de_build(
                name=name, unroll=unroll, block_size=current), check=True)

            result = rodar_teste_pegar_media(name, algs, loop, True)

            sum_float = 0
            count = 0
            for key, value in result.items():
                gflops_avg = value["gflops_avg"]
                current_float = current_result.get(
                    key, {}).get("gflops_avg", gflops_avg)
                sum_float += gflops_avg / current_float
                count += 1

            if sum_float / count >= 1:
                current_result = result
                min_block_size = current
                max_block_size = current*2
                current = current*2

                already_pass.setdefault(current, 0)
                already_pass[current] += 1
            else:
                current = (max_block_size - min_block_size)//2 + min_block_size

                if AVX512_ENABLE:
                    current = current - current % (8 * unroll)
                else:
                    current = current - current % (4 * unroll)

                max_block_size = current

                already_pass.setdefault(current, 0)
                already_pass[current] += 1

        melhores_block_size.setdefault(current, 0)
        melhores_block_size[current] += 1

    return max(melhores_block_size, key=melhores_block_size.get)


def rodar_dgemm(indice, unroll=8, block_size=32):
    print(f"inicializando teste: {indice}")

    loop = "32:512:32"
    algs = "simple,transpose,simd_manual,simple_unroll,transpose_unroll,simd_manual_unroll,simple_blocking,transpose_blocking,simd_manual_blocking,simple_parallel,transpose_parallel,simd_manual_parallel"

    if AVX256_ENABLE:
        algs += ",avx256,avx256_unroll,avx256_blocking,avx256_parallel"

    if AVX512_ENABLE:
        algs += ",avx512,avx512_unroll,avx512_blocking,avx512_parallel"

    time.sleep(indice)

    name = "./out/dgemm"
    subprocess.run(criar_comando_de_build(
        name=name, unroll=unroll, block_size=block_size), check=True)

    return rodar_teste(name, algs, loop, True)


def rodar_todas_dgemm(unroll=8, block_size=32):
    dgemms = [(i, unroll, block_size) for i in range(5)]
    results = []
    final = {}

    with multiprocessing.Pool() as pool:
        workes = [pool.apply_async(rodar_dgemm, dgemm) for dgemm in dgemms]
        results = [result.get() for result in workes]

    for result in results:
        for alg_name, averages in result.items():
            final.setdefault(alg_name, {})
            for line, values in averages.items():
                final[alg_name].setdefault(line, {
                    "gflops_sum": 0, "gflops_avg": 0,
                    "ms_sum": 0, "ms_avg": 0, "count": 0
                })
                final[alg_name][line]["gflops_sum"] += values["gflops_sum"]
                final[alg_name][line]["ms_sum"] += values["ms_sum"]
                final[alg_name][line]["count"] += 1

    for alg in final.values():
        for line in alg.values():
            line["gflops_avg"] = line["gflops_sum"] / line["count"]
            line["ms_avg"] = line["ms_sum"] / line["count"]

    results = ""
    for alg_name, alg in final.items():
        for line_number, line in alg.items():
            gflops_avg = line["gflops_avg"]
            ms_avg = line["ms_avg"]
            results += f"{alg_name},{line_number},{ms_avg},{gflops_avg}\n"

    with open("./out_csv//dgemm.csv", "w") as file:
        file.write(results)


UNROLL = achar_melhor_unroll()
print(f"Melhor UNROLL {UNROLL}")
BLOCK_SIZE = achar_melhor_block_size(UNROLL)
print(f"Melhor BLOCK_SIZE {BLOCK_SIZE}")
rodar_todas_dgemm(UNROLL, BLOCK_SIZE)
