#!/usr/bin/env python
# pylint: disable=C0111
import argparse
import datetime
import subprocess
import time

import cpuinfo
from cpuinfo.cpuinfo import multiprocessing

info = cpuinfo.get_cpu_info()
AVX256_ENABLE = "avx2" in info["flags"] or "avx" in info["flags"]
AVX512_ENABLE = "avx512" in info["flags"]

NUM_PROCESS = 1
NUM_BUILDS = 10

DEFAULT_NAME = "./out/name"
DEFAULT_LOOP = "32:2048:32"
DEFAULT_UNROLL = 8
DEFAULT_BLOCK_SIZE = 32
AVX256_QT_DOUBLE = 4
AVX512_QT_DOUBLE = 8
SIMD_MANUAL_QT_DOUBLE = 8

RANGE_ROUNDS = 5
MAX_ROUNDS = 3
MIN_DIFF = 1

ERROR_MESSAGE = 1
INFO_MESSAGE = 3
STATUS_MESSAGE = 4
DEBUG_MESSAGE = 5
MAX_LOG_LEVEL = INFO_MESSAGE


def log(message, level=DEBUG_MESSAGE):
    if MAX_LOG_LEVEL >= level:
        formatted_time = datetime.datetime.now().strftime("%d/%m/%Y-%H:%M:%S")
        print(f"{formatted_time} - {message}")


def criar_build(name=DEFAULT_NAME, unroll=DEFAULT_UNROLL,
                block_size=DEFAULT_BLOCK_SIZE):
    command = ["gcc", "-O3", "-fopenmp", "-march=native", "src/main.c",
               "src/dgemm.c", "-o", name, "-DUNROLL="+str(unroll),
               "-DBLOCK_SIZE="+str(block_size), "-lm"]

    subprocess.run(command, check=True)


def rodar_amostra(name=DEFAULT_NAME, algs="simple"):
    command = [name, "-d", algs, "-o", "32:512:32", "-p"]

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


def rodar_teste(name=DEFAULT_NAME, algs="simple"):
    command = [name, "-d", algs, "-o", DEFAULT_LOOP, "-p"]

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

    if AVX256_ENABLE:
        algs += ",avx256_unroll"

    if AVX512_ENABLE:
        algs += ",avx512_unroll"

    melhores_unrolls = {}

    for i in range(RANGE_ROUNDS):
        log(f"inicializando amostra de unroll, rounding: {i}", STATUS_MESSAGE)

        min_unroll = 2
        max_unroll = 4
        current = 2
        current_result = {}

        while (max_unroll - min_unroll) > MIN_DIFF:
            log(f"inicializando amostra para unroll: {current}")

            name = "./out/dgemm_unroll" + str(current)

            criar_build(name, current, current*AVX512_QT_DOUBLE)

            result = rodar_amostra(name, algs)

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

    if AVX256_ENABLE:
        algs += ",avx256_blocking"
    if AVX512_ENABLE:
        algs += ",avx512_blocking"

    melhores_block_size = {}

    for i in range(RANGE_ROUNDS):
        log(
            f"inicializando amostra de block size, rounding: {i}", STATUS_MESSAGE)

        min_block_size = unroll
        current = unroll
        max_block_size = unroll*2

        if AVX512_ENABLE:
            min_block_size = current * AVX512_QT_DOUBLE
            current = current * AVX512_QT_DOUBLE
            max_block_size = unroll * AVX512_QT_DOUBLE * 2
        elif AVX256_ENABLE:
            min_block_size = current * AVX256_QT_DOUBLE
            current = current * AVX256_QT_DOUBLE
            max_block_size = unroll * AVX256_QT_DOUBLE * 2
        else:
            min_block_size = current * SIMD_MANUAL_QT_DOUBLE
            current = current * SIMD_MANUAL_QT_DOUBLE
            max_block_size = current * SIMD_MANUAL_QT_DOUBLE * 2

        current_result = {}
        already_pass = {current: 0}

        while (max_block_size - min_block_size) > MIN_DIFF or \
                already_pass[current] > MAX_ROUNDS:
            log(f"inicializando amostra para blocksize: {current}")

            name = "./out/dgemm_unroll" + str(current)

            criar_build(name=name, unroll=unroll, block_size=current)

            result = rodar_amostra(name, algs)

            sum_float = 0
            count = 0
            for key, value in result.items():
                gflops_avg = value["gflops_avg"]
                current_float = current_result.get(key, {})\
                    .get("gflops_avg", gflops_avg)
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
                    current = current - current % (AVX512_QT_DOUBLE * unroll)
                elif AVX256_ENABLE:
                    current = current - current % (AVX256_QT_DOUBLE * unroll)
                else:
                    current = current - \
                        current % (SIMD_MANUAL_QT_DOUBLE * unroll)

                max_block_size = current

                already_pass.setdefault(current, 0)
                already_pass[current] += 1

        melhores_block_size.setdefault(current, 0)
        melhores_block_size[current] += 1

    return max(melhores_block_size, key=melhores_block_size.get)


def rodar_dgemm(indice, unroll=DEFAULT_UNROLL, block_size=DEFAULT_BLOCK_SIZE):

    algs = "simple,transpose,simd_manual,simple_unroll,transpose_unroll,simd_manual_unroll,simple_blocking,transpose_blocking,simd_manual_blocking,simple_parallel,transpose_parallel,simd_manual_parallel"

    if AVX256_ENABLE:
        algs += ",avx256,avx256_unroll,avx256_blocking,avx256_parallel"

    if AVX512_ENABLE:
        algs += ",avx512,avx512_unroll,avx512_blocking,avx512_parallel"

    if indice < NUM_PROCESS:
        time.sleep(indice*120)

    log(f"inicializando build, rounding: {indice}", INFO_MESSAGE)

    start_time = time.time()
    name = "./out/dgemm"
    criar_build(name=name, unroll=unroll, block_size=block_size)
    result = rodar_teste(name, algs)
    elapsed_time = time.time() - start_time

    log(
        f"build finalizado, tempo de execução: {elapsed_time*1000:.0f}ms . Rounding: {indice}", INFO_MESSAGE)

    return result


def rodar_todas_dgemm(unroll=DEFAULT_UNROLL, block_size=DEFAULT_BLOCK_SIZE):
    dgemms = [(i, unroll, block_size) for i in range(NUM_BUILDS)]
    results = []
    final = {}

    log(f"Rodando DGEMM com UNROLL={unroll} e BLOCK_SIZE={block_size}")

    with multiprocessing.Pool(processes=NUM_PROCESS) as pool:
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

    saida = ""
    for alg_name, alg in final.items():
        for line_number, line in alg.items():
            gflops_avg = line["gflops_avg"]
            ms_avg = line["ms_avg"]
            saida += f"{alg_name},{line_number},{ms_avg},{gflops_avg}\n"

    with open("./out_csv/dgemm.csv", "w") as file:
        file.write(saida)


parser = argparse.ArgumentParser()
parser.add_argument("-l", "--log", type=int, help="Especificar nível do log")
parser.add_argument("-u", "--unroll", type=int, help="Especificar UNROLL")
parser.add_argument("-b", "--block-size", type=int,
                    help="Especificar BLOCK_SIZE")

argmunets = parser.parse_args()
if argmunets.log:
    MAX_LOG_LEVEL = argmunets.log

if argmunets.unroll:
    UNROLL = argmunets.unroll
else:
    log("Achando melhor UNROLL", INFO_MESSAGE)
    UNROLL = achar_melhor_unroll()
    log(f"Melhor UNROLL {UNROLL}", INFO_MESSAGE)

if argmunets.block_size:
    BLOCK_SIZE = argmunets.block_size
else:
    log("Achando melhor BLOCK_SIZE", INFO_MESSAGE)
    BLOCK_SIZE = achar_melhor_block_size(UNROLL)
    log(f"Melhor BLOCK_SIZE {BLOCK_SIZE}", INFO_MESSAGE)

start_time = time.time()
rodar_todas_dgemm(UNROLL, BLOCK_SIZE)
elapsed_time = time.time() - start_time

log(f"todos builds finalizados, tempo de execução: {elapsed_time*1000:.0f}ms", INFO_MESSAGE)
