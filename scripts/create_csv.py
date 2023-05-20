#!/usr/bin/env python
# pylint: disable=C0111
import argparse
import datetime
import multiprocessing
import subprocess
import time

import cpuinfo

info = cpuinfo.get_cpu_info()
AVX256_ENABLE = "avx2" in info["flags"] or "avx" in info["flags"]
AVX512_ENABLE = "avx512" in info["flags"]

NUM_BUILDS = 10
AVX256_QT_DOUBLE = 4
AVX512_QT_DOUBLE = 8
SIMD_MANUAL_QT_DOUBLE = 8

DEFAULT_NAME = "./out/name"
DEFAULT_UNROLL = 8
DEFAULT_BLOCK_SIZE = 32
DEFAULT_NUM_PROCESS = 1
DEFAULT_LOOP = "32:1024:32"


ALL_ALGS = "simple,transpose,simd_manual,simple_unroll,transpose_unroll,simd_manual_unroll,simple_blocking,transpose_blocking,simd_manual_blocking,simple_parallel,transpose_parallel,simd_manual_parallel"

if AVX256_ENABLE:
    ALL_ALGS += ",avx256,avx256_unroll,avx256_blocking,avx256_parallel"

if AVX512_ENABLE:
    ALL_ALGS += ",avx512,avx512_unroll,avx512_blocking,avx512_parallel"

RANGE_ROUNDS = 5
MAX_ROUNDS = 3
MIN_DIFF = 1

ERROR_MESSAGE = 1
INFO_MESSAGE = 3
STATUS_MESSAGE = 4
DEBUG_MESSAGE = 5
DEFAULT_LOG_LEVEL = INFO_MESSAGE
MAX_LOG_LEVEL = INFO_MESSAGE


def log(message, level=DEBUG_MESSAGE):
    if MAX_LOG_LEVEL >= level:
        formatted_time = datetime.datetime.now().strftime("%d/%m/%Y-%H:%M:%S")
        print(f"{formatted_time} - {message}")


def criar_build(name, unroll, block_size):
    command = ["gcc", "-O3", "-fopenmp", "-march=native", "src/main.c",
               "src/dgemm.c", "-o", name, "-DUNROLL="+str(unroll),
               "-DBLOCK_SIZE="+str(block_size), "-lm"]

    subprocess.run(command, check=True)


def rodar_amostra(name, algs):
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


def rodar_build(name, algs, loop):
    command = [name, "-d", algs, "-o", loop, "-p"]

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


def achar_melhor_unroll(algs):
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


def achar_melhor_block_size(unroll, algs):
    melhores_block_size = {}

    for i in range(RANGE_ROUNDS):
        log(
            f"inicializando amostra de block size, rounding: {i}",
            STATUS_MESSAGE
        )

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


def rodar_dgemm_thread(indice, num_process, algs, unroll, block_size, loop):

    if indice < num_process:
        time.sleep(indice*120)

    log(f"build {indice} inicializado", INFO_MESSAGE)

    start_time = time.time()
    name = "./out/dgemm"
    criar_build(name=name, unroll=unroll, block_size=block_size)
    result = rodar_build(name, algs, loop)
    diff_time = (time.time() - start_time)*1000

    log(
        f"build {indice} finalizado, tempo de execução: {diff_time:.0f}ms",
        INFO_MESSAGE
    )

    return result


def rodar_todas_dgemm(algs, unroll, block_size, loop, num_process, filename):
    dgemms = [(i, num_process, algs, unroll, block_size, loop)
              for i in range(NUM_BUILDS)]
    csvs = []
    result = {}

    log(f"Rodando DGEMM com UNROLL={unroll} e BLOCK_SIZE={block_size}")

    with multiprocessing.Pool(processes=num_process) as pool:
        workes = [pool.apply_async(rodar_dgemm_thread, dgemm)
                  for dgemm in dgemms]
        csvs = [result.get() for result in workes]

    for csv in csvs:
        for alg_name, averages in csv.items():
            result.setdefault(alg_name, {})
            for line, values in averages.items():
                result[alg_name].setdefault(line, {
                    "gflops_sum": 0, "ms_sum": 0, "count": 0
                })
                result[alg_name][line]["gflops_sum"] += values["gflops_sum"]
                result[alg_name][line]["ms_sum"] += values["ms_sum"]
                result[alg_name][line]["count"] += 1

    saida = ""
    for alg_name, alg in result.items():
        for line_number, line in alg.items():
            gflops_avg = line["gflops_sum"] / line["count"]
            ms_avg = line["ms_sum"] / line["count"]
            saida += f"{alg_name},{line_number},{ms_avg},{gflops_avg}\n"

    with open(filename, "w", encoding="utf-8") as file:
        file.write(saida)


class CustomHelpFormatter(argparse.HelpFormatter):
    def add_argument(self, action):
        if action.option_strings == ["-h", "--help"]:
            action.help = "Mostra essa mensagem de ajuda e para a execução"
        super().add_argument(action)


def main():
    parser = argparse.ArgumentParser(
        description="Faz o build e gera um csv coma média do tempo de execução de todas a DGEMM",
        formatter_class=CustomHelpFormatter
    )
    parser.add_argument("-u", "--unroll", type=int, help="Especifica UNROLL")
    parser.add_argument("-b", "--block-size", type=int,
                        help="Especifica BLOCK_SIZE")
    parser.add_argument("-m", "--loop-min", default=32, type=int,
                        help="Especifica o limite mínimo do loop")
    parser.add_argument("-M", "--loop-max", default=1024, type=int,
                        help="Especifica o limite máximo do loop")
    parser.add_argument("-s", "--loop-step", default=32, type=int,
                        help="Especifica o passo do loop")
    parser.add_argument("-p", "--process", default=DEFAULT_NUM_PROCESS,
                        type=int, help="Quantidade de processos paralelos")
    parser.add_argument("-f", "--file",  default="./out_csv/dgemm.csv",
                        type=str, help="Caminho do arquivo csv gerado")
    parser.add_argument("-l", "--log", type=int, default=DEFAULT_LOG_LEVEL,
                        help="Especifica nível do log")

    arguments = parser.parse_args()

    global MAX_LOG_LEVEL
    MAX_LOG_LEVEL = arguments.log

    if arguments.unroll:
        unroll = arguments.unroll
    else:
        algs = "simple_unroll,transpose_unroll,simd_manual_unroll"

        if AVX256_ENABLE:
            algs += ",avx256_unroll"
        if AVX512_ENABLE:
            algs += ",avx512_unroll"

        log("Achando melhor UNROLL", INFO_MESSAGE)
        unroll = achar_melhor_unroll(algs)
        log(f"Melhor UNROLL {unroll}", INFO_MESSAGE)

    if arguments.block_size:
        block_size = arguments.block_size
    else:
        algs = "simple_blocking,transpose_blocking,simd_manual_blocking"

        if AVX256_ENABLE:
            algs += ",avx256_blocking"
        if AVX512_ENABLE:
            algs += ",avx512_blocking"

        log("Achando melhor BLOCK_SIZE", INFO_MESSAGE)
        block_size = achar_melhor_block_size(unroll, algs)
        log(f"Melhor BLOCK_SIZE {block_size}", INFO_MESSAGE)

    loop = f"{arguments.loop_min}:{arguments.loop_max}:{arguments.loop_step}"
    num_process = arguments.process
    file = arguments.file

    start_time = time.time()
    rodar_todas_dgemm(ALL_ALGS, unroll, block_size, loop, num_process, file)
    diff_time = (time.time() - start_time)*1000

    log(
        f"todos builds finalizados, tempo de execução: {diff_time:.0f}ms",
        INFO_MESSAGE
    )


if __name__ == "__main__":
    main()
