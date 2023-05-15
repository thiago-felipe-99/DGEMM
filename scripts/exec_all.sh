#!/usr/bin/env bash

./exec.sh ./out/dgemm_simple
./exec.sh ./out/dgemm_transpose
./exec.sh ./out/dgemm_transpose_unroll_2
./exec.sh ./out/dgemm_transpose_unroll_4
./exec.sh ./out/dgemm_transpose_unroll_8
./exec.sh ./out/dgemm_transpose_unroll_16
./exec.sh ./out/dgemm_transpose_unroll_32
./exec.sh ./out/dgemm_avx256
./exec.sh ./out/dgemm_avx256_unroll_2
./exec.sh ./out/dgemm_avx256_unroll_4
./exec.sh ./out/dgemm_avx256_unroll_8
./exec.sh ./out/dgemm_avx256_unroll_16
./exec.sh ./out/dgemm_avx256_unroll_32
