.PHONY: all
all: dgemm_simple dgemm_transpose dgemm_transpose_unroll dgemm_avx_disable dgemm_avx256  dgemm_avx256_unroll

.PHONY: prepare
prepare:
	mkdir -p ./out
	mkdir -p ./out_csv/raw

.PHONY: dgemm_simple
dgemm_simple: prepare
	gcc -O3 -o out/dgemm_simple src/dgemm_simple.c src/debug.c -lm

.PHONY: dgemm_transpose
dgemm_transpose: prepare
	gcc -O3 -o out/dgemm_transpose src/dgemm_transpose.c src/debug.c -lm

.PHONY: dgemm_transpose_unroll
dgemm_transpose_unroll: 
	gcc -O3 -mavx2 -DUNROLL=2  -o out/dgemm_transpose_unroll_02 src/dgemm_transpose_unroll.c src/debug.c -lm
	gcc -O3 -mavx2 -DUNROLL=4  -o out/dgemm_transpose_unroll_04 src/dgemm_transpose_unroll.c src/debug.c -lm
	gcc -O3 -mavx2 -DUNROLL=8  -o out/dgemm_transpose_unroll_08 src/dgemm_transpose_unroll.c src/debug.c -lm
	gcc -O3 -mavx2 -DUNROLL=16 -o out/dgemm_transpose_unroll_16 src/dgemm_transpose_unroll.c src/debug.c -lm
	gcc -O3 -mavx2 -DUNROLL=32 -o out/dgemm_transpose_unroll_32 src/dgemm_transpose_unroll.c src/debug.c -lm

.PHONY: dgemm_avx_disable
dgemm_avx_disable: prepare
	gcc -O3 -o out/dgemm_avx_disable src/dgemm_avx.c src/debug.c -lm

.PHONY: dgemm_avx256
dgemm_avx256: prepare
	gcc -O3 -mavx2 -o out/dgemm_avx256 src/dgemm_avx.c src/debug.c -lm

.PHONY: dgemm_avx256_unroll
dgemm_avx256_unroll: 
	gcc -O3 -mavx2 -DUNROLL=2  -o out/dgemm_avx256_unroll_02 src/dgemm_avx256_unroll.c src/debug.c -lm
	gcc -O3 -mavx2 -DUNROLL=4  -o out/dgemm_avx256_unroll_04 src/dgemm_avx256_unroll.c src/debug.c -lm
	gcc -O3 -mavx2 -DUNROLL=8  -o out/dgemm_avx256_unroll_08 src/dgemm_avx256_unroll.c src/debug.c -lm
	gcc -O3 -mavx2 -DUNROLL=16 -o out/dgemm_avx256_unroll_16 src/dgemm_avx256_unroll.c src/debug.c -lm
	gcc -O3 -mavx2 -DUNROLL=32 -o out/dgemm_avx256_unroll_32 src/dgemm_avx256_unroll.c src/debug.c -lm

.PHONY: dgemm_avx256_unroll_blocking
dgemm_avx256_unroll_blocking: 
	gcc -O3 -mavx2 -DUNROLL=8 -DBLOCK_SIZE=32  -o out/dgemm_avx256_unroll_blocking_032 src/dgemm_avx256_unroll_blocking.c src/debug.c -lm
	gcc -O3 -mavx2 -DUNROLL=8 -DBLOCK_SIZE=64  -o out/dgemm_avx256_unroll_blocking_064 src/dgemm_avx256_unroll_blocking.c src/debug.c -lm
	gcc -O3 -mavx2 -DUNROLL=8 -DBLOCK_SIZE=128 -o out/dgemm_avx256_unroll_blocking_128 src/dgemm_avx256_unroll_blocking.c src/debug.c -lm
	gcc -O3 -mavx2 -DUNROLL=8 -DBLOCK_SIZE=256 -o out/dgemm_avx256_unroll_blocking_256 src/dgemm_avx256_unroll_blocking.c src/debug.c -lm
	gcc -O3 -mavx2 -DUNROLL=8 -DBLOCK_SIZE=512 -o out/dgemm_avx256_unroll_blocking_512 src/dgemm_avx256_unroll_blocking.c src/debug.c -lm

.PHONY: dgemm_avx512
dgemm_avx512: prepare
	gcc -O3 -mavx512f -o out/dgemm_avx512 src/dgemm_avx512.c src/debug.c -lm

.PHONY: clean
clean:
	rm -fr ./out/*
	rm -fr ./out_csv/*
	rm -fr ./out_csv/raw/*

.PHONY: csv
csv: clean all
	exec scripts/csv.sh ./out
