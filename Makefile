.PHONY: all
all: dgemm_simple dgemm_transpose dgemm_transpose_unroll dgemm_avx256 dgemm_avx256_unroll 

.PHONY prepare:
prepare:
	mkdir -p ./out

.PHONY: python
python:
	python main.py 16
	python main.py 32
	python main.py 64
	python main.py 128
	python main.py 256
	python main.py 512
	python main.py 1024

.PHONY: dgemm_simple
dgemm_simple: prepare
	gcc -O3 -lm -o out/dgemm_simple dgemm_simple.c debug.c

.PHONY: dgemm_transpose
dgemm_transpose: prepare
	gcc -O3 -lm -o out/dgemm_transpose dgemm_transpose.c debug.c

.PHONY: dgemm_transpose_unroll_2
dgemm_transpose_unroll_2: prepare
	gcc -O3 -lm -mavx2 -DUNROLL=2 -o out/dgemm_transpose_unroll_2 dgemm_transpose_unroll.c debug.c

.PHONY: dgemm_transpose_unroll_4
dgemm_transpose_unroll_4: prepare
	gcc -O3 -lm -mavx2 -DUNROLL=4 -o out/dgemm_transpose_unroll_4 dgemm_transpose_unroll.c debug.c

.PHONY: dgemm_transpose_unroll_8
dgemm_transpose_unroll_8: prepare
	gcc -O3 -lm -mavx2 -DUNROLL=8 -o out/dgemm_transpose_unroll_8 dgemm_transpose_unroll.c debug.c

.PHONY: dgemm_transpose_unroll_16
dgemm_transpose_unroll_16: prepare
	gcc -O3 -lm -mavx2 -DUNROLL=16 -o out/dgemm_transpose_unroll_16 dgemm_transpose_unroll.c debug.c

.PHONY: dgemm_transpose_unroll_32
dgemm_transpose_unroll_32: prepare
	gcc -O3 -lm -mavx2 -DUNROLL=32 -o out/dgemm_transpose_unroll_32 dgemm_transpose_unroll.c debug.c

.PHONY: dgemm_transpose_unroll
dgemm_transpose_unroll: dgemm_transpose_unroll_2 dgemm_transpose_unroll_4 dgemm_transpose_unroll_8 dgemm_transpose_unroll_16 dgemm_transpose_unroll_32

.PHONY: dgemm_avx256
dgemm_avx256: prepare
	gcc -O3 -lm -mavx2 -o out/dgemm_avx256 dgemm_avx256.c debug.c

.PHONY: dgemm_avx256_unroll_2
dgemm_avx256_unroll_2: prepare
	gcc -O3 -lm -mavx2 -DUNROLL=2 -o out/dgemm_avx256_unroll_2 dgemm_avx256_unroll.c debug.c

.PHONY: dgemm_avx256_unroll_4
dgemm_avx256_unroll_4: prepare
	gcc -O3 -lm -mavx2 -DUNROLL=4 -o out/dgemm_avx256_unroll_4 dgemm_avx256_unroll.c debug.c

.PHONY: dgemm_avx256_unroll_8
dgemm_avx256_unroll_8: prepare
	gcc -O3 -lm -mavx2 -DUNROLL=8 -o out/dgemm_avx256_unroll_8 dgemm_avx256_unroll.c debug.c

.PHONY: dgemm_avx256_unroll_16
dgemm_avx256_unroll_16: prepare
	gcc -O3 -lm -mavx2 -DUNROLL=16 -o out/dgemm_avx256_unroll_16 dgemm_avx256_unroll.c debug.c

.PHONY: dgemm_avx256_unroll_32
dgemm_avx256_unroll_32: prepare
	gcc -O3 -lm -mavx2 -DUNROLL=32 -o out/dgemm_avx256_unroll_32 dgemm_avx256_unroll.c debug.c

.PHONY: dgemm_avx256_unroll
dgemm_avx256_unroll: dgemm_avx256_unroll_2 dgemm_avx256_unroll_4 dgemm_avx256_unroll_8 dgemm_avx256_unroll_16 dgemm_avx256_unroll_32

.PHONY: dgemm_avx512
dgemm_avx512: prepare
	gcc -O3 -lm -mavx512f -o out/dgemm_avx512 dgemm_avx512.c debug.c

.PHONY: clean
clean:
	rm -r ./out/*
