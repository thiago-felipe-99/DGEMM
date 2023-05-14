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
	gcc -O3 -o out/dgemm_simple dgemm_simple.c
	./exec.sh ./out/dgemm_simple

.PHONY: dgemm_transpose
dgemm_transpose: prepare
	gcc -O3 -o out/dgemm_transpose dgemm_transpose.c
	./exec.sh ./out/dgemm_transpose

.PHONY: dgemm_avx256
dgemm_avx256: prepare
	gcc -O3 -mavx2 -o out/dgemm_avx256 dgemm_avx256.c
	./exec.sh ./out/dgemm_avx256

.PHONY: dgemm_avx256_unroll_2
dgemm_avx256_unroll_2: prepare
	gcc -O3 -mavx2 -DUNROLL=2 -o out/dgemm_avx256_unroll_2 dgemm_avx256_unroll.c
	./exec.sh ./out/dgemm_avx256_unroll_2

.PHONY: dgemm_avx256_unroll_4
dgemm_avx256_unroll_4: prepare
	gcc -O3 -mavx2 -DUNROLL=4 -o out/dgemm_avx256_unroll_4 dgemm_avx256_unroll.c
	./exec.sh ./out/dgemm_avx256_unroll_4

.PHONY: dgemm_avx256_unroll_8
dgemm_avx256_unroll_8: prepare
	gcc -O3 -mavx2 -DUNROLL=8 -o out/dgemm_avx256_unroll_8 dgemm_avx256_unroll.c
	./exec.sh ./out/dgemm_avx256_unroll_8

.PHONY: dgemm_avx256_unroll_16
dgemm_avx256_unroll_16: prepare
	gcc -O3 -mavx2 -DUNROLL=16 -o out/dgemm_avx256_unroll_16 dgemm_avx256_unroll.c
	./exec.sh ./out/dgemm_avx256_unroll_16

.PHONY: dgemm_avx256_unroll_32
dgemm_avx256_unroll_32: prepare
	gcc -O3 -mavx2 -DUNROLL=32 -o out/dgemm_avx256_unroll_32 dgemm_avx256_unroll.c
	./exec.sh ./out/dgemm_avx256_unroll_32

.PHONY: dgemm_avx256_unroll
dgemm_avx256_unroll: dgemm_avx256_unroll_2 dgemm_avx256_unroll_4 dgemm_avx256_unroll_8 dgemm_avx256_unroll_16 dgemm_avx256_unroll_32

.PHONY: dgemm_avx512
dgemm_avx512: prepare
	gcc -O3 -mavx512f -o out/dgemm_avx512 dgemm_avx512.c
	./exec.sh ./out/dgemm_avx512

.PHONY: clean
clean:
	rm -r ./out/*
