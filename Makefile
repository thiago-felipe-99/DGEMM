python:
	python main.py 16
	python main.py 32
	python main.py 64
	python main.py 128
	python main.py 256
	python main.py 512
	python main.py 1024

dgemm_simple:
	gcc -O3 -o out/dgemm_simple dgemm_simple.c
	./exec.sh ./out/dgemm_simple

dgemm_avx256:
	gcc -O3 -mavx2 -o out/dgemm_avx256 dgemm_avx256.c
	./exec.sh ./out/dgemm_avx256

dgemm_avx256_unroll_2:
	gcc -O3 -mavx2 -DUNROLL=2 -o out/dgemm_avx256_unroll_2 dgemm_avx256_unroll.c
	./exec.sh ./out/dgemm_avx256_unroll_2

dgemm_avx256_unroll_4:
	gcc -O3 -mavx2 -DUNROLL=4 -o out/dgemm_avx256_unroll_4 dgemm_avx256_unroll.c
	./exec.sh ./out/dgemm_avx256_unroll_4

dgemm_avx256_unroll_8:
	gcc -O3 -mavx2 -DUNROLL=8 -o out/dgemm_avx256_unroll_8 dgemm_avx256_unroll.c
	./exec.sh ./out/dgemm_avx256_unroll_8

dgemm_avx256_unroll_16:
	gcc -O3 -mavx2 -DUNROLL=16 -o out/dgemm_avx256_unroll_16 dgemm_avx256_unroll.c
	./exec.sh ./out/dgemm_avx256_unroll_16

dgemm_avx256_unroll_32:
	gcc -O3 -mavx2 -DUNROLL=32 -o out/dgemm_avx256_unroll_32 dgemm_avx256_unroll.c
	./exec.sh ./out/dgemm_avx256_unroll_32

dgemm_avx256_unroll_all: dgemm_avx256_unroll_2 dgemm_avx256_unroll_4 dgemm_avx256_unroll_8 dgemm_avx256_unroll_16 dgemm_avx256_unroll_32

dgemm_avx512:
	gcc -O3 -mavx512f -o out/dgemm_avx512 dgemm_avx512.c
	./exec.sh ./out/dgemm_avx512

clean:
	rm -r ./out/*
